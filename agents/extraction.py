"""Agent 4 - Field Extraction: document-type-specific fields, each linked to page + bounding boxes."""
import re

import numpy as np

from .common import make_field, parse_amount, parse_date

AMT = r"([\d,]+(?:\.\d{1,2})?)"
CUR = r"(?:Rs\.?|INR|USD|₹|\$|€|£)?"
LABEL_RX = re.compile(r"^(bill(ed)?\s*to|ship(ped)?\s*to|sold\s*to|invoice(\s*(date|no\.?|number|#))?|due\s*date|date|description|qty|quantity|"
                      r"unit\s*price|rate|amount|total|subtotal|p\.?o\.?\s*#?|name|address|from|to|tax|notes?)\s*[:#]?$", re.I)
DATE = r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})"


class Doc:
    """Regex search over OCR text that maps every match back to word boxes (the source-link mechanism)."""

    def __init__(self, ocr, images):
        self.ocr, self.images = ocr, images

    def search(self, pattern, group=1, flags=re.I | re.M):
        hits = []
        for pg in self.ocr["pages"]:
            for m in re.finditer(pattern, pg["text"], flags):
                s, e = m.span(group)
                if e <= s:
                    continue
                ws = [w for w in pg["words"] if w["start"] < e and w["end"] > s]
                if not ws:
                    continue
                bysegment = {}
                for w in ws:
                    bysegment.setdefault(w["seg"], []).append(w["bbox"])
                boxes = [[min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
                         for bs in bysegment.values()]
                bb = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                      max(b[3] for b in boxes)]
                hits.append({"full": m.group(0), "value": m.group(group).strip(), "page": pg["page"], "boxes": boxes, "bbox": bb,
                             "conf": float(np.mean([w["conf"] for w in ws])), "span": (s, e)})
        return hits

    def first(self, pattern, **kw):
        h = self.search(pattern, **kw)
        return h[0] if h else None

    def last(self, pattern, **kw):
        h = self.search(pattern, **kw)
        return h[-1] if h else None


def _text_field(hit, factor=1.0):
    """A value that is itself a form label (two-column layouts) is NOT a value: report missing instead of guessing."""
    if not hit or LABEL_RX.match(hit["value"].strip()):
        return None
    return make_field(hit["value"], hit, factor=factor)


def _date_field(hit):
    if not hit:
        return None
    d = parse_date(hit["value"])
    f = make_field(d.isoformat() if d else hit["value"], hit, raw=hit["value"])
    if not d:
        f["parse_failed"] = True
    return f


def _amount_field(hit):
    if not hit:
        return None
    return make_field(parse_amount(hit["value"]), hit, raw=hit["value"])


def has_ink(img, bbox, pad_above=32, reach=450):
    """Signature check: ink just above the label's baseline, extending to its right (where people sign)."""
    x0, y0, x1, y1 = bbox
    reg = img[max(0, y0 - pad_above): max(1, y0 - 3), x0: min(img.shape[1], x1 + reach)]
    return reg.size > 0 and float((reg < 165).mean()) > 0.004


def signature_fields(doc, label_pattern):
    sigs = []
    for h in doc.search(label_pattern, group=0):
        img = doc.images[h["page"] - 1]
        signed = has_ink(img, h["bbox"])
        f = make_field("signed" if signed else "missing", h, factor=0.9, method="ink-detection")
        f["label"] = h["value"]
        sigs.append(f)
    return sigs


# ---------------------------------------------------------------- invoice
def extract_invoice(doc):
    F = {}
    F["invoice_no"] = _text_field(doc.first(r"Invoice\s*(?:No|Number|#)\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]+)"))
    F["invoice_date"] = _date_field(doc.first(r"Invoice\s*Date\s*[:\-]?\s*" + DATE))
    F["due_date"] = _date_field(doc.first(r"Due\s*Date\s*[:\-]?\s*" + DATE))
    # PO number may wrap onto a second OCR segment ("PO" / "Number: PO-2291")
    F["po_number"] = _text_field(doc.first(r"(?:P\.?O\.?|Purchase\s*Order)\s*(?:Number|No\.?)?\s*[:\-]\s*([A-Z0-9][A-Z0-9\-/]+)")
                                 or doc.first(r"Number\s*:\s*(PO[A-Z0-9\-/]+)"))
    F["gstin"] = _text_field(doc.first(r"GSTIN\s*[:\-]?\s*([A-Z0-9]{10,16})"))
    F["bill_to"] = _text_field(doc.first(r"Bill\s*To\s*[:\-]?\s*(.+)$"))
    # vendor: first prominent segment on page 1 that is not a title / reference line
    vendor = None
    p0 = doc.ocr["pages"][0]
    marks = [s["bbox"][1] for s in p0["segments"]
             if re.match(r"^(bill(ed)?|ship(ped)?)\s*to|^invoice\s*(#|no|number|date)|^date\b", s["text"].strip(), re.I)]
    limit = min(marks) if marks else int(p0["height"] * 0.3)  # vendor name sits above the billing block
    for s in p0["segments"][:8]:
        if s["bbox"][1] > limit + 20:  # allow the same row as the invoice-no label
            break
        t = s["text"]
        if re.search(r"invoice|tax|date|no[:.]|gstin|bill|page|number|po\b|\d{3,}", t, re.I):
            continue
        if LABEL_RX.match(t.strip()):
            continue
        if 3 <= len(t) <= 60 and sum(c.isalpha() for c in t) >= 3:
            vendor = {"value": t, "page": 1, "boxes": [s["bbox"]], "bbox": s["bbox"], "conf": s["conf"]}
            break
    F["vendor"] = make_field(vendor["value"], vendor, factor=0.95, method="layout-heuristic") if vendor else None
    F["subtotal"] = _amount_field(doc.first(r"Sub\s*-?\s*total\s*[:\-]?\s*" + CUR + r"\s*" + AMT))
    F["total"] = _amount_field(doc.last(r"\bTotal(?:\s*Amount|\s*Due|\s*Payable)?\s*[:\-]?\s*" + CUR + r"\s*" + AMT))
    # tax rows (kept individually so duplicated rows can be detected)
    tax_rows = []
    tax_rx = (r"^([^\n]*?\b(?:(?:C|S|I|U)?GST|VAT|Sales\s+Tax|Service\s+Tax|Tax|Cess)\b[^\n]*?)\s*[:\-]?\s*"
              + CUR + r"\s*([\d,]+\.\d{2})\s*$")
    for h in doc.search(tax_rx, group=2):
        label = re.sub(r"[\s:\-$₹€£]*[\d,]+\.\d{2}\s*$", "", h["full"]).replace("\n", " ").strip()
        if re.search(r"invoice|total\b", label, re.I) and not re.search(r"tax|gst|vat", label, re.I):
            continue
        tax_rows.append({"label": label or "Tax", "amount": parse_amount(h["value"]), "page": h["page"],
                         "boxes": h["boxes"], "bbox": h["bbox"], "confidence": round(h["conf"], 3)})
    # line items from table rows (header-anchored)
    items = []
    for pg in doc.ocr["pages"]:
        in_table = False
        for ridx in pg["rows"]:
            segs = [pg["segments"][i] for i in ridx]
            joined = " ".join(s["text"] for s in segs).lower()
            if "qty" in joined and "amount" in joined:
                in_table = True
                continue
            if re.search(r"sub\s*-?total|gst|total", joined):
                in_table = False
            if not in_table:
                continue
            nums, k = [], len(segs)
            while k > 0 and re.fullmatch(r"[\d,]+(?:\.\d+)?", segs[k - 1]["text"]):
                nums.insert(0, segs[k - 1])
                k -= 1
            if len(nums) < 2 or k == 0:
                continue
            vals = [parse_amount(n["text"]) for n in nums]
            qty = vals[0] if len(vals) >= 3 else None
            rate = vals[-2]
            boxes = [s["bbox"] for s in segs]
            items.append({"description": " ".join(s["text"] for s in segs[:k]), "qty": qty, "rate": rate,
                          "amount": vals[-1], "page": pg["page"],
                          "bbox": [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                                   max(b[3] for b in boxes)], "boxes": [[min(b[0] for b in boxes),
                                                                     min(b[1] for b in boxes),
                                                                     max(b[2] for b in boxes),
                                                                     max(b[3] for b in boxes)]],
                          "confidence": round(float(np.mean([s["conf"] for s in segs])), 3)})
    # tax_total synthesized from tax rows (editable by a reviewer)
    if tax_rows:
        F["tax_total"] = {"value": round(sum(r["amount"] for r in tax_rows), 2), "raw": None, "method": "sum-of-rows",
                          "confidence": round(min(r["confidence"] for r in tax_rows), 3), "page": tax_rows[0]["page"],
                          "bbox": tax_rows[0]["bbox"], "boxes": [b for r in tax_rows for b in r["boxes"]]}
    alltext = " ".join(p["text"] for p in doc.ocr["pages"])
    code = "INR"
    if re.search(r"\$\s?\d", alltext) and not re.search(r"₹|\bRs\b|\bINR\b|GSTIN", alltext):
        code = "USD"
    elif re.search(r"€\s?\d", alltext):
        code = "EUR"
    F["currency"] = make_field(code, None, method="classifier", conf=0.9)
    sigs = signature_fields(doc, r"^(?:Authori[sz]ed\s+Signatory|Signature)[^\n]*$")
    F["signature"] = sigs[0] if sigs else None
    return F, {"line_items": items, "tax_rows": tax_rows}


# ---------------------------------------------------------------- KYC
def _label_value(doc, label, same_line_sep=r"\s*[:\-]\s*"):
    """Value on the same segment after the label, or on the next segment (PAN-card style)."""
    h = doc.first(r"^(?:%s)%s(.+)$" % (label, same_line_sep))
    if h:
        return h
    for pg in doc.ocr["pages"]:
        segs = pg["segments"]
        for i, s in enumerate(segs[:-1]):
            if re.fullmatch(r"(?:%s)\s*[:\-]?" % label, s["text"].strip(), re.I):
                n = segs[i + 1]
                return {"value": n["text"], "page": pg["page"], "boxes": [n["bbox"]], "bbox": n["bbox"],
                        "conf": n["conf"], "span": None}
    return None


ID_PATTERNS = {
    "pan": r"\b([A-Z]{5}\d{4}[A-Z])\b",
    "aadhaar": r"(?<![\d/])(\d{4} ?\d{4} ?\d{4})(?![\d/])",
    "driving_licence": r"(?:DL\s*No\.?\s*[:\-]?\s*)?\b([A-Z]{2}\d{2}\s?\d{6,13})\b",
    "passport": r"\b([A-Z]\d{7})\b",
}


def extract_kyc(doc, subtype):
    F = {"id_type": make_field(subtype or "unknown", None, method="classifier", conf=0.9)}
    pat = ID_PATTERNS.get(subtype)
    hit = doc.first(pat, flags=re.M) if pat else None
    if subtype == "driving_licence":
        hit = doc.first(r"DL\s*No\.?\s*[:\-]?\s*([A-Z0-9 ]{8,20})$") or hit
    F["id_number"] = _text_field(hit)
    F["name"] = _text_field(_label_value(doc, r"Name"))
    F["father_name"] = _text_field(_label_value(doc, r"Father'?s?\s*Name"))
    F["dob"] = _date_field(doc.first(r"(?:Date\s*of\s*Birth|DOB)\s*[:\-]?\s*" + DATE)
                           or _label_value(doc, r"Date\s*of\s*Birth|DOB"))
    F["expiry_date"] = _date_field(doc.first(r"(?:Valid\s*(?:Till|Upto|Up\s*to)|Expiry(?:\s*Date)?|Date\s*of\s*Expiry)\s*[:\-]?\s*" + DATE))
    F["address"] = _text_field(doc.first(r"Address\s*[:\-]\s*(.+)$"))
    sigs = signature_fields(doc, r"^Signature[^\n]*$")
    F["signature"] = sigs[0] if sigs else None
    # a PAN card's "Name" label must not pick up "Father's Name" value
    if F.get("name") and F.get("father_name") and F["name"]["value"] == F["father_name"]["value"]:
        F["name"] = None
    return F, {}


# ---------------------------------------------------------------- contract
def extract_contract(doc):
    F = {}
    F["effective_date"] = _date_field(doc.first(r"made\s*on\s*" + DATE))
    parties = doc.first(r"between\s+([^\n]+(?:\n[^\n]+)?)\s*\(\s*[\"“”']?(?:Client|Buyer|Customer)", group=1)
    p2 = doc.first(r"\)\s*and\s+([^\n(]+?)\s*\(\s*[\"“”']?(?:Vendor|Supplier|Seller|Provider)")
    F["party_a"] = _text_field(parties) if parties else None
    if F["party_a"]:
        F["party_a"]["value"] = F["party_a"]["value"].replace("\n", " ")
    F["party_b"] = _text_field(p2)
    clauses = []
    for pg in doc.ocr["pages"]:
        segs = pg["segments"]
        cur = None
        for s in segs:
            m = re.match(r"^(\d+)\.\s*([A-Za-z &/]+)$", s["text"].strip())
            if m:
                cur = {"number": int(m.group(1)), "title": m.group(2).strip(), "text": "", "page": pg["page"],
                       "boxes": [s["bbox"]], "confidence": [s["conf"]]}
                clauses.append(cur)
            elif cur is not None and not re.match(r"^signed\s+for", s["text"], re.I):
                cur["text"] = (cur["text"] + " " + s["text"]).strip()
                cur["boxes"].append(s["bbox"])
                cur["confidence"].append(s["conf"])
            else:
                cur = None
    for c in clauses:
        c["confidence"] = round(float(np.mean(c["confidence"])), 3)
        b = c["boxes"]
        c["bbox"] = [min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)]
        c["boxes"] = [c["bbox"]]
    by = {c["title"].lower(): c for c in clauses}

    def from_clause(key, name, rx, conv=float):
        c = by.get(key)
        if not c:
            return None
        m = re.search(rx, c["text"], re.I)
        if not m:
            return None
        hit = {"page": c["page"], "boxes": c["boxes"], "bbox": c["bbox"], "conf": c["confidence"]}
        return make_field(conv(m.group(1)), hit, raw=m.group(0))

    F["payment_terms_days"] = from_clause("payment terms", "pt", r"within\s+(\d+)\s+days", int)
    F["termination_notice_days"] = from_clause("termination", "tn", r"(\d+)\s+days", int)
    c = by.get("liability")
    if c:
        hit = {"page": c["page"], "boxes": c["boxes"], "bbox": c["bbox"], "conf": c["confidence"]}
        F["liability_capped"] = make_field("no" if re.search(r"unlimited", c["text"], re.I) else "yes", hit,
                                           method="clause-reading")
    c = by.get("governing law")
    if c:
        m = re.search(r"laws of ([A-Za-z ]+?)\.?$", c["text"])
        hit = {"page": c["page"], "boxes": c["boxes"], "bbox": c["bbox"], "conf": c["confidence"]}
        F["governing_law"] = make_field(m.group(1).strip() if m else c["text"], hit, method="clause-reading")
    sigs = signature_fields(doc, r"^Signed\s+for[^\n]*$")
    for i, s in enumerate(sigs):
        F["signature_%d" % (i + 1)] = s
    return {k: v for k, v in F.items() if v}, {"clauses": clauses}


# ---------------------------------------------------------------- claim
def extract_claim(doc):
    F = {}
    F["claim_number"] = _text_field(doc.first(r"Claim\s*(?:No|Number)\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]+)"))
    F["policy_number"] = _text_field(doc.first(r"Policy\s*(?:No|Number)\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]+)"))
    F["claimant"] = _text_field(doc.first(r"(?:Claimant|Policyholder)(?:\s*Name)?\s*[:\-]\s*(.+)$"))
    F["date_of_loss"] = _date_field(doc.first(r"Date\s*of\s*Loss\s*[:\-]?\s*" + DATE))
    F["claim_date"] = _date_field(doc.first(r"Claim\s*Date\s*[:\-]?\s*" + DATE))
    F["claim_amount"] = _amount_field(doc.first(r"Claim(?:ed)?\s*Amount\s*[:\-]?\s*" + CUR + r"\s*" + AMT))
    F["description"] = _text_field(doc.first(r"Description\s*[:\-]\s*(.+)$"))
    sigs = signature_fields(doc, r"^(?:Claimant\s+)?Signature[^\n]*$")
    F["signature"] = sigs[0] if sigs else None
    return F, {}


def extract(doc_type, subtype, ocr, images):
    doc = Doc(ocr, images)
    if doc_type == "invoice":
        F, T = extract_invoice(doc)
    elif doc_type == "kyc":
        F, T = extract_kyc(doc, subtype)
    elif doc_type == "contract":
        F, T = extract_contract(doc)
    elif doc_type == "claim":
        F, T = extract_claim(doc)
    else:
        F, T = {}, {}
    return {k: v for k, v in F.items() if v}, T
