"""Agent 6 - Reference & Business Rule Matching: master lists, ID formats, PO check, SME rules, clause diff."""
import difflib
import operator
import re

from .common import load_json, money, parse_date, parse_amount, today
from .validation import mkflag, ref, val

_d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_p = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_ok(num):
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _d[c][_p[i % 8][int(ch)]]
    return c == 0


GSTIN_RX = r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]$"
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne}


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def _best_match(name, candidates):
    best, ratio = None, 0.0
    for c in candidates:
        r = difflib.SequenceMatcher(None, _norm(name), _norm(c["name"])).ratio()
        if r > ratio:
            best, ratio = c, r
    return best, ratio


def apply_rules(record, rules, known_records):
    dtype, F, T = record["doc_type"], record["fields"], record["tables"]
    flags, comparison = [], []
    if dtype == "invoice":
        flags += _invoice(F, rules, known_records, record["id"])
    elif dtype == "kyc":
        flags += _kyc(F, record.get("subtype"))
    elif dtype == "contract":
        f2, comparison = _contract(F, T, rules)
        flags += f2
    elif dtype == "claim":
        flags += _claim(F, rules)
    flags += _custom(dtype, F, rules)
    for f in flags:
        f["agent"] = "business_rules"
    return flags, comparison


def _invoice(F, rules, known, my_id):
    flags = []
    vendors = load_json("vendors.json")
    vname = val(F, "vendor")
    if vname:
        m, r = _best_match(vname, vendors)
        if r >= rules["vendor_match_min_ratio"]:
            g = val(F, "gstin")
            if g and m["gstin"] and g.replace(" ", "") != m["gstin"]:
                flags.append(mkflag("gstin_mismatch", "high", "gstin",
                                    "GSTIN %s differs from vendor master (%s)" % (g, m["gstin"]),
                                    "Vendor '%s' is registered with a different GSTIN - possible fraud or OCR error." % m["name"],
                                    "Verify GSTIN before payment.", ref(F["gstin"])))
        elif r >= 0.6:
            flags.append(mkflag("vendor_near_match", "medium", "vendor",
                                "Vendor '%s' not in master; closest is '%s' (%.0f%%)" % (vname, m["name"], r * 100),
                                "Name is similar but not identical to a registered vendor.",
                                "Confirm whether this is the registered vendor.", ref(F["vendor"]),
                                {"field": "vendor", "value": m["name"], "note": "Closest vendor in master"}))
        else:
            flags.append(mkflag("unregistered_vendor", "high", "vendor", "Vendor '%s' not found in vendor master" % vname,
                                "No registered vendor matches this name.", "Onboard/verify vendor before paying.",
                                ref(F["vendor"])))
    g = val(F, "gstin")
    if g and not re.match(GSTIN_RX, g.replace(" ", "")):
        flags.append(mkflag("bad_gstin_format", "medium", "gstin", "GSTIN '%s' has an invalid format" % g,
                            "Expected 15 characters: 2-digit state, PAN, entity, Z, checksum.",
                            "Verify against the source image.", ref(F["gstin"])))
    po = val(F, "po_number")
    pos = load_json("purchase_orders.json")
    if not po:
        if rules["require_po_for_invoices"]:
            flags.append(mkflag("missing_po", "medium", "po_number", "No PO number on invoice",
                                "Policy requires a PO reference for invoices.", "Request PO reference from vendor."))
    else:
        match = next((p for p in pos if p["po"].replace("-", "") == po.replace("-", "")), None)
        if not match:
            flags.append(mkflag("po_not_found", "high", "po_number", "PO %s does not exist in purchase system" % po,
                                "No such purchase order.", "Hold until PO is verified.", ref(F["po_number"])))
        else:
            base = val(F, "subtotal") if val(F, "subtotal") is not None else val(F, "total")
            if base is not None and abs(base - match["amount"]) > match["amount"] * rules["po_amount_tolerance_pct"] / 100:
                flags.append(mkflag("po_amount_mismatch", "high", "total",
                                    "Invoice %s differs from PO amount %s" % (money(base), money(match["amount"])),
                                    "Beyond the %s%% tolerance." % rules["po_amount_tolerance_pct"],
                                    "Check with purchasing.", ref(F.get("subtotal") or F.get("total"))))
            if vname and _best_match(vname, [{"name": match["vendor"]}])[1] < 0.8:
                flags.append(mkflag("po_vendor_mismatch", "high", "vendor",
                                    "PO %s belongs to '%s', not '%s'" % (po, match["vendor"], vname), "",
                                    "Verify vendor.", ref(F["vendor"])))
    inv, ven = val(F, "invoice_no"), vname
    if inv and ven:
        for r in known:
            if r["id"] != my_id and r.get("doc_type") == "invoice" and r["fields"].get("invoice_no", {}).get("value") == inv \
                    and _norm(r["fields"].get("vendor", {}).get("value")) == _norm(ven):
                flags.append(mkflag("duplicate_invoice", "high", "invoice_no",
                                    "Invoice %s from %s was already processed (%s)" % (inv, ven, r["id"]),
                                    "Same vendor and invoice number.", "Do not pay twice; reject duplicate.",
                                    ref(F["invoice_no"])))
                break
    total = val(F, "total")
    if total is not None and total > rules["invoice_review_threshold"]:
        flags.append(mkflag("above_threshold", "medium", "total",
                            "Invoice total %s exceeds approval threshold %s" % (money(total), money(rules["invoice_review_threshold"])),
                            "SME rule: invoices above the threshold need approver sign-off.", "Route to approver.",
                            ref(F["total"])))
    return flags


def _kyc(F, subtype):
    flags = []
    idn = val(F, "id_number")
    if idn:
        s = idn.replace(" ", "")
        if subtype == "pan":
            if not re.match(r"^[A-Z]{5}\d{4}[A-Z]$", s):
                flags.append(mkflag("bad_id_format", "high", "id_number", "PAN '%s' has an invalid format" % idn,
                                    "PAN is 5 letters, 4 digits, 1 letter.", "Verify against image.", ref(F["id_number"])))
            elif s[3] not in "PCHFATBLJG":
                flags.append(mkflag("bad_id_format", "medium", "id_number", "PAN 4th character '%s' is not a valid holder type" % s[3],
                                    "", "Verify against image.", ref(F["id_number"])))
        elif subtype == "aadhaar":
            if not re.match(r"^\d{12}$", s) or not verhoeff_ok(s):
                flags.append(mkflag("bad_id_format", "high", "id_number", "Aadhaar number fails format/checksum",
                                    "Aadhaar is 12 digits with a Verhoeff check digit.", "Verify against image.",
                                    ref(F["id_number"])))
        elif subtype == "driving_licence":
            if not re.match(r"^[A-Z]{2}\d{2}\d{4,11}$", s):
                flags.append(mkflag("bad_id_format", "medium", "id_number", "Driving licence number format unexpected",
                                    "Expected state code, RTO code, then digits.", "Verify against image.", ref(F["id_number"])))
        elif subtype == "passport" and not re.match(r"^[A-Z]\d{7}$", s):
            flags.append(mkflag("bad_id_format", "high", "id_number", "Passport number format invalid", "",
                                "Verify against image.", ref(F["id_number"])))
    name = val(F, "name")
    if name:
        m, r = _best_match(name, load_json("customers.json"))
        if r < 0.85:
            flags.append(mkflag("customer_not_found", "medium", "name", "'%s' not found in customer master" % name,
                                "No customer record matches this name.", "Confirm identity / create customer.",
                                ref(F["name"])))
    return flags


def _claim(F, rules):
    flags = []
    pol = val(F, "policy_number")
    pols = load_json("policies.json")
    if pol:
        p = next((x for x in pols if x["policy"] == pol), None)
        if not p:
            flags.append(mkflag("policy_not_found", "high", "policy_number", "Policy %s not found" % pol, "",
                                "Verify policy number.", ref(F["policy_number"])))
        else:
            amt = val(F, "claim_amount")
            if amt is not None and amt > p["limit"]:
                flags.append(mkflag("over_policy_limit", "high", "claim_amount",
                                    "Claim %s exceeds policy limit %s" % (money(amt), money(p["limit"])),
                                    "Claimed amount is above the sum insured.", "Escalate to claims assessor.",
                                    ref(F["claim_amount"])))
            dl = parse_date(val(F, "date_of_loss"))
            if dl and not (parse_date(p["valid_from"]) <= dl <= parse_date(p["valid_to"])):
                flags.append(mkflag("loss_outside_policy", "high", "date_of_loss", "Loss date outside policy period", "",
                                    "Escalate to claims assessor.", ref(F["date_of_loss"])))
            cl = val(F, "claimant")
            if cl and _best_match(cl, [{"name": p["holder"]}])[1] < 0.8:
                flags.append(mkflag("claimant_mismatch", "medium", "claimant",
                                    "Claimant differs from policyholder '%s'" % p["holder"], "", "Verify.", ref(F["claimant"])))
    return flags


def _contract(F, T, rules):
    flags, comparison = [], []
    tpl = load_json("contract_template.json")["clauses"]
    for c in T.get("clauses", []):
        std = tpl.get(c["title"].lower())
        if std is None:
            comparison.append({"title": c["title"], "status": "extra", "found": c["text"], "standard": None,
                               "diff": [], "similarity": 0.0})
            continue
        a, b = std.split(), c["text"].split()
        sm = difflib.SequenceMatcher(None, a, b)
        diff = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                diff.append({"op": "same", "text": " ".join(a[i1:i2])})
            else:
                if i2 > i1:
                    diff.append({"op": "del", "text": " ".join(a[i1:i2])})
                if j2 > j1:
                    diff.append({"op": "add", "text": " ".join(b[j1:j2])})
        comparison.append({"title": c["title"], "status": "standard" if sm.ratio() > 0.98 else "deviates",
                           "found": c["text"], "standard": std, "diff": diff, "similarity": round(sm.ratio(), 2),
                           "page": c["page"], "boxes": c["boxes"]})
    found = {c["title"].lower() for c in T.get("clauses", [])}
    for t in tpl:
        if t not in found:
            comparison.append({"title": t.title(), "status": "missing", "found": None, "standard": tpl[t], "diff": [],
                               "similarity": 0.0})
            flags.append(mkflag("missing_clause", "medium", None, "Standard clause '%s' is missing" % t.title(),
                                "Present in the standard template but not in this contract.", "Legal review."))
    ptd, tnd = val(F, "payment_terms_days"), val(F, "termination_notice_days")
    if ptd is not None and ptd > rules["contract_max_payment_days"]:
        flags.append(mkflag("nonstandard_payment_terms", "medium", "payment_terms_days",
                            "Payment terms %d days exceed the %d-day maximum" % (ptd, rules["contract_max_payment_days"]),
                            "Standard template: 30 days.", "Negotiate or get legal approval.", ref(F["payment_terms_days"])))
    if tnd is not None and tnd < rules["contract_min_termination_notice_days"]:
        flags.append(mkflag("short_termination_notice", "medium", "termination_notice_days",
                            "Termination notice %d days is below the %d-day minimum" % (tnd, rules["contract_min_termination_notice_days"]),
                            "Standard template: 30 days.", "Legal review.", ref(F["termination_notice_days"])))
    if rules["contract_require_liability_cap"] and val(F, "liability_capped") == "no":
        flags.append(mkflag("unlimited_liability", "high", "liability_capped", "Liability clause is uncapped",
                            "Standard template caps liability at the PO value.", "Legal review before signing.",
                            ref(F["liability_capped"])))
    return flags, comparison


def _custom(dtype, F, rules):
    flags = []
    for r in rules.get("custom_rules", []):
        if r.get("doc_type") not in (dtype, "any"):
            continue
        f = F.get(r["field"])
        if not f or f["value"] is None:
            continue
        v, rhs = f["value"], r["value"]
        num = parse_amount(rhs) if isinstance(v, (int, float)) else None
        try:
            if isinstance(v, (int, float)):
                ok = OPS[r["op"]](float(v), float(num))
            else:
                ok = OPS[r["op"]](str(v).lower(), str(rhs).lower())
        except Exception:
            continue
        if ok:
            flags.append(mkflag("custom_rule", r.get("severity", "medium"), r["field"],
                                r.get("message") or "Custom rule: %s %s %s" % (r["field"], r["op"], rhs),
                                "SME-defined rule: %s %s %s (value %s)." % (r["field"], r["op"], rhs, v),
                                "Review per company policy.", ref(f)))
    return flags
