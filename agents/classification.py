"""Agent 2 - Classification: doc type + sub-type from OCR text. Unknown templates are flagged, not forced."""
import re

KEYWORDS = {
    "invoice": ["invoice", "tax invoice", "gstin", "bill to", "subtotal", "qty", "rate", "amount", "due date",
                "hsn", "po number", "total", "gst"],
    "kyc": ["date of birth", "dob", "permanent account number", "income tax department", "aadhaar", "uidai",
            "driving licence", "driving license", "passport", "election commission", "valid till", "father's name",
            "govt. of india", "आधार", "जन्म", "नाम", "पैन"],
    "contract": ["agreement", "parties", "party", "termination", "governing law", "payment terms", "liability",
                 "hereinafter", "whereas", "scope of services", "signed for", "witness"],
    "claim": ["claim", "claimant", "policy no", "policy number", "date of loss", "claim amount", "insurer",
              "policyholder", "insurance"],
}
SUBTYPES = {
    "kyc": {"pan": ["permanent account number", "income tax department"],
            "aadhaar": ["aadhaar", "uidai", "आधार"],
            "driving_licence": ["driving licence", "driving license", "dl no"],
            "passport": ["passport", "republic of india"],
            "utility_bill": ["electricity", "consumption", "meter", "units consumed", "water bill"]},
    "claim": {"motor": ["motor", "vehicle", "collision"], "health": ["hospital", "diagnosis", "medical"]},
    "invoice": {"tax_invoice": ["tax invoice", "gstin"]},
    "contract": {"service_agreement": ["service agreement", "scope of services"], "nda": ["confidential"]},
}


def _has(text, k):
    if not k.isascii():
        return k in text
    return re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", text) is not None


def classify(ocr, min_hits=2):
    text = " ".join(p["text"] for p in ocr["pages"]).lower()
    hits = {t: sorted({k for k in kws if _has(text, k)}) for t, kws in KEYWORDS.items()}
    counts = {t: len(v) for t, v in hits.items()}
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    (top, n), (second, n2) = ranked[0], ranked[1]
    if n < min_hits:
        return {"doc_type": "other", "subtype": None, "confidence": round(0.3 + 0.1 * n, 2), "template_match": False,
                "evidence": hits, "note": "No known template matched; routed to manual triage"}
    conf = min(0.99, 0.55 + 0.08 * n)
    if n2 >= 0.6 * n:
        conf -= 0.2  # ambiguous between two types
    subtype = None
    for st, kws in SUBTYPES.get(top, {}).items():
        if any(_has(text, k) for k in kws):
            subtype = st
            break
    if top == "kyc" and subtype is None:
        if re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", " ".join(p["text"] for p in ocr["pages"])):
            subtype = "pan"
    return {"doc_type": top, "subtype": subtype, "confidence": round(max(0.05, conf), 2),
            "template_match": conf >= 0.6, "evidence": {top: hits[top]}, "runner_up": second if n2 else None}
