"""Agent 9 - Structured Output: downstream schema, JSON + CSV + human summary, every field source-linked."""
import csv
import io
import json
import os

from .common import money
from .store import doc_dir

STATUS = {"auto_approve": "auto_approved", "needs_review": "needs_review", "reject": "rejected"}


def final_status(record):
    rv = record.get("review")
    if rv and rv["status"] == "resolved":
        return {"approved": "approved_by_reviewer", "rejected": "rejected_by_reviewer"}.get(rv.get("resolution"), "reviewed")
    return STATUS[record["score"]["decision"]]


def _src(record, f):
    if not f or f.get("page") is None:
        return None
    return {"file": record["filename"], "page": f["page"], "bbox": f["bbox"],
            "ref": "%s#p%d" % (record["filename"], f["page"])}


def _fv(record, k):
    f = record["fields"].get(k)
    return f["value"] if f else None


def build_output(record):
    F, T, dtype = record["fields"], record["tables"], record["doc_type"]
    src = {k: _src(record, f) for k, f in F.items() if f.get("page") is not None}
    meta = {"document_id": record["id"], "status": final_status(record), "confidence": record["score"]["confidence"],
            "priority": record["score"]["priority"], "review_team": (record.get("review") or {}).get("team"),
            "source_ref": "%s#p1" % record["filename"], "flags": [f["message"] for f in record["flags"]]}
    if dtype == "invoice":
        body = {"schema": "accounting.invoice.v1", "invoice_no": _fv(record, "invoice_no"),
                "vendor": {"name": _fv(record, "vendor"), "gstin": _fv(record, "gstin")},
                "buyer": _fv(record, "bill_to"), "invoice_date": _fv(record, "invoice_date"),
                "due_date": _fv(record, "due_date"), "po_number": _fv(record, "po_number"), "currency": _fv(record, "currency") or "INR",
                "subtotal": _fv(record, "subtotal"), "tax_total": _fv(record, "tax_total"), "total": _fv(record, "total"),
                "line_items": [{"description": i["description"], "qty": i["qty"], "rate": i["rate"], "amount": i["amount"],
                                "source_ref": "%s#p%d" % (record["filename"], i["page"]), "bbox": i["bbox"]}
                               for i in T.get("line_items", [])]}
    elif dtype == "kyc":
        body = {"schema": "crm.kyc_record.v1", "customer_name": _fv(record, "name"), "id_type": _fv(record, "id_type"),
                "id_number": _fv(record, "id_number"), "date_of_birth": _fv(record, "dob"),
                "id_expiry": _fv(record, "expiry_date"), "address": _fv(record, "address"),
                "signature_present": _fv(record, "signature") == "signed"}
    elif dtype == "contract":
        body = {"schema": "legal.contract_record.v1", "party_a": _fv(record, "party_a"), "party_b": _fv(record, "party_b"),
                "effective_date": _fv(record, "effective_date"), "payment_terms_days": _fv(record, "payment_terms_days"),
                "termination_notice_days": _fv(record, "termination_notice_days"),
                "liability_capped": _fv(record, "liability_capped"), "governing_law": _fv(record, "governing_law"),
                "signatures": [f["value"] for k, f in F.items() if k.startswith("signature")],
                "clauses": [{"number": c["number"], "title": c["title"], "text": c["text"]} for c in T.get("clauses", [])]}
    elif dtype == "claim":
        body = {"schema": "claims.claim_record.v1", "claim_number": _fv(record, "claim_number"),
                "policy_number": _fv(record, "policy_number"), "claimant": _fv(record, "claimant"),
                "date_of_loss": _fv(record, "date_of_loss"), "claim_date": _fv(record, "claim_date"),
                "claim_amount": _fv(record, "claim_amount"), "description": _fv(record, "description")}
    else:
        body = {"schema": "triage.unclassified.v1", "note": "Unrecognised document; manual triage required"}
    return {**body, **meta, "field_sources": src}


def to_csv(out, record):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["document_id", "field", "value", "confidence", "source_ref", "bbox"])
    for k, f in record["fields"].items():
        s = out["field_sources"].get(k)
        w.writerow([record["id"], k, f["value"], f["confidence"], s["ref"] if s else "", s["bbox"] if s else ""])
    for i in record["tables"].get("line_items", []):
        w.writerow([record["id"], "line_item", "%s | qty=%s | rate=%s | amount=%s" % (
            i["description"], i["qty"], i["rate"], i["amount"]), i["confidence"], "%s#p%d" % (record["filename"], i["page"]),
            i["bbox"]])
    return buf.getvalue()


def to_summary(record):
    s, lines = record["score"], []
    lines.append("# %s" % record["filename"])
    lines.append("Type: %s%s | Status: %s | Confidence: %.2f | Priority: %s" % (
        record["doc_type"], " (%s)" % record["subtype"] if record.get("subtype") else "", final_status(record),
        s["confidence"], s["priority"]))
    lines.append("")
    for k, f in record["fields"].items():
        v = f["value"]
        v = money(v) if k in ("total", "subtotal", "tax_total", "claim_amount") and isinstance(v, (int, float)) else v
        lines.append("- %s: %s (conf %.2f)" % (k.replace("_", " "), v, f["confidence"]))
    if record["flags"]:
        lines.append("")
        lines.append("Issues:")
        for f in record["flags"]:
            lines.append("- [%s] %s - %s" % (f["severity"].upper(), f["message"], f["recommendation"]))
    return "\n".join(lines)


def write_outputs(record):
    out = build_output(record)
    d = doc_dir(record["id"])
    with open(os.path.join(d, "output.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(d, "output.csv"), "w", encoding="utf-8", newline="") as fh:
        fh.write(to_csv(out, record))
    summary = to_summary(record)
    with open(os.path.join(d, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)
    return out, summary
