"""Agent 7 - Confidence & Anomaly Scoring: aggregate OCR confidence + flags into a document decision."""
import numpy as np

from .validation import SEV_ORDER

PENALTY = {"high": 0.35, "medium": 0.15, "low": 0.03}
REJECT_CODES = {"expired_id", "unknown_doc_type", "duplicate_invoice", "unreadable_document"}


def score(record, flags, rules):
    confs = [f["confidence"] for f in record["fields"].values()
             if f.get("method") not in ("classifier", "sum-of-rows") and f.get("confidence") is not None]
    field_conf = float(np.mean(confs)) if confs else 0.0
    risk = min(1.0, sum(PENALTY[f["severity"]] for f in flags))
    confidence = round(field_conf * (1 - 0.7 * risk), 3)  # document score: extraction confidence discounted by risk
    reasons = []
    rejects = [f for f in flags if f["code"] in REJECT_CODES]
    blocking = [f for f in flags if f["severity"] in ("high", "medium")]
    if rejects:
        decision = "reject"
        reasons = [f["message"] for f in rejects]
    elif blocking:
        decision = "needs_review"
        reasons = [f["message"] for f in sorted(blocking, key=lambda f: -SEV_ORDER[f["severity"]])]
    elif field_conf < rules["auto_approve_min_confidence"]:
        decision = "needs_review"
        reasons = ["Average field confidence %.2f is below auto-approval threshold %.2f" % (
            field_conf, rules["auto_approve_min_confidence"])]
    else:
        decision = "auto_approve"
        reasons = ["All validations passed and confidence %.2f >= %.2f" % (field_conf, rules["auto_approve_min_confidence"])]
    top = max((SEV_ORDER[f["severity"]] for f in flags), default=0)
    priority = {3: "high", 2: "medium", 1: "low", 0: "none"}[top]
    return {"decision": decision, "confidence": confidence, "field_confidence": round(field_conf, 3),
            "risk": round(risk, 3), "priority": priority, "reasons": reasons, "flag_count": len(flags)}
