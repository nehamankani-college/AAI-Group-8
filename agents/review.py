"""Agent 8 - Human Review Routing: queue, team routing, highlighted crops, SLA tracking, corrections."""
import datetime as dt
import os
import time

import cv2

from .store import doc_dir
from .validation import SEV_ORDER

TEAM = {"invoice": "Finance", "kyc": "Compliance", "contract": "Legal", "claim": "Claims", "other": "Operations Triage"}


def _crop(img, refs, path, pad=60):
    """Crop around all flagged boxes on the first referenced page and draw red rectangles."""
    if not refs:
        return None
    page = refs[0]["page"]
    boxes = [b for r in refs if r["page"] == page for b in r["boxes"]]
    if not boxes:
        return None
    rgb = cv2.cvtColor(img[page - 1] if isinstance(img, list) else img, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        cv2.rectangle(rgb, (b[0] - 4, b[1] - 4), (b[2] + 4, b[3] + 4), (0, 0, 220), 3)
    x0 = max(0, min(b[0] for b in boxes) - pad * 3)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(rgb.shape[1], max(b[2] for b in boxes) + pad * 3)
    y1 = min(rgb.shape[0], max(b[3] for b in boxes) + pad)
    cv2.imwrite(path, rgb[y0:y1, x0:x1])
    return os.path.basename(path)


def build_review(record, images, rules):
    """Create/refresh the review item. Returns None when the document is auto-approved."""
    decision = record["score"]["decision"]
    if decision == "auto_approve":
        return None
    d = os.path.join(doc_dir(record["id"]), "crops")
    os.makedirs(d, exist_ok=True)
    prev = record.get("review") or {}
    now = prev.get("created_at") or time.time()
    items = []
    for i, f in sorted(enumerate(record["flags"]), key=lambda t: -SEV_ORDER[t[1]["severity"]]):
        if f["severity"] == "low":
            continue
        crop = _crop(images, f["refs"], os.path.join(d, "flag_%d.png" % i)) if f["refs"] else None
        items.append({"flag_index": i, "field": f["field"], "severity": f["severity"], "message": f["message"],
                      "explanation": f["explanation"], "recommendation": f["recommendation"],
                      "suggestion": f.get("suggestion"), "crop": crop, "page": f["refs"][0]["page"] if f["refs"] else None})
    prio = record["score"]["priority"]
    sla_h = rules["sla_hours"].get(prio, 72)
    return {"team": TEAM.get(record["doc_type"], "Operations Triage"), "status": prev.get("status", "pending"),
            "priority": prio, "created_at": now, "sla_hours": sla_h, "sla_due": now + sla_h * 3600,
            "resolved_at": prev.get("resolved_at"), "turnaround_s": prev.get("turnaround_s"),
            "resolution": prev.get("resolution"), "entries": items}


def resolve(record, resolution, reviewer="reviewer"):
    r = record["review"]
    now = time.time()
    r.update(status="resolved", resolution=resolution, resolved_at=now, turnaround_s=round(now - r["created_at"], 1),
             resolved_by=reviewer)


def sla_state(review, now=None):
    now = now or time.time()
    if not review:
        return None
    if review["status"] == "resolved":
        return "met" if review["resolved_at"] <= review["sla_due"] else "breached"
    return "overdue" if now > review["sla_due"] else "on_track"


def fmt_time(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%d %b %H:%M")
