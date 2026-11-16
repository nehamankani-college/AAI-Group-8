"""Agent 10 - Orchestrator (Agentic AI & Enterprise Automation — PS-15)
======================================================================
Role: coordinate the full 10-agent pipeline and maintain an end-to-end
audit trail for every document processed.

Agent loop
----------
  ingest → classify → OCR (adaptive retry) → extract → [LLM cross-check]
  → validate → business rules → score → review-route / auto-approve → output

Decision authority: orchestration only. The orchestrator never approves,
rejects, or modifies extracted values. Those decisions belong either to
Agent 7 (scoring, deterministic) or to a human reviewer.

Human-in-the-loop path
-----------------------
  When a human corrects a field, `correct()` re-runs agents 5-9 so the
  correction propagates through validation, rules, scoring, review and output.
  The full correction diff is recorded in the audit trail.
"""
import os
import re
import shutil
import time
import uuid

from . import classification, extraction, ingestion, llm_agent, ocr, output, review, rules_agent, scoring, store, validation
from . import common
from .common import Audit, OUT, load_rules, parse_amount, parse_date

# Retry ladder when OCR confidence is too low: re-render at higher DPI, then upscale + add Hindi model.
ATTEMPTS = [
    {"dpi": 200, "upscale": 1.0, "lang": "eng"},
    {"dpi": 300, "upscale": 1.0, "lang": "eng"},
    {"dpi": 300, "upscale": 1.5, "lang": "eng+hin"},
]


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(name)[0]).strip("_")[:30]


def _llm_try(name, fn, audit, meta):
    """Run an LLM step; any failure is logged and the pipeline simply continues without it."""
    try:
        t = time.time()
        out = fn()
        usage = out[-1]
        meta["tokens"]["in"] += usage.get("in", 0)
        meta["tokens"]["out"] += usage.get("out", 0)
        audit.log("llm", name, seconds=round(time.time() - t, 1), tokens_in=usage.get("in", 0), tokens_out=usage.get("out", 0))
        return out[:-1]
    except Exception as e:  # noqa: BLE001 - deliberately broad: the LLM is optional
        audit.log("llm", "%s failed - continuing without the LLM" % name, error="%s: %s" % (type(e).__name__, str(e)[:200]))
        return None


def _briefing(record, rules, audit):
    """AI-written plain-English summary for reviewers (never changes the decision)."""
    record["llm_summary"] = None
    ok, _ = llm_agent.available(rules)
    meta = record.setdefault("llm", {"enabled": ok, "tokens": {"in": 0, "out": 0}})
    if ok and record["score"]["decision"] != "auto_approve":
        out = _llm_try("reviewer briefing", lambda: llm_agent.summarize(record), audit, meta)
        if out:
            record["llm_summary"] = out[0]


def evaluate(record, images, rules, known, audit):
    """Agents 5-9 on an extracted record. Used for first pass and again after a human correction."""
    t = time.time()
    common.set_currency((record["fields"].get("currency") or {}).get("value", "INR"))
    f5 = validation.validate(record, rules)
    audit.log("validation", "cross-field checks", flags=[f["code"] for f in f5])
    f6, comparison = rules_agent.apply_rules(record, rules, known)
    audit.log("business_rules", "reference + SME rule checks", flags=[f["code"] for f in f6])
    record["flags"] = f5 + f6 + record.get("llm_flags", [])   # LLM cross-check flags (agent: llm)
    record["clause_comparison"] = comparison
    record["score"] = scoring.score(record, record["flags"], rules)
    audit.log("scoring", "scored document", **{k: record["score"][k] for k in ("decision", "confidence", "priority")},
              reasons=record["score"]["reasons"][:5])
    record["review"] = review.build_review(record, images, rules)
    if record["review"]:
        audit.log("review_routing", "routed to %s queue" % record["review"]["team"], priority=record["review"]["priority"],
                  items=len(record["review"]["entries"]), sla_hours=record["review"]["sla_hours"])
    out, summary = output.write_outputs(record)
    record["status"] = output.final_status(record)
    audit.log("output", "wrote output.json / output.csv / summary.md", status=record["status"], schema=out["schema"])
    record["timings"]["evaluate_s"] = round(time.time() - t, 2)


def process(path, rules=None, filename=None):
    rules = rules or load_rules()
    filename = filename or os.path.basename(path)
    t0 = time.time()
    audit, timings = Audit(), {}
    doc_id = "%s-%s" % (_slug(filename), uuid.uuid4().hex[:5])
    audit.log("orchestrator", "start", file=filename, id=doc_id)

    best, attempts_log = None, []
    for n, a in enumerate(ATTEMPTS):
        if a["lang"] == "eng+hin" and "hin" not in _langs():
            continue
        ts = time.time()
        ing = ingestion.ingest(path, dpi=a["dpi"], upscale=a["upscale"])
        res = ocr.run_ocr(ing, lang=a["lang"])
        info = {"attempt": n + 1, **a, "mean_conf": res["mean_conf"], "quality": ingestion.summary(ing)["quality_score"],
                "seconds": round(time.time() - ts, 1)}
        attempts_log.append(info)
        audit.log("ingestion", "prepared %d page(s)" % len(ing["pages"]), **ingestion.summary(ing))
        audit.log("ocr", "OCR attempt %d (%s, dpi %d, x%.1f)" % (n + 1, a["lang"], a["dpi"], a["upscale"]),
                  mean_conf=res["mean_conf"], low_conf_segments=res["low_conf_segments"])
        if best is None or res["mean_conf"] > best[1]["mean_conf"]:
            best = (ing, res)
        if res["mean_conf"] >= rules["ocr_retry_below"]:
            break
        audit.log("orchestrator", "OCR confidence %.2f < %.2f - retrying with a stronger pass instead of guessing" % (
            res["mean_conf"], rules["ocr_retry_below"]))
    timings["ingest_ocr_s"] = round(time.time() - t0, 2)
    ing, res = best

    images = [p["image"] for p in ing["pages"]]
    llm_ok, llm_why = llm_agent.available(rules)
    llm_meta = {"enabled": llm_ok, "note": llm_why, "model": llm_agent.model() if llm_ok else None,
                "tokens": {"in": 0, "out": 0}}
    if not llm_ok:
        audit.log("llm", "not used: %s" % llm_why)

    cls = classification.classify(res)
    audit.log("classification", "classified", doc_type=cls["doc_type"], subtype=cls["subtype"], confidence=cls["confidence"])
    if llm_ok and (cls["doc_type"] == "other" or cls["confidence"] < 0.6):
        text = " ".join(p["text"] for p in res["pages"])
        out = _llm_try("classification fallback (OCR text only)", lambda: llm_agent.classify(text), audit, llm_meta)
        if out and out[0]["doc_type"] != "other":
            audit.log("classification", "LLM reclassified %s -> %s" % (cls["doc_type"], out[0]["doc_type"]),
                      subtype=out[0]["subtype"], confidence=out[0]["confidence"], reason=out[0]["reason"])
            cls = out[0]
    t = time.time()
    fields, tables = extraction.extract(cls["doc_type"], cls["subtype"], res, images)
    timings["extract_s"] = round(time.time() - t, 2)
    audit.log("extraction", "extracted %d fields" % len(fields),
              fields={k: {"value": v["value"], "conf": v["confidence"], "page": v["page"]} for k, v in fields.items()})

    llm_flags = []
    if llm_ok and cls["doc_type"] in llm_agent.SCHEMAS:
        if cls["doc_type"] == "kyc" and not rules.get("llm_allow_kyc", False):
            audit.log("llm", "vision extraction skipped for KYC (identity documents are not sent to the API; "
                             "enable 'llm_allow_kyc' in Business rules to allow)")
        else:
            t = time.time()
            out = _llm_try("vision extraction", lambda: llm_agent.extract(cls["doc_type"], images, rules), audit, llm_meta)
            if out:
                lf, items, = out
                llm_flags, stats = llm_agent.apply(fields, tables, lf, items, extraction.Doc(res, images))
                llm_meta["cross_check"] = stats
                audit.log("llm", "cross-checked against the regex extractor",
                          agree=stats["agree"], disagree=stats["disagree"], filled=stats["filled"],
                          unverified=stats["unverified"])
            timings["llm_extract_s"] = round(time.time() - t, 2)

    store.save_page_images(doc_id, images)
    src_copy = os.path.join(OUT, doc_id, "source" + os.path.splitext(filename)[1])
    shutil.copyfile(path, src_copy)

    record = {"id": doc_id, "filename": filename, "created_at": time.time(), "doc_type": cls["doc_type"],
              "subtype": cls["subtype"], "classification": cls, "ingestion": ingestion.summary(ing),
              "ocr": {"lang": res["lang"], "mean_conf": res["mean_conf"], "attempts": attempts_log,
                      "low_conf_segments": res["low_conf_segments"]},
              "pages": [{"page": p["page"], "width": p["width"], "height": p["height"]} for p in ing["pages"]],
              "fields": fields, "tables": tables, "corrections": [], "timings": timings,
              "llm": llm_meta, "llm_flags": llm_flags, "llm_summary": None}
    known = [r for r in store.list_records() if r["id"] != doc_id]
    evaluate(record, images, rules, known, audit)
    _briefing(record, rules, audit)
    record["processing_seconds"] = round(time.time() - t0, 2)
    audit.log("orchestrator", "done", status=record["status"], seconds=record["processing_seconds"])
    record["audit"] = audit.events
    store.save(record)
    return record


def _langs():
    try:
        import pytesseract
        return pytesseract.get_languages()
    except Exception:
        return ["eng"]


def _coerce(field, new):
    old = field["value"]
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        v = parse_amount(new)
        if v is None:
            raise ValueError("Enter a number")
        return int(v) if isinstance(old, int) else v
    if field.get("raw") and parse_date(field["value"]) is not None or "date" in str(field.get("label", "")):
        d = parse_date(new)
        if d is None:
            raise ValueError("Enter a date as dd/mm/yyyy")
        return d.isoformat()
    return new.strip()


def correct(doc_id, field_name, new_value, reviewer="reviewer"):
    """Human fixes a field; show before/after, re-run validation + scoring."""
    rules = load_rules()
    record = store.load(doc_id)
    audit = Audit(record["audit"])
    f = record["fields"][field_name]
    before = f["value"]
    after = _coerce(f, new_value)
    f.update(value=after, confidence=1.0, method="human-corrected", corrected=True)
    record["corrections"].append({"field": field_name, "before": before, "after": after, "by": reviewer, "ts": time.time()})
    audit.log("review", "field corrected by %s" % reviewer, field=field_name, before=before, after=after)
    record["llm_flags"] = [x for x in record.get("llm_flags", []) if x.get("field") != field_name]  # human settled it
    images = store.load_page_images(doc_id, len(record["pages"]))
    known = [r for r in store.list_records() if r["id"] != doc_id]
    evaluate(record, images, rules, known, audit)
    _briefing(record, rules, audit)
    rv = record.get("review")
    if rv and not any(i["severity"] in ("high", "medium") for i in rv["entries"]):
        review.resolve(record, "approved", reviewer)
        record["status"] = output.final_status(record)
        audit.log("review", "all blocking flags cleared - resolved", turnaround_s=record["review"]["turnaround_s"])
        output.write_outputs(record)
    elif rv is None:
        audit.log("review", "document now passes all checks")
    record["audit"] = audit.events
    store.save(record)
    return record


def decide(doc_id, resolution, reviewer="reviewer"):
    """Reviewer overrides: approve as-is or reject."""
    record = store.load(doc_id)
    audit = Audit(record["audit"])
    if not record.get("review"):
        return record
    review.resolve(record, resolution, reviewer)
    record["status"] = output.final_status(record)
    audit.log("review", "reviewer %s the document" % resolution, by=reviewer, turnaround_s=record["review"]["turnaround_s"])
    output.write_outputs(record)
    record["audit"] = audit.events
    store.save(record)
    return record
