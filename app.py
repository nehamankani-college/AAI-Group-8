"""Web UI + API for the Document Intelligence & Extraction Agent."""
import glob
import json
import os
import tempfile
import time

from jinja2 import pass_context
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for, flash

from agents import llm_agent, orchestrator, review, store
from agents.common import OUT, load_rules, save_rules, CURRENCY_SYMBOLS

app = Flask(__name__, template_folder="web/templates")
app.secret_key = "dev-only"
SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def _ensure_samples():
    """Auto-generate synthetic sample documents if the samples/ folder is empty."""
    if not os.path.isdir(SAMPLES) or not any(
        f.endswith((".pdf", ".jpg", ".png")) for f in os.listdir(SAMPLES)
    ):
        import subprocess
        import sys
        try:
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_samples.py")
            subprocess.run([sys.executable, script], check=True, timeout=60)
        except Exception as e:  # noqa: BLE001
            print("[docai] Warning: could not auto-generate samples: %s" % e)


_ensure_samples()
DEMO_ORDER = ["Invoice_0847.pdf", "PAN_card_supriya.jpg", "DL_expired_rajesh.pdf", "Service_Agreement_Shree.pdf",
              "Claim_CLM5521.pdf", "Claim_photo_blurry.jpg"]

# Full 20-document demo dataset (run make_demo_dataset.py to generate)
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_dataset")
DATASET_ORDER = [
    "01_inv_shree_clean.pdf", "02_inv_mehta_valid.pdf", "03_inv_apex_highvalue.pdf",
    "04_inv_unregistered_vendor.pdf", "05_inv_math_error.pdf", "06_inv_duplicate.pdf",
    "07_inv_no_po.pdf", "08_inv_future_date.pdf",
    "09_kyc_pan_valid.pdf", "10_kyc_pan_bad_format.pdf", "11_kyc_dl_expired.pdf",
    "12_kyc_aadhaar_valid.pdf", "13_kyc_passport_underage.pdf",
    "14_contract_standard.pdf", "15_contract_long_payment.pdf",
    "16_contract_no_liability_cap.pdf", "17_contract_short_notice.pdf",
    "18_claim_valid.pdf", "19_claim_over_limit.pdf", "20_claim_blurry_photo.jpg",
]


@app.template_filter("money")
@pass_context
def _money(ctx, x):
    if not isinstance(x, (int, float)):
        return x
    r = ctx.get("r")
    code = r.get("fields", {}).get("currency", {}).get("value") if isinstance(r, dict) else None
    return "%s{:,.2f}".format(x) % CURRENCY_SYMBOLS.get(code, "₹")


@app.template_filter("ts")
def _ts(x):
    return review.fmt_time(x) if x else ""


@app.template_filter("dur")
def _dur(s):
    if s is None:
        return ""
    return "%.0fs" % s if s < 90 else "%.1f min" % (s / 60) if s < 5400 else "%.1f h" % (s / 3600)


app.jinja_env.globals.update(sla_state=review.sla_state, now=time.time)


def stats(recs):
    rules = load_rules()
    n = len(recs)
    by = {}
    for r in recs:
        by[r["status"]] = by.get(r["status"], 0) + 1
    proc = sum(r["processing_seconds"] for r in recs)
    pending = [r for r in recs if r.get("review") and r["review"]["status"] == "pending"]
    resolved = [r for r in recs if r.get("review") and r["review"]["status"] == "resolved"]
    auto = by.get("auto_approved", 0)
    manual_min = n * rules["manual_minutes_per_doc"]
    return {"n": n, "by": by, "avg_s": round(proc / n, 1) if n else 0, "total_s": round(proc, 1),
            "pending": len(pending), "overdue": sum(1 for r in pending if review.sla_state(r["review"]) == "overdue"),
            "avg_turnaround": (sum(r["review"]["turnaround_s"] for r in resolved) / len(resolved)) if resolved else None,
            "manual_min": manual_min, "saved_min": round(max(0, manual_min - proc / 60), 1),
            "auto_rate": round(100 * auto / n) if n else 0, "assumed_min": rules["manual_minutes_per_doc"]}


@app.route("/")
def dashboard():
    recs = store.list_records()
    return render_template("dashboard.html", recs=recs, st=stats(recs), page="dashboard", llm=llm_agent.status(load_rules()))


@app.route("/upload", methods=["POST"])
def upload():
    files = [f for f in request.files.getlist("files") if f.filename]
    if not files:
        flash("Choose at least one file.")
        return redirect(url_for("dashboard"))
    last = None
    for f in files:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, os.path.basename(f.filename))
            f.save(p)
            last = orchestrator.process(p, filename=os.path.basename(f.filename))
    return redirect(url_for("doc", doc_id=last["id"]) if len(files) == 1 else url_for("dashboard"))


@app.route("/demo", methods=["POST"])
def demo():
    store.clear_all()
    for name in DEMO_ORDER:
        p = os.path.join(SAMPLES, name)
        if os.path.exists(p):
            orchestrator.process(p)
    return redirect(url_for("dashboard"))


@app.route("/demo_dataset", methods=["POST"])
def demo_dataset():
    """Run the full 20-document demo dataset."""
    store.clear_all()
    for name in DATASET_ORDER:
        p = os.path.join(DATASET_DIR, name)
        if os.path.exists(p):
            orchestrator.process(p)
        else:
            flash(f"Dataset file missing: {name}. Run python3 make_demo_dataset.py first.")
            return redirect(url_for("dashboard"))
    return redirect(url_for("dashboard"))


@app.route("/doc/<doc_id>")
def doc(doc_id):
    r = store.load(doc_id)
    if not r:
        abort(404)
    overlay = {"fields": {k: {"page": f["page"], "boxes": f["boxes"]} for k, f in r["fields"].items() if f.get("page")},
               "flags": [{"i": i, "sev": f["severity"], "refs": f["refs"]} for i, f in enumerate(r["flags"])],
               "pages": r["pages"]}
    return render_template("doc.html", r=r, overlay=json.dumps(overlay), page="doc")


@app.route("/file/<doc_id>/<path:name>")
def file(doc_id, name):
    return send_from_directory(os.path.join(OUT, doc_id), name)


@app.route("/doc/<doc_id>/output.<ext>")
def download(doc_id, ext):
    if ext not in ("json", "csv"):
        abort(404)
    return send_from_directory(os.path.join(OUT, doc_id), "output." + ext, as_attachment=True,
                               download_name="%s.%s" % (doc_id, ext))


@app.route("/doc/<doc_id>/correct", methods=["POST"])
def correct(doc_id):
    try:
        orchestrator.correct(doc_id, request.form["field"], request.form["value"])
    except Exception as e:
        flash("Could not apply correction: %s" % e)
    return redirect(request.form.get("next") or url_for("doc", doc_id=doc_id))


@app.route("/doc/<doc_id>/decide", methods=["POST"])
def decide(doc_id):
    orchestrator.decide(doc_id, request.form["resolution"])
    return redirect(request.form.get("next") or url_for("doc", doc_id=doc_id))


@app.route("/review")
def review_queue():
    recs = store.list_records()
    pending = [r for r in recs if r.get("review") and r["review"]["status"] == "pending"]
    done = [r for r in recs if r.get("review") and r["review"]["status"] == "resolved"]
    teams = {}
    for r in sorted(pending, key=lambda r: r["review"]["sla_due"]):
        teams.setdefault(r["review"]["team"], []).append(r)
    return render_template("review.html", teams=teams, done=done, st=stats(recs), page="review")


@app.route("/rules", methods=["GET", "POST"])
def rules_panel():
    rules = load_rules()
    if request.method == "POST":
        f = request.form
        for k in ("auto_approve_min_confidence", "ocr_retry_below", "low_field_confidence", "amount_tolerance",
                  "invoice_review_threshold", "po_amount_tolerance_pct", "vendor_match_min_ratio",
                  "manual_minutes_per_doc"):
            rules[k] = float(f[k])
        for k in ("contract_max_payment_days", "contract_min_termination_notice_days"):
            rules[k] = int(f[k])
        for k in ("require_po_for_invoices", "require_signatures", "contract_require_liability_cap", "llm_enabled",
                  "llm_allow_kyc"):
            rules[k] = k in f
        rules["sla_hours"] = {s: float(f["sla_" + s]) for s in ("high", "medium", "low")}
        custom = []
        for i in range(int(f.get("n_custom", 0))):
            if f.get("c_del_%d" % i) or not f.get("c_field_%d" % i, "").strip():
                continue
            custom.append({"doc_type": f["c_type_%d" % i], "field": f["c_field_%d" % i].strip(), "op": f["c_op_%d" % i],
                           "value": f["c_value_%d" % i], "severity": f["c_sev_%d" % i], "message": f["c_msg_%d" % i]})
        rules["custom_rules"] = custom
        save_rules(rules)
        flash("Rules saved - they apply to the next document processed.")
        return redirect(url_for("rules_panel"))
    return render_template("rules.html", rules=rules, page="rules", llm=llm_agent.status(rules))


@app.route("/rules/nl", methods=["POST"])
def rules_nl():
    """Plain-English rule -> structured rule (validated) -> appended to the rule table."""
    text = request.form.get("text", "").strip()
    ok, why = llm_agent.available(load_rules())
    if not ok:
        flash("Plain-English rules need the LLM: %s." % why)
        return redirect(url_for("rules_panel"))
    try:
        rule = llm_agent.parse_rule(text)
    except Exception as e:  # noqa: BLE001
        flash("Could not turn that into a rule (%s). Try naming the document type, field and limit." % e)
        return redirect(url_for("rules_panel"))
    rules = load_rules()
    rules.setdefault("custom_rules", []).append(rule)
    save_rules(rules)
    flash("Added rule: when %s %s %s %s -> flag as %s. You can edit or delete it in the table below." % (
        rule["doc_type"], rule["field"], rule["op"], rule["value"], rule["severity"]))
    return redirect(url_for("rules_panel"))


@app.route("/api/process", methods=["POST"])
def api_process():
    f = request.files.get("file")
    if not f:
        return jsonify(error="send multipart field 'file'"), 400
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, os.path.basename(f.filename))
        f.save(p)
        r = orchestrator.process(p, filename=os.path.basename(f.filename))
    with open(os.path.join(OUT, r["id"], "output.json"), encoding="utf-8") as fh:
        return jsonify(json.load(fh))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
