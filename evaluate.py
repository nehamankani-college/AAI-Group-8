"""Score the pipeline against an answer key.
Usage: python evaluate.py [folder]   (folder holds documents + expected.csv; default: testset)
expected.csv columns: file, doc_type, decision (auto_approve|needs_review|reject), fields (JSON of field->true value)"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents import orchestrator, store  # noqa: E402

DEC = {"auto_approved": "auto_approve", "needs_review": "needs_review", "rejected": "reject"}


def same(a, b):
    if isinstance(b, (int, float)) and a is not None:
        try:
            return abs(float(a) - float(b)) < 0.5
        except (TypeError, ValueError):
            return False
    return str(a).strip().lower().replace(" ", "") == str(b).strip().lower().replace(" ", "")


def main(folder):
    store.clear_all()
    rows = list(csv.DictReader(open(os.path.join(folder, "expected.csv"), encoding="utf-8")))
    tot_f = ok_f = ok_t = ok_d = unsafe = 0
    report = []
    print("%-28s %-9s %-7s %-14s %s" % ("file", "type", "fields", "decision", "notes"))
    for r in rows:
        rec = orchestrator.process(os.path.join(folder, r["file"]))
        exp = json.loads(r["fields"] or "{}")
        wrong = []
        for k, v in exp.items():
            got = rec["fields"].get(k, {}).get("value")
            tot_f += 1
            if same(got, v):
                ok_f += 1
            else:
                wrong.append("%s: got %r expected %r" % (k, got, v))
        dec = DEC.get(rec["status"], rec["status"])
        t_ok, d_ok = rec["doc_type"] == r["doc_type"], dec == r["decision"]
        ok_t += t_ok
        ok_d += d_ok
        if wrong and dec == "auto_approve":
            unsafe += 1
        notes = []
        if not t_ok:
            notes.append("type=%s" % rec["doc_type"])
        if not d_ok:
            notes.append("decision=%s (expected %s) because: %s" % (dec, r["decision"], "; ".join(rec["score"]["reasons"][:2])))
        notes += wrong
        print("%-28s %-9s %-7s %-14s %s" % (r["file"], "OK" if t_ok else "WRONG",
                                            "%d/%d" % (len(exp) - len(wrong), len(exp)), "OK" if d_ok else "DIFF", " | ".join(notes)))
        report.append({"file": r["file"], "type_ok": t_ok, "decision_ok": d_ok, "got_decision": dec, "wrong_fields": "; ".join(wrong)})
    n = len(rows)
    print("\nDocument type accuracy : %d/%d" % (ok_t, n))
    print("Field accuracy         : %d/%d (%.0f%%)" % (ok_f, tot_f, 100 * ok_f / max(1, tot_f)))
    print("Decision agreement     : %d/%d" % (ok_d, n))
    print("UNSAFE (wrong field AND auto-approved): %d  <- must be 0" % unsafe)
    with open(os.path.join(folder, "report.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, report[0].keys())
        w.writeheader()
        w.writerows(report)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "testset"))
