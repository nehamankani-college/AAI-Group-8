"""CLI: process every file in samples/ (or given paths) and print a decision table."""
import glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents import orchestrator, store

def run(paths, clear=True):
    if clear:
        store.clear_all()
    recs = []
    for p in paths:
        recs.append(orchestrator.process(p))
    return recs

if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(__file__), "samples", "*")))
    paths = [p for p in paths if not (p.endswith("PAN_card_supriya.pdf"))]
    for r in run(paths):
        s = r["score"]
        print("%-28s %-9s %-14s conf=%.2f prio=%-6s %.1fs" % (r["filename"], r["doc_type"], r["status"], s["confidence"], s["priority"], r["processing_seconds"]))
        for f in r["flags"]:
            print("     [%s] %s" % (f["severity"], f["message"]))
