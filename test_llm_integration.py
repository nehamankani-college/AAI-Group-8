"""Tests for the optional LLM layer. No API key or network needed: the model is replaced by a stub.
Run:  python test_llm_integration.py"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings  # noqa: E402
warnings.filterwarnings("ignore")
from agents import extraction, ingestion, llm_agent, ocr, orchestrator, store  # noqa: E402
from agents.common import load_rules  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PASS_PDF = os.path.join(HERE, "samples", "Invoice_PASS_shree.pdf")
ITEMS = [{"description": "Steel Brackets 40mm", "qty": 100, "rate": 120, "amount": 12000},
         {"description": "Hex Bolts M10", "qty": 500, "rate": 18, "amount": 9000},
         {"description": "Mounting Plates", "qty": 50, "rate": 250, "amount": 12500},
         {"description": "Washers Assorted", "qty": 1000, "rate": 3.5, "amount": 3500},
         {"description": "Packaging and Freight", "qty": 1, "rate": 5300, "amount": 5300}]
TRUE = {"invoice_no": "INV-0902", "invoice_date": "2026-10-28", "due_date": "2026-11-27", "po_number": "PO-2291",
        "vendor": "Shree Traders", "gstin": "27ABCDE1234F1Z5", "bill_to": "Kalyani Engineering Pvt Ltd",
        "subtotal": 42300, "tax_total": 2115, "total": 44415, "currency": "INR", "signature": "signed", "line_items": ITEMS}

REPLY, CALLS = {}, []
results = []


def stub(content, system, max_tokens=2000):
    CALLS.append(system[:40])
    if isinstance(REPLY.get("raise"), Exception):
        raise REPLY["raise"]
    key = ("extract" if "extraction engine" in system else "classify" if "classify business" in system
           else "summary" if "briefings" in system else "rule")
    r = REPLY[key]
    return (r if isinstance(r, str) else json.dumps(r)), {"in": 10, "out": 5}


def use_llm(on=True):
    llm_agent._call = stub
    llm_agent.available = (lambda rules: (True, "model stub")) if on else (lambda rules: (False, "no key"))
    CALLS.clear()


def check(name, cond, extra=""):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name + ("  -> " + str(extra) if (extra and not cond) else ""))


def run(path=PASS_PDF):
    store.clear_all()
    return orchestrator.process(path)


# 1. no LLM -> identical behaviour, nothing extra
use_llm(False)
r = run()
check("no key: pipeline unchanged, auto-approved", r["status"] == "auto_approved" and not r["llm_flags"] and not CALLS, r["status"])
check("no key: audit explains why LLM was not used", any(e["agent"] == "llm" for e in r["audit"]))

# 2. model agrees on everything
use_llm()
REPLY.update(extract=TRUE, summary="n/a")
r = run()
check("agree: still auto-approved", r["status"] == "auto_approved", [f["message"] for f in r["flags"]])
check("agree: fields marked as confirmed", r["fields"]["total"]["llm"]["agree"] is True and r["fields"]["total"]["confidence"] >= 0.95)
check("agree: tokens counted", r["llm"]["tokens"]["in"] > 0)
check("agree: no briefing for an auto-approved doc", r["llm_summary"] is None)

# 3. model disagrees about the total (value NOT in OCR text) -> needs review
wrong = dict(TRUE, total=44451)
REPLY.update(extract=wrong, summary="The total differs between readers; check the total line.")
r = run()
codes = [f["code"] for f in r["flags"]]
check("disagree: flagged for review", r["status"] == "needs_review" and "llm_disagreement" in codes, (r["status"], codes))
f = next(f for f in r["flags"] if f["code"] == "llm_disagreement")
check("disagree: names both readings, no unsafe suggestion", "44451" in f["message"] and f["suggestion"] is None)
check("disagree: reviewer briefing stored", r["llm_summary"] and "total" in r["llm_summary"])

# 4. model agrees with OCR text but regex was wrong -> disagreement WITH a suggestion
ing = ingestion.ingest(PASS_PDF)
res = ocr.run_ocr(ing)
doc = extraction.Doc(res, [p["image"] for p in ing["pages"]])
fields, tables = extraction.extract("invoice", "tax_invoice", res, [p["image"] for p in ing["pages"]])
fields["po_number"]["value"] = "PO-9999"          # simulate a regex mistake
lf = {k: llm_agent._norm_value(k, v) for k, v in TRUE.items() if k != "line_items"}
flags, stats = llm_agent.apply(fields, tables, lf, ITEMS, doc)
fl = next(f for f in flags if f["field"] == "po_number")
check("regex wrong, model right (value is in OCR text): suggestion offered",
      fl["suggestion"] and fl["suggestion"]["value"] == "PO-2291", fl)

# 5. regex missed a field, model supplies it and it IS in the OCR text -> adopted with a source box
fields, tables = extraction.extract("invoice", "tax_invoice", res, [p["image"] for p in ing["pages"]])
del fields["po_number"]
flags, stats = llm_agent.apply(fields, tables, lf, ITEMS, doc)
check("missing field filled from model when verifiable in OCR text",
      "po_number" in stats["filled"] and fields["po_number"]["boxes"] and fields["po_number"]["method"] == "llm+ocr")

# 6. hallucination: model invents a value that is not in the document -> NOT adopted
fields, tables = extraction.extract("invoice", "tax_invoice", res, [p["image"] for p in ing["pages"]])
del fields["po_number"]
lf2 = dict(lf, po_number="PO-7777")
flags, stats = llm_agent.apply(fields, tables, lf2, ITEMS, doc)
check("hallucinated value is rejected and flagged", "po_number" not in fields and "po_number" in stats["unverified"]
      and any(f["code"] == "llm_unverified" for f in flags))

# 7. line-item disagreement
fields, tables = extraction.extract("invoice", "tax_invoice", res, [p["image"] for p in ing["pages"]])
flags, stats = llm_agent.apply(fields, tables, lf, ITEMS[:-1], doc)
check("line-item mismatch flagged", any(f["code"] == "llm_line_items" for f in flags))

# 8. identity documents are not sent unless allowed
use_llm()
REPLY.update(extract={}, summary="x")
store.clear_all()
r = orchestrator.process(os.path.join(HERE, "samples", "PAN_card_supriya.jpg"))
check("KYC not sent to the API by default", not any("extraction engine" in c for c in CALLS)
      and any("skipped for KYC" in e["action"] for e in r["audit"]))

# 9. failure of the model never breaks processing
use_llm()
REPLY["raise"] = RuntimeError("network down")
r = run()
check("LLM failure: document still processed", r["status"] == "auto_approved")
check("LLM failure: logged in audit trail", any("failed" in e["action"] for e in r["audit"]))
REPLY.pop("raise")

# 10. bad JSON from the model is survivable
REPLY["extract"] = "Sure! Here you go: not json"
r = run()
check("garbled model reply: document still processed", r["status"] == "auto_approved")

# 11. classification fallback (text only) + rule parsing
REPLY["classify"] = {"doc_type": "invoice", "subtype": None, "confidence": 0.8, "reason": "has line items"}
c, _ = llm_agent.classify("some unclear text")
check("classification fallback parses reply", c["doc_type"] == "invoice" and c["source"] == "llm" and c["confidence"] <= 0.9)
REPLY["rule"] = {"doc_type": "claim", "field": "claim_amount", "op": ">", "value": "100000", "severity": "high", "message": "Big claim"}
check("plain-English rule parsed", llm_agent.parse_rule("flag claims above 100000")["field"] == "claim_amount")
for bad, why in [({"doc_type": "claim", "field": "password", "op": ">", "value": "1"}, "unknown field"),
                 ({"doc_type": "claim", "field": "claim_amount", "op": "contains", "value": "1"}, "bad operator"),
                 ({"doc_type": "claim", "field": "claim_amount", "op": ">", "value": ""}, "no value"),
                 ({"error": "cannot express"}, "model error")]:
    REPLY["rule"] = bad
    try:
        llm_agent.parse_rule("x")
        check("rule validation rejects: " + why, False)
    except ValueError:
        check("rule validation rejects: " + why, True)

# 12. the REAL anthropic SDK against a local fake server (checks request shape + response handling)
try:
    import anthropic  # noqa: F401
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(path=self.path, body=body, key=self.headers.get("x-api-key"))
            out = json.dumps({"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                              "content": [{"type": "text", "text": json.dumps(TRUE)}], "stop_reason": "end_turn",
                              "stop_sequence": None, "usage": {"input_tokens": 1234, "output_tokens": 56}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ.update(ANTHROPIC_API_KEY="test-key", ANTHROPIC_BASE_URL="http://127.0.0.1:%d" % srv.server_port)
    llm_agent._CLIENT.clear()
    import importlib
    importlib.reload(llm_agent)                    # restore the real _call / available
    fields_out, items_out, usage = llm_agent.extract("invoice", [p["image"] for p in ing["pages"]], load_rules())
    blocks = seen["body"]["messages"][0]["content"]
    check("real SDK: request goes to /v1/messages with model, system and image blocks",
          seen["path"].endswith("/v1/messages") and seen["body"]["model"] == llm_agent.model()
          and seen["body"]["system"] and any(b["type"] == "image" and b["source"]["media_type"] == "image/png" for b in blocks),
          seen.get("path"))
    check("real SDK: JSON reply parsed, usage counted", fields_out["total"] == 44415.0 and len(items_out) == 5
          and usage == {"in": 1234, "out": 56})
    srv.shutdown()
except ImportError:
    print("SKIP  real-SDK test (pip install anthropic to run it)")

# 13. Gemini (free-tier) provider against a fake server
class G(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.update(gpath=self.path, gbody=body, gkey=self.headers.get("x-goog-api-key"))
        out = json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps(TRUE)}]}}],
                          "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 40}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


seen = {}
gs = HTTPServer(("127.0.0.1", 0), G)
threading.Thread(target=gs.serve_forever, daemon=True).start()
for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
    os.environ.pop(k, None)
os.environ.update(GEMINI_API_KEY="g-test", DOCAI_GEMINI_BASE="http://127.0.0.1:%d" % gs.server_port)
import importlib
importlib.reload(llm_agent)
check("gemini: chosen automatically when only GEMINI_API_KEY is set", llm_agent.provider() == "gemini" and llm_agent.available({})[0])
fo, io_, us = llm_agent.extract("invoice", [p["image"] for p in ing["pages"]], load_rules())
gp = seen["gbody"]["contents"][0]["parts"]
check("gemini: request has key header, system instruction and inline images",
      ":generateContent" in seen["gpath"] and seen["gkey"] == "g-test" and seen["gbody"]["system_instruction"]
      and any("inline_data" in x for x in gp), seen.get("gpath"))
check("gemini: reply parsed, usage counted", fo["total"] == 44415.0 and len(io_) == 5 and us == {"in": 900, "out": 40})
gs.shutdown()

store.clear_all()
print("\n%d/%d checks passed" % (sum(results), len(results)))
sys.exit(0 if all(results) else 1)
