"""
DocAgent — 10-Agent Pipeline for Document Intelligence & Extraction (PS-15)
===========================================================================
Lab: Agentic AI & Enterprise Automation

Each agent in this package is an autonomous specialist with a single,
well-defined role. Agents communicate through structured data contracts
(dicts / lists) and are orchestrated by Agent 10 (orchestrator.py).
No agent can approve or reject a document on its own — final authority
rests either with the scoring engine (auto-approve path) or a human
reviewer (human-in-the-loop path).

AGENT MANIFEST
--------------
 #  Module            Role                                    Decision Authority
 1  ingestion         Load, deskew, denoise, quality-score    None (pre-processing)
 2  classification    Identify doc type + subtype             Routing only
 3  ocr               Extract text + bounding boxes           None (sensing)
 4  extraction        Pull typed fields per doc template      None (data only)
 5  validation        Cross-field arithmetic & date checks    Raise flags
 6  rules_agent       Business-rule & reference-data checks   Raise flags
 --  llm_agent        Vision cross-check, NL rule parser,     Raise flags only
                      reviewer briefing (optional, graceful)
 7  scoring           Aggregate flags → confidence + decision Auto-approve / hold
 8  review            Route to human queue with SLA           None (routing)
 9  output            Write JSON / CSV / Markdown outputs     None (I/O)
10  orchestrator      Coordinate all agents, audit trail      Orchestration only

Human reviewer: approve · reject · correct a field (re-runs agents 5-9)
"""

AGENT_MANIFEST = [
    {"id": 1,  "module": "ingestion",       "role": "Document Ingestion",          "authority": "none"},
    {"id": 2,  "module": "classification",  "role": "Document Classification",     "authority": "routing"},
    {"id": 3,  "module": "ocr",             "role": "OCR & Layout Analysis",        "authority": "none"},
    {"id": 4,  "module": "extraction",      "role": "Field Extraction",            "authority": "none"},
    {"id": 5,  "module": "validation",      "role": "Cross-Field Validation",      "authority": "flags"},
    {"id": 6,  "module": "rules_agent",     "role": "Business Rule Enforcement",   "authority": "flags"},
    {"id": 0,  "module": "llm_agent",       "role": "LLM Vision Cross-Check",      "authority": "flags-only"},
    {"id": 7,  "module": "scoring",         "role": "Confidence Scoring & Decision","authority": "auto-approve"},
    {"id": 8,  "module": "review",          "role": "Human Review Routing",        "authority": "routing"},
    {"id": 9,  "module": "output",          "role": "Structured Output Generation","authority": "none"},
    {"id": 10, "module": "orchestrator",    "role": "Pipeline Orchestration",      "authority": "orchestration"},
]

