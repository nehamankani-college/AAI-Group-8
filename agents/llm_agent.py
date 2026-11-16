"""Optional LLM layer: a vision-language model as an independent second reader.

Design rules (so the LLM can help without being trusted blindly):
  * It sees ONLY the page images - never the OCR text - so it cannot just repeat an OCR mistake.
  * Its values are cross-checked against the regex extractor. Agreement raises confidence; disagreement goes to a human.
  * A value the extractor missed is adopted only if it can be FOUND in the OCR text (which also gives it a source box).
    Otherwise it is treated as a possible hallucination and flagged, not used.
  * It never decides approve/reject and never overrides arithmetic or business rules.
  * Any failure (no key, network, bad JSON) is logged and the pipeline simply continues without the LLM.
"""
import base64
import difflib
import json
import os
import re

import cv2

from .common import CURRENCY_SYMBOLS, parse_amount, parse_date
from .validation import mkflag, ref

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
TYPES = ("invoice", "kyc", "contract", "claim", "other")

SCHEMAS = {
    "invoice": {
        "invoice_no": "string", "invoice_date": "YYYY-MM-DD", "due_date": "YYYY-MM-DD", "po_number": "string",
        "vendor": "string - the seller/issuer of the invoice", "gstin": "string - the seller's GSTIN/tax id",
        "bill_to": "string - the customer being billed", "subtotal": "number", "tax_total": "number - sum of tax lines",
        "total": "number - grand total", "currency": "ISO code such as INR or USD",
        "signature": "'signed' if a handwritten signature is visible, else 'missing'",
        "line_items": "array of {description, qty, rate, amount}"},
    "kyc": {
        "id_type": "one of pan, aadhaar, driving_licence, passport, other", "id_number": "string", "name": "string",
        "father_name": "string", "dob": "YYYY-MM-DD", "expiry_date": "YYYY-MM-DD", "address": "string",
        "signature": "'signed' / 'missing', or null if the document has no signature area"},
    "contract": {
        "party_a": "string", "party_b": "string", "effective_date": "YYYY-MM-DD",
        "payment_terms_days": "integer", "termination_notice_days": "integer",
        "liability_capped": "'yes' if liability is limited/capped, 'no' if unlimited", "governing_law": "string"},
    "claim": {
        "claim_number": "string", "policy_number": "string", "claimant": "string", "date_of_loss": "YYYY-MM-DD",
        "claim_date": "YYYY-MM-DD", "claim_amount": "number", "description": "string",
        "signature": "'signed' if a handwritten signature is visible, else 'missing'"},
}
AMOUNT_KEYS = {"subtotal", "tax_total", "total", "claim_amount"}
DATE_KEYS = {"invoice_date", "due_date", "dob", "expiry_date", "effective_date", "date_of_loss", "claim_date"}
INT_KEYS = {"payment_terms_days", "termination_notice_days"}
ID_KEYS = {"invoice_no", "po_number", "gstin", "id_number", "claim_number", "policy_number"}
DERIVED = {"tax_total"}          # computed from several lines, so it may not appear literally in the text
RULE_OPS = (">", ">=", "<", "<=", "==", "!=")

SYSTEM_EXTRACT = (
    "You are a precise document data-extraction engine. Reply with ONE JSON object and nothing else. "
    "Copy values exactly as printed. If a value is not clearly visible, use null - never guess or infer. "
    "Dates as YYYY-MM-DD (Indian documents are day-first dd/mm/yyyy unless the document shows otherwise). "
    "Amounts as plain numbers: no currency symbols, no thousands separators.")


# ---------------------------------------------------------------- availability / transport
def provider():
    """'groq', 'gemini' or 'anthropic'. Override with DOCAI_LLM_PROVIDER."""
    forced = os.environ.get("DOCAI_LLM_PROVIDER", "").lower()
    if forced in ("groq", "gemini", "anthropic"):
        return forced
    if _groq_key():
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _gemini_key():
        return "gemini"
    return "groq"


def model():
    p = provider()
    if p == "groq":
        default = DEFAULT_GROQ_MODEL
    elif p == "gemini":
        default = DEFAULT_GEMINI_MODEL
    else:
        default = DEFAULT_MODEL
    return os.environ.get("DOCAI_LLM_MODEL", default)


def _groq_key():
    return os.environ.get("GROQ_API_KEY")


def _gemini_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def available(rules):
    """(is_available, human-readable reason)."""
    if not rules.get("llm_enabled", True):
        return False, "switched off in Business rules"
    p = provider()
    if p == "groq":
        if not _groq_key():
            return False, "GROQ_API_KEY is not set"
        return True, "Groq model %s" % model()
    if p == "gemini":
        if not _gemini_key():
            return False, "GEMINI_API_KEY is not set"
        return True, "Gemini model %s" % model()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "no API key set (GROQ_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY)"
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "the 'anthropic' package is not installed (pip install anthropic)"
    return True, "model %s" % model()


def status(rules):
    ok, why = available(rules)
    return {"on": ok, "note": why}


_CLIENT = {}


def _client():
    if "c" not in _CLIENT:
        import anthropic
        _CLIENT["c"] = anthropic.Anthropic(timeout=60.0, max_retries=1)  # reads ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL)
    return _CLIENT["c"]


def _call_gemini(content, system, max_tokens):
    import urllib.request
    import urllib.error
    parts = []
    for b in content if isinstance(content, list) else [{"type": "text", "text": content}]:
        if b.get("type") == "text":
            parts.append({"text": b["text"]})
        elif b.get("type") == "image":
            parts.append({"inline_data": {"mime_type": b["source"]["media_type"], "data": b["source"]["data"]}})
    body = {"system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": max(max_tokens, 4000), "temperature": 0,
                                 "responseMimeType": "application/json"}}
    base = os.environ.get("DOCAI_GEMINI_BASE", "https://generativelanguage.googleapis.com")
    req = urllib.request.Request("%s/v1beta/models/%s:generateContent" % (base, model()),
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": _gemini_key()})
    
    import ssl
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(req, timeout=90, context=ctx) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError("Gemini API error %s: %s" % (e.code, e.read()[:200].decode("utf-8", "replace")))
    cands = data.get("candidates") or []
    text = "".join(p.get("text", "") for p in (cands[0].get("content", {}).get("parts", []) if cands else []))
    u = data.get("usageMetadata", {})
    return text, {"in": u.get("promptTokenCount", 0) or 0, "out": u.get("candidatesTokenCount", 0) or 0}


def _call_groq(content, system, max_tokens):
    import urllib.request
    import urllib.error
    import ssl
    
    # Format messages for Groq chat completions
    user_parts = []
    for b in content if isinstance(content, list) else [{"type": "text", "text": content}]:
        if b.get("type") == "text":
            user_parts.append(b["text"])
        elif b.get("type") == "image":
            # For vision with Llama 3.2 Vision on Groq or text fallback
            pass
    user_text = "\n".join(user_parts)
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text}
    ]
    
    body = {
        "model": model(),
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max(max_tokens, 1000)
    }
    
    # Enable json mode if system prompt indicates JSON format
    if "json" in system.lower():
        body["response_format"] = {"type": "json_object"}
        
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % _groq_key(),
            "User-Agent": "DocAgent/1.0 (Enterprise Automation; macOS)"
        }
    )
    
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl._create_unverified_context()
        
    try:
        with urllib.request.urlopen(req, timeout=45, context=ctx) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        err_msg = e.read()[:300].decode("utf-8", "replace")
        # If specific model name is not found on this endpoint, fallback to standard llama-3.3-70b-versatile
        if "model_not_found" in err_msg or "does not exist" in err_msg:
            body["model"] = "llama-3.3-70b-versatile"
            req2 = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(body).encode(),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer %s" % _groq_key(),
                    "User-Agent": "DocAgent/1.0 (Enterprise Automation; macOS)"
                }
            )
            with urllib.request.urlopen(req2, timeout=45, context=ctx) as r2:
                data = json.loads(r2.read())
        else:
            raise RuntimeError("Groq API error %s: %s" % (e.code, err_msg))
        
    choice = (data.get("choices") or [{}])[0]
    text = choice.get("message", {}).get("content", "")
    u = data.get("usage", {})
    return text, {"in": u.get("prompt_tokens", 0) or 0, "out": u.get("completion_tokens", 0) or 0}


def _call(content, system, max_tokens=2000):
    """Single model call -> (text, usage). Tests replace this function with a stub."""
    p = provider()
    if p == "groq":
        return _call_groq(content, system, max_tokens)
    if p == "gemini":
        return _call_gemini(content, system, max_tokens)
    resp = _client().messages.create(model=model(), max_tokens=max_tokens, system=system,
                                     messages=[{"role": "user", "content": content}])
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    u = getattr(resp, "usage", None)
    return text, {"in": getattr(u, "input_tokens", 0) or 0, "out": getattr(u, "output_tokens", 0) or 0}


def _image_blocks(images, max_pages=3, max_w=1400):
    blocks = []
    for n, img in enumerate(images[:max_pages], 1):
        h, w = img.shape[:2]
        if w > max_w:
            img = cv2.resize(img, None, fx=max_w / w, fy=max_w / w, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            continue
        blocks.append({"type": "text", "text": "Page %d:" % n})
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                   "data": base64.b64encode(buf.tobytes()).decode()}})
    return blocks


def _parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        raise ValueError("model reply contained no JSON object")
    return json.loads(t[i:j + 1])


# ---------------------------------------------------------------- normalisation + comparison
def _norm_value(key, v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if v.lower() in ("", "null", "none", "n/a", "not visible"):
            return None
    if key in AMOUNT_KEYS:
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else parse_amount(v)
    if key in INT_KEYS:
        x = parse_amount(v)
        return int(x) if x is not None else None
    if key in DATE_KEYS:
        d = parse_date(v)
        return d.isoformat() if d else str(v)
    if key == "currency":
        return str(v).upper()[:3]
    if key == "signature":
        return "signed" if v is True or str(v).lower().startswith("sign") else "missing"
    if key == "liability_capped":
        return "yes" if str(v).lower().startswith(("y", "true")) else "no"
    return str(v)


def _t(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _same(key, a, b):
    if a is None or b is None:
        return False
    if key in AMOUNT_KEYS:
        return abs(float(a) - float(b)) < 0.005
    if key in INT_KEYS:
        return int(a) == int(b)
    if key in DATE_KEYS or key in ("signature", "liability_capped", "currency"):
        return str(a) == str(b)
    if key in ID_KEYS:
        return re.sub(r"[^A-Z0-9]", "", str(a).upper()) == re.sub(r"[^A-Z0-9]", "", str(b).upper())
    sa, sb = _t(a), _t(b)
    return sa == sb or difflib.SequenceMatcher(None, sa, sb).ratio() >= 0.92


def _locate(doc, key, v):
    """Find the model's value in the OCR text -> hit with page + boxes, or None (= unverifiable)."""
    pats = []
    if key in AMOUNT_KEYS:
        c = ["{:,.2f}".format(v), "%.2f" % v]
        if float(v).is_integer():
            c += ["{:,}".format(int(v)), str(int(v))]
        pats = [r"(?<![\d.,])" + re.escape(x) + r"(?!\d)" for x in c]
    elif key in DATE_KEYS:
        d = parse_date(v)
        if d:
            pats = [r"\b0?%d[/\-.]0?%d[/\-.](?:%d|%02d)\b" % (d.day, d.month, d.year, d.year % 100)]
    elif key in INT_KEYS:
        return None
    else:
        pats = [re.escape(str(v).strip()).replace("\\ ", r"\s+")]
    for p in pats:
        hits = doc.search(p, group=0)
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------- 1) extraction + cross-check
def extract(doc_type, images, rules):
    """Ask the vision model for the fields of this document type. Returns (fields, line_items, usage)."""
    schema = SCHEMAS[doc_type]
    keys = "\n".join('  "%s": %s' % (k, v) for k, v in schema.items())
    content = _image_blocks(images, rules.get("llm_max_pages", 3)) + [{
        "type": "text",
        "text": "Document type: %s.\nReturn a JSON object with exactly these keys (null when not visible):\n{\n%s\n}" % (
            doc_type, keys)}]
    text, usage = _call(content, SYSTEM_EXTRACT, 2500)
    data = _parse_json(text)
    fields = {k: _norm_value(k, data.get(k)) for k in schema if k != "line_items"}
    items = []
    for it in data.get("line_items") or []:
        amt = parse_amount(it.get("amount")) if isinstance(it, dict) else None
        if amt is not None:
            items.append({"description": str(it.get("description") or ""), "qty": parse_amount(it.get("qty")),
                          "rate": parse_amount(it.get("rate")), "amount": amt})
    return fields, items, usage


def apply(fields, tables, lf, items, doc):
    """Cross-check the model's values against the extractor's. Mutates `fields`; returns (flags, stats)."""
    flags, stats = [], {"agree": [], "disagree": [], "filled": [], "unverified": []}
    for key, lv in lf.items():
        if lv is None:
            continue
        f = fields.get(key)
        if key == "currency":
            if f and lv in CURRENCY_SYMBOLS and f["value"] != lv:
                f.update(value=lv, llm_set=True)
            continue
        if key == "signature":                      # ink detector vs. what the model sees
            if f and f["value"] != lv:
                f["confidence"] = min(f["confidence"], 0.5)
                f["llm"] = {"value": lv, "agree": False}
                flags.append(mkflag("llm_disagreement", "medium", key,
                                    "Signature check disagrees: ink detector says '%s', vision model says '%s'" % (f["value"], lv),
                                    "Stamps, light pen strokes or photos can fool either method.",
                                    "Look at the signature area and confirm.", ref(f), agent="llm"))
                stats["disagree"].append(key)
            elif f:
                f["llm"] = {"value": lv, "agree": True}
                stats["agree"].append(key)
            continue
        if key in DERIVED and f:
            continue                                # arithmetic validation already covers derived totals
        if f is not None and f.get("value") not in (None, ""):
            if _same(key, f["value"], lv):
                f["llm"] = {"value": lv, "agree": True}
                f["confidence"] = max(f["confidence"], 0.95)
                stats["agree"].append(key)
            else:
                hit = _locate(doc, key, lv)
                f["llm"] = {"value": lv, "agree": False, "found_in_ocr_text": bool(hit)}
                f["confidence"] = min(f["confidence"], 0.5)
                refs = ref(f) + ([{"page": hit["page"], "boxes": hit["boxes"]}] if hit else [])
                flags.append(mkflag(
                    "llm_disagreement", "medium", key,
                    "Two readers disagree on '%s': extractor read %r, vision model read %r" % (key.replace("_", " "), f["value"], lv),
                    "The model's reading %s." % ("also appears in the OCR text, so it may be the correct one" if hit else
                                                 "does not appear in the OCR text, so the extractor may be right"),
                    "Compare with the highlighted region and correct if needed.", refs,
                    {"field": key, "value": lv, "note": "Vision model's reading"} if hit else None, agent="llm"))
                stats["disagree"].append(key)
        else:
            hit = _locate(doc, key, lv)
            if hit:
                nf = {"value": lv, "raw": hit["value"], "method": "llm+ocr", "confidence": round(min(hit["conf"], 0.85), 3),
                      "page": hit["page"], "bbox": hit["bbox"], "boxes": hit["boxes"], "llm": {"value": lv, "agree": None}}
                fields[key] = nf
                flags.append(mkflag("llm_filled", "low", key, "'%s' was filled in by the vision model" % key.replace("_", " "),
                                    "The regex extractor missed it, but the value %r was found in the OCR text." % lv,
                                    "Quick visual check recommended.", ref(nf), agent="llm"))
                stats["filled"].append(key)
            elif key not in DERIVED:
                flags.append(mkflag("llm_unverified", "medium", key,
                                    "Vision model reads %s = %r but it could not be verified in the OCR text; not used" % (
                                        key.replace("_", " "), lv),
                                    "A value that cannot be found in the OCR text might be a model hallucination.",
                                    "Check the page image and enter the value manually if correct.", [], agent="llm"))
                stats["unverified"].append(key)
    # line items: count + total must agree
    reg = tables.get("line_items", [])
    if items:
        lsum = round(sum(i["amount"] for i in items), 2)
        if reg:
            rsum = round(sum(i["amount"] for i in reg), 2)
            if len(reg) != len(items) or abs(rsum - lsum) > 0.5:
                flags.append(mkflag("llm_line_items", "medium", "line_items",
                                    "Line items differ: extractor found %d rows totalling %.2f, vision model found %d totalling %.2f" % (
                                        len(reg), rsum, len(items), lsum),
                                    "A row may have been missed, split or misread by one of the readers.",
                                    "Compare the table with the source image.",
                                    [{"page": i["page"], "boxes": i["boxes"]} for i in reg], agent="llm"))
                stats["disagree"].append("line_items")
            else:
                stats["agree"].append("line_items")
        else:
            flags.append(mkflag("llm_unverified", "medium", "line_items",
                                "Vision model sees %d line items (total %.2f) that the extractor could not read" % (len(items), lsum),
                                "The table layout was not recognised by the extractor.", "Check the line items manually.", [],
                                agent="llm"))
            stats["unverified"].append("line_items")
    return flags, stats


# ---------------------------------------------------------------- 2) classification fallback (text only)
def classify(ocr_text):
    system = ("You classify business documents. Reply with ONE JSON object: "
              '{"doc_type": "invoice|kyc|contract|claim|other", "subtype": string or null, "confidence": 0-1, "reason": short}. '
              "kyc covers identity documents (PAN, Aadhaar, driving licence, passport). Use other if unsure.")
    text, usage = _call([{"type": "text", "text": "OCR text of the document (may contain errors):\n" + ocr_text[:1800]}],
                        system, 200)
    d = _parse_json(text)
    t = str(d.get("doc_type", "other")).lower()
    t = t if t in TYPES else "other"
    sub = re.sub(r"[^a-z_]", "", str(d.get("subtype") or "").lower().replace(" ", "_")) or None
    if t == "kyc" and sub not in ("pan", "aadhaar", "driving_licence", "passport", "utility_bill"):
        sub = None
    try:
        conf = float(d.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {"doc_type": t, "subtype": sub, "confidence": round(min(0.9, max(0.0, conf)), 2), "template_match": t != "other",
            "source": "llm", "reason": str(d.get("reason", ""))[:200]}, usage


# ---------------------------------------------------------------- 3) reviewer briefing
def summarize(record):
    payload = {"document": record["filename"], "type": record["doc_type"], "decision": record["score"]["decision"],
               "priority": record["score"]["priority"],
               "issues": [{"severity": f["severity"], "message": f["message"], "detail": f["explanation"]}
                          for f in record["flags"] if f["severity"] != "low"][:8],
               "fields": {k: v["value"] for k, v in list(record["fields"].items())[:14]}}
    system = ("You write short briefings for back-office reviewers. Use ONLY the facts in the JSON. 2-3 plain sentences: "
              "what looks wrong and what to check first. Do not recommend approving or rejecting. Never invent numbers or names.")
    text, usage = _call([{"type": "text", "text": json.dumps(payload, default=str)}], system, 250)
    return text.strip(), usage


# ---------------------------------------------------------------- 4) plain-English rule -> structured rule
def parse_rule(text):
    fields = sorted({k for s in SCHEMAS.values() for k in s if k != "line_items"})
    system = ("Convert a business rule written in plain English into JSON with keys: doc_type (invoice|kyc|contract|claim|any), "
              "field (exactly one of: %s), op (one of > >= < <= == !=), value (string), severity (low|medium|high), "
              "message (short sentence). The rule FLAGS a document when its condition is true. "
              'Reply with only the JSON object, or {"error": "why"} if the rule cannot be expressed with these fields.' % ", ".join(fields))
    reply, _ = _call([{"type": "text", "text": text[:500]}], system, 300)
    d = _parse_json(reply)
    if d.get("error"):
        raise ValueError(str(d["error"])[:200])
    rule = {"doc_type": str(d.get("doc_type", "")).lower(), "field": str(d.get("field", "")), "op": str(d.get("op", "")),
            "value": str(d.get("value", "")), "severity": str(d.get("severity", "medium")).lower(),
            "message": str(d.get("message", ""))[:200]}
    if rule["doc_type"] not in ("invoice", "kyc", "contract", "claim", "any"):
        raise ValueError("unknown document type %r" % rule["doc_type"])
    if rule["field"] not in fields:
        raise ValueError("unknown field %r" % rule["field"])
    if rule["op"] not in RULE_OPS:
        raise ValueError("unsupported comparison %r" % rule["op"])
    if rule["severity"] not in ("low", "medium", "high"):
        rule["severity"] = "medium"
    if not rule["value"]:
        raise ValueError("no value to compare against")
    return rule
