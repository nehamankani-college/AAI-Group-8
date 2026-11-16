"""Agent 5 - Cross-Field Validation: internal consistency within one document."""
import itertools

from .common import money, parse_date, today

REQUIRED = {
    "invoice": ["vendor", "invoice_no", "invoice_date", "total"],
    "kyc": ["name", "id_number", "dob"],
    "contract": ["party_a", "party_b", "effective_date"],
    "claim": ["claim_number", "policy_number", "claimant", "claim_amount", "date_of_loss"],
}
SEV_ORDER = {"high": 3, "medium": 2, "low": 1}


def mkflag(code, severity, field, message, explanation, recommendation, refs=None, suggestion=None, agent="validation"):
    return {"code": code, "severity": severity, "field": field, "message": message, "explanation": explanation,
            "recommendation": recommendation, "refs": refs or [], "suggestion": suggestion, "agent": agent}


def ref(obj):
    if not obj or not obj.get("boxes"):
        return []
    return [{"page": obj["page"], "boxes": obj["boxes"]}]


def val(fields, k):
    f = fields.get(k)
    return f["value"] if f else None


def validate(record, rules):
    dtype, F, T = record["doc_type"], record["fields"], record["tables"]
    flags = []
    tol = rules["amount_tolerance"]

    if dtype in ("other",):
        return [mkflag("unknown_doc_type", "high", None, "Document does not match any known template",
                       "Classifier could not identify an invoice, KYC document, contract or claim.",
                       "Send to manual triage.")]

    # required fields
    for k in REQUIRED.get(dtype, []):
        if k not in F or F[k]["value"] in (None, ""):
            sev = "high" if k in ("total", "invoice_no", "id_number", "claim_amount") else "medium"
            flags.append(mkflag("missing_field", sev, k, "Required field '%s' not found" % k.replace("_", " "),
                                "The extractor found no value for this field; nothing was guessed.",
                                "Check the page image and enter the value manually."))
    # unparseable dates
    for k, f in F.items():
        if f.get("parse_failed"):
            flags.append(mkflag("bad_date", "medium", k, "Date '%s' could not be parsed" % f["raw"],
                                "Text does not look like a valid dd/mm/yyyy date (possible OCR error).",
                                "Confirm the date against the source image.", ref(f)))
    # low-confidence fields (never silently accepted)
    for k, f in F.items():
        if f.get("method") in ("classifier", "sum-of-rows"):
            continue
        if f["confidence"] < rules["low_field_confidence"]:
            flags.append(mkflag("low_confidence", "medium", k,
                                "Low OCR confidence on '%s' (%.2f)" % (k.replace("_", " "), f["confidence"]),
                                "Value '%s' was read with low confidence; it may be wrong or illegible." % f["value"],
                                "Verify against the highlighted region.", ref(f), agent="confidence"))

    # signatures
    if rules.get("require_signatures"):
        for k, f in F.items():
            if k.startswith("signature") and f["value"] == "missing":
                who = f.get("label", "signature")
                flags.append(mkflag("missing_signature", "high" if dtype in ("contract", "claim") else "medium", k,
                                    "Missing signature: %s" % who.strip(" _:;.|-"),
                                    "No ink detected above the signature line.",
                                    "Request a signed copy before processing.", ref(f)))

    if dtype == "invoice":
        flags += _invoice(F, T, rules, tol)
    elif dtype == "kyc":
        flags += _kyc(F, rules)
    elif dtype == "contract":
        pass  # clause-level checks live in the business-rule agent (template comparison)
    elif dtype == "claim":
        d, c = parse_date(val(F, "date_of_loss")), parse_date(val(F, "claim_date"))
        if d and c and c < d:
            flags.append(mkflag("date_logic", "high", "claim_date", "Claim dated before the date of loss",
                                "Claim date %s precedes loss date %s." % (c, d), "Verify both dates.",
                                ref(F["claim_date"]) + ref(F["date_of_loss"])))
        if d and d > today(rules):
            flags.append(mkflag("date_logic", "high", "date_of_loss", "Date of loss is in the future", str(d),
                                "Verify the date.", ref(F["date_of_loss"])))
        amt = val(F, "claim_amount")
        if amt is not None and amt <= 0:
            flags.append(mkflag("bad_amount", "high", "claim_amount", "Claim amount is not positive", str(amt),
                                "Verify the amount.", ref(F["claim_amount"])))
    return flags


def _invoice(F, T, rules, tol):
    flags = []
    items, tax_rows = T.get("line_items", []), T.get("tax_rows", [])
    items_sum = round(sum(i["amount"] for i in items), 2) if items else None
    subtotal, total = val(F, "subtotal"), val(F, "total")
    tax_total = val(F, "tax_total") or 0.0

    for i in items:
        if i.get("qty") and i.get("rate") and abs(i["qty"] * i["rate"] - i["amount"]) > tol:
            flags.append(mkflag("line_math", "medium", "line_items",
                                "Line '%s': %g x %g = %s, but amount is %s" % (
                                    i["description"], i["qty"], i["rate"], money(i["qty"] * i["rate"]), money(i["amount"])),
                                "Quantity x rate does not equal the line amount.", "Check qty, rate and amount.",
                                [{"page": i["page"], "boxes": i["boxes"]}]))
    if items_sum is not None and subtotal is not None and abs(items_sum - subtotal) > tol:
        flags.append(mkflag("subtotal_mismatch", "high", "subtotal",
                            "Line-item sum (%s) does not match printed subtotal (%s)" % (money(items_sum), money(subtotal)),
                            "Discrepancy of %s between line items and the subtotal." % money(abs(items_sum - subtotal)),
                            "Hold for manual verification.", ref(F["subtotal"]) + [
                                {"page": i["page"], "boxes": i["boxes"]} for i in items]))
    base = subtotal if subtotal is not None else items_sum
    if base is not None and total is not None:
        expected = round(base + tax_total, 2)
        if abs(expected - total) > tol:
            diff = round(total - expected, 2)
            expl = "Subtotal %s + tax rows %s = %s, but the printed total is %s (%s %s)." % (
                money(base), money(tax_total), money(expected), money(total),
                "over by" if diff > 0 else "under by", money(abs(diff)))
            suggestion, extra_refs = None, []
            # is there a duplicated tax row that would explain it?
            dup = None
            for a, b in itertools.combinations(tax_rows, 2):
                if a["label"] == b["label"] and abs(a["amount"] - b["amount"]) < 0.01:
                    if abs(base + tax_total - a["amount"] - total) <= tol:
                        dup = (a, b)
                        break
            if dup:
                expl = ("Tax line '%s' of %s appears on two rows, inflating the tax total. %s With one row removed "
                        "the document reconciles to the printed total of %s.") % (
                    dup[0]["label"], money(dup[0]["amount"]), expl, money(total))
                suggestion = {"field": "tax_total", "value": round(tax_total - dup[0]["amount"], 2),
                              "note": "Remove duplicated tax row"}
                extra_refs = [{"page": r["page"], "boxes": r["boxes"]} for r in dup]
            flags.append(mkflag("total_mismatch", "high", "total",
                                "Totals do not reconcile: computed %s vs printed %s" % (money(expected), money(total)),
                                expl, "Hold for manual verification before payment approval; flagged rows highlighted.",
                                (ref(F["total"]) + extra_refs) or ref(F["total"]), suggestion))
    # tax duplicated even when totals happen to reconcile
    if len(tax_rows) >= 2 and not F.get("tax_total", {}).get("corrected") and not any(f["code"] == "total_mismatch" for f in flags):
        for a, b in itertools.combinations(tax_rows, 2):
            if a["label"] == b["label"] and abs(a["amount"] - b["amount"]) < 0.01:
                flags.append(mkflag("duplicate_tax_row", "medium", "tax_total",
                                    "Tax line %s appears twice" % money(a["amount"]), "Possible double-counted tax.",
                                    "Verify tax rows.", [{"page": r["page"], "boxes": r["boxes"]} for r in (a, b)]))
                break
    # date logic
    idt, ddt = parse_date(val(F, "invoice_date")), parse_date(val(F, "due_date"))
    if idt and ddt and ddt < idt:
        flags.append(mkflag("date_logic", "high", "due_date", "Due date is before the invoice date",
                            "Invoice %s, due %s." % (idt, ddt), "Verify both dates.",
                            ref(F["due_date"]) + ref(F["invoice_date"])))
    if idt and idt > today(rules):
        flags.append(mkflag("date_logic", "medium", "invoice_date", "Invoice date is in the future", str(idt),
                            "Verify the date.", ref(F["invoice_date"])))
    if ddt and ddt < today(rules):
        flags.append(mkflag("overdue", "low", "due_date", "Invoice is past its due date (%s)" % ddt,
                            "Payment is overdue as of %s." % today(rules), "Prioritise payment if approved.",
                            ref(F["due_date"])))
    return flags


def _kyc(F, rules):
    flags = []
    exp, dob = parse_date(val(F, "expiry_date")), parse_date(val(F, "dob"))
    if exp and exp < today(rules):
        flags.append(mkflag("expired_id", "high", "expiry_date", "ID expired on %s" % exp,
                            "Document validity ended %d days ago." % (today(rules) - exp).days,
                            "Reject and request a valid ID.", ref(F["expiry_date"])))
    if dob:
        age = (today(rules) - dob).days // 365
        if age < rules["min_customer_age"]:
            flags.append(mkflag("underage", "high", "dob", "Customer age %d is below the minimum %d" % (
                age, rules["min_customer_age"]), "Date of birth %s." % dob, "Escalate to compliance.", ref(F["dob"])))
        if dob > today(rules):
            flags.append(mkflag("date_logic", "high", "dob", "Date of birth is in the future", str(dob),
                                "Verify the date.", ref(F["dob"])))
    return flags
