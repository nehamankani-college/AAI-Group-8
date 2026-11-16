"""Shared helpers: paths, config, audit trail, small parsers."""
import datetime as dt
import json
import os
import re
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)

# Auto-load .env if present
_env_path = os.path.join(BASE, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())


def load_json(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def load_rules():
    return load_json("rules.json")


def save_rules(rules):
    with open(os.path.join(DATA, "rules.json"), "w", encoding="utf-8") as fh:
        json.dump(rules, fh, indent=2, ensure_ascii=False)


class Audit:
    """Append-only audit trail. Every agent logs what it did and why."""

    def __init__(self, events=None):
        self.events = events if events is not None else []

    def log(self, agent, action, **detail):
        self.events.append({"ts": round(time.time(), 3), "agent": agent, "action": action, "detail": detail})


_DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y", "%Y-%m-%d"]


def parse_date(s):
    """Indian convention: day first. Returns datetime.date or None."""
    if not s:
        return None
    s = str(s).strip()
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(s):
    if s is None:
        return None
    s = re.sub(r"[^\d.]", "", str(s).replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return None


def today(rules=None):
    if rules and rules.get("as_of"):
        return parse_date(rules["as_of"])
    return dt.date.today()


CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}
_CUR = {"sym": "₹"}


def set_currency(code):
    _CUR["sym"] = CURRENCY_SYMBOLS.get(code, "₹")


def money(x):
    return "%s{:,.2f}".format(x) % _CUR["sym"] if x is not None else "n/a"


_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def ascii_digits(s):
    return s.translate(_DEVANAGARI_DIGITS)


def make_field(value, hit=None, raw=None, factor=1.0, method="regex", conf=None):
    """A single extracted field with provenance back to the page + bounding boxes."""
    f = {"value": value, "raw": raw if raw is not None else value, "method": method}
    if hit is not None:
        f.update(confidence=round(min(1.0, hit["conf"] * factor), 3), page=hit["page"],
                 bbox=hit["bbox"], boxes=hit["boxes"])
    else:
        f.update(confidence=conf if conf is not None else 0.0, page=None, bbox=None, boxes=[])
    return f
