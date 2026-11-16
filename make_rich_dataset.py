#!/usr/bin/env python3
"""
make_rich_dataset.py

Generates 25 synthetic Indian SME documents (PDF / JPG / PNG / TIFF) for testing a
document-processing pipeline. Writes rich_dataset/MANIFEST.csv, zips the folder into
rich_dataset.zip, and prints the manifest table.

Install:  pip3 install pillow numpy opencv-python reportlab
Run:      python3 make_rich_dataset.py

All names, numbers and IDs are fictional. Every document carries a SPECIMEN footer.
"""

import csv
import datetime as dt
import glob
import itertools
import os
import random
import shutil
import zipfile
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

OUT_DIR_NAME = "rich_dataset"
ZIP_NAME = "rich_dataset.zip"
MANIFEST_NAME = "MANIFEST.csv"
MASTER_SEED = 2026
TODAY = dt.date.today()
SPECIMEN_NOTE = "SPECIMEN - SYNTHETIC TEST DATA - NOT A REAL DOCUMENT"

# ============================================================== FONTS

REG_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",          # macOS
    "/Library/Fonts/Arial.ttf",                              # macOS
    "C:/Windows/Fonts/arial.ttf",                            # Windows
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",     # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # Linux DejaVu
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
]
BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/DejaVuSans-Bold.ttf",
]

CURRENCY = "Rs."          # switched to the rupee sign only if a TTF font loads
REG_FONT = None
BOLD_FONT = None
PDF_FONT = "Helvetica"
PDF_BOLD = "Helvetica-Bold"


def _first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def _any_ttf():
    patterns = ["/usr/share/fonts/**/*.ttf", "/Library/Fonts/*.ttf",
                "/System/Library/Fonts/**/*.ttf", "C:/Windows/Fonts/*.ttf"]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def setup_fonts():
    """Pick a TTF font (Arial -> DejaVu -> any TTF). Never crashes on a missing font."""
    global CURRENCY, REG_FONT, BOLD_FONT, PDF_FONT, PDF_BOLD
    REG_FONT = _first_existing(REG_CANDIDATES) or _any_ttf()
    BOLD_FONT = _first_existing(BOLD_CANDIDATES) or REG_FONT
    if REG_FONT is None:
        print("  ! No TTF font found - using built-in fonts and 'Rs.' instead of the rupee sign.")
        return
    try:
        pdfmetrics.registerFont(TTFont("DocReg", REG_FONT))
        pdfmetrics.registerFont(TTFont("DocBold", BOLD_FONT or REG_FONT))
        PDF_FONT, PDF_BOLD = "DocReg", "DocBold"
        CURRENCY = "₹"
        print(f"  Font: {REG_FONT}")
    except Exception as exc:
        print(f"  ! Could not register {REG_FONT} ({exc}); using Helvetica and 'Rs.'")
        REG_FONT = None


def pil_font(size, bold=False):
    path = BOLD_FONT if bold else REG_FONT
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


# ============================================================== FORMATTING

def inr(amount):
    """Indian digit grouping: 150000 -> '1,50,000.00'."""
    whole, frac = f"{amount:.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{whole}.{frac}"


def money(amount):
    return f"{CURRENCY} {inr(amount)}"


def rng_for(n):
    return random.Random(MASTER_SEED * 1000 + n)


# ============================================================== DOC MODEL

@dataclass
class Doc:
    doc_id: str
    slug: str
    doc_type: str
    fmt: str            # pdf | jpg | png | tiff
    expected: str       # PASS | FLAG | REVIEW
    flag_code: str
    title: str
    blocks: list        # ("h"|"sec"|"p", text) | ("kv", [(k,v),...]) | ("table", headers, rows) | ("break",)
    scan: dict = field(default_factory=dict)
    seed: int = 0

    @property
    def filename(self):
        return f"{self.doc_id}_{self.slug}.{self.fmt}"


def make_scan(r, lowres=None, noise=None, blur=None):
    """Per-document scan degradation parameters (each doc gets its own randomness)."""
    return {
        "angle": r.uniform(1.0, 4.0) * r.choice([-1, 1]),
        "noise": noise if noise is not None else r.uniform(4.0, 11.0),
        "blur": blur if blur is not None else r.uniform(0.4, 0.9),
        "lowres": lowres,
        "quality": r.randint(72, 90),
    }


def make_doc(n, doc_id, slug, doc_type, fmt, expected, flag, title, blocks,
             lowres=None, noise=None, blur=None):
    seed = MASTER_SEED * 1000 + n
    scan = make_scan(random.Random(seed + 7919), lowres, noise, blur)
    return Doc(doc_id, slug, doc_type, fmt, expected, flag, title,
               blocks + [("p", SPECIMEN_NOTE)], scan, seed)


# ============================================================== RENDERERS

def render_pdf(doc, path, r):
    size = r.choice([9.5, 10, 10.5, 11])
    margin = r.choice([14, 16, 18]) * mm
    width = A4[0] - 2 * margin
    accent = colors.HexColor(r.choice(["#1F3A5F", "#2E5E4E", "#5A2D82", "#333333"]))
    st = {
        "h": ParagraphStyle("h", fontName=PDF_BOLD, fontSize=size + 6,
                            leading=(size + 6) * 1.3, textColor=accent, spaceAfter=8),
        "sec": ParagraphStyle("sec", fontName=PDF_BOLD, fontSize=size + 1.5,
                              leading=(size + 1.5) * 1.4, spaceBefore=8, spaceAfter=3),
        "p": ParagraphStyle("p", fontName=PDF_FONT, fontSize=size,
                            leading=size * 1.45, spaceAfter=4),
        "cell": ParagraphStyle("cell", fontName=PDF_FONT, fontSize=size,
                               leading=size * 1.3, spaceAfter=0),
    }
    story = []
    for block in doc.blocks:
        kind = block[0]
        if kind == "break":
            story.append(PageBreak())
        elif kind in ("h", "sec", "p"):
            story.append(Paragraph(escape(block[1]), st[kind]))
        elif kind == "kv":
            rows = [[k, Paragraph(escape(v), st["cell"])] for k, v in block[1]]
            t = Table(rows, colWidths=[width * 0.36, width * 0.64])
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), PDF_BOLD),
                ("FONTSIZE", (0, 0), (0, -1), size),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story += [t, Spacer(1, 6)]
        elif kind == "table":
            headers, rows = block[1], block[2]
            cols = list(zip(headers, *rows))
            weights = [max(len(str(c)) for c in col) + 2 for col in cols]
            total = sum(weights)
            t = Table([headers] + rows,
                      colWidths=[width * w / total for w in weights],
                      repeatRows=1)
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), PDF_BOLD),
                ("FONTNAME", (0, 1), (-1, -1), PDF_FONT),
                ("FONTSIZE", (0, 0), (-1, -1), size - 0.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EBF2")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#8A8A8A")),
                ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story += [t, Spacer(1, 8)]

    SimpleDocTemplate(path, pagesize=A4, leftMargin=margin, rightMargin=margin,
                      topMargin=margin, bottomMargin=margin, title=doc.title,
                      author="Synthetic test data").build(story)


INK_COLORS = [(25, 25, 25), (20, 35, 80), (45, 45, 45)]


def text_h(draw, font):
    l, t, rgt, b = draw.textbbox((0, 0), "Ag", font=font)
    return b - t


def wrap_px(draw, text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if not cur or draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def render_image(doc, r):
    """Draw the document onto an A4-proportioned Pillow canvas (150 dpi)."""
    W, H = 1240, 1754
    paper = r.randint(236, 250)
    bg = (paper, paper - r.randint(0, 4), paper - r.randint(3, 9))
    ink = r.choice(INK_COLORS)
    grey = (130, 130, 130)
    size = r.randint(26, 31)
    F = {
        "h": pil_font(size + 14, True),
        "sec": pil_font(size + 4, True),
        "p": pil_font(size),
        "b": pil_font(size, True),
    }
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    m = r.randint(95, 130)
    maxw = W - 2 * m
    gap = text_h(d, F["p"]) + 12
    y = m

    for block in doc.blocks:
        kind = block[0]
        if kind == "h":
            d.text((m, y), block[1], font=F["h"], fill=ink)
            y += text_h(d, F["h"]) + 34
        elif kind == "sec":
            y += 12
            d.text((m, y), block[1], font=F["sec"], fill=ink)
            y += text_h(d, F["sec"]) + 14
        elif kind == "p":
            for line in wrap_px(d, block[1], F["p"], maxw):
                d.text((m, y), line, font=F["p"], fill=ink)
                y += gap
            y += 10
        elif kind == "kv":
            col = int(maxw * 0.36)
            for key, value in block[1]:
                d.text((m, y), key, font=F["b"], fill=ink)
                yy = y
                for line in wrap_px(d, value, F["p"], maxw - col):
                    d.text((m + col, yy), line, font=F["p"], fill=ink)
                    yy += gap
                y = max(y + gap, yy)
            y += 10
        elif kind == "table":
            headers, rows = block[1], block[2]
            cols = list(zip(headers, *rows))
            weights = [max(len(str(c)) for c in col) + 2 for col in cols]
            total = sum(weights)
            widths = [int(maxw * w / total) for w in weights]
            step = text_h(d, F["p"]) + 22
            top = y
            all_rows = [headers] + rows
            for ri, row in enumerate(all_rows):
                ry = top + ri * step
                if ri == 0:
                    d.rectangle([m, ry, m + maxw, ry + step], fill=(222, 222, 222))
                x = m
                for ci, cell in enumerate(row):
                    font = F["b"] if ri == 0 else F["p"]
                    text = str(cell)
                    if ci >= 3:   # numeric columns right-aligned
                        tw = d.textlength(text, font=font)
                        tx = x + widths[ci] - 8 - tw
                    else:
                        tx = x + 8
                    d.text((tx, ry + 10), text, font=font, fill=ink)
                    x += widths[ci]
                d.line([m, ry + step, m + maxw, ry + step], fill=grey)
            y = top + len(all_rows) * step + 20

    return img


def degrade(img, scan, rng):
    """Simulate a phone photo / office scanner: rotation, low-res, noise, mild blur."""
    img = img.rotate(scan["angle"], resample=Image.BICUBIC, expand=True,
                     fillcolor=(250, 250, 250))
    arr = np.asarray(img).astype(np.uint8)
    if scan.get("lowres"):
        f = scan["lowres"]
        small = cv2.resize(arr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        arr = cv2.resize(small, (img.width, img.height), interpolation=cv2.INTER_LINEAR)
    noise = rng.normal(0.0, scan["noise"], arr.shape)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    arr = cv2.GaussianBlur(arr, (5, 5), scan["blur"])
    return Image.fromarray(arr)


def save_image(img, path, fmt, scan):
    if fmt == "jpg":
        img.save(path, "JPEG", quality=scan["quality"])
    elif fmt == "png":
        img.save(path, "PNG")
    elif fmt == "tiff":
        img.save(path, "TIFF")


def render(doc, path):
    layout_rng = random.Random(doc.seed)
    if doc.fmt == "pdf":
        render_pdf(doc, path, layout_rng)
    else:
        noise_rng = np.random.default_rng(doc.seed)
        img = render_image(doc, layout_rng)
        img = degrade(img, doc.scan, noise_rng)
        save_image(img, path, doc.fmt, doc.scan)


# ============================================================== INVOICES

VENDORS = {
    "Shree Traders": {"gstin": "27AAFCS4417K1Z3",
                      "address": "Shop 14, Laxmi Market, Pune 411030"},
    "Mehta Steel Works": {"gstin": "24AABFM7721P1ZQ",
                          "address": "Plot 22, GIDC Estate, Vatva, Ahmedabad 382445"},
    "Apex Packaging": {"gstin": "27AACCA9034L1Z8",
                       "address": "Gat No. 88, MIDC Chakan, Pune 410501"},
    "Ravi & Sons General Store": {"gstin": "27AEKPR3186N1ZD",
                                  "address": "Main Road, Kothrud, Pune 411038"},
}

BUYERS = [
    "Kapoor Engineering Works, Pune",
    "Sunrise Hospitality LLP, Mumbai",
    "Jai Hind Retail Pvt. Ltd., Nagpur",
    "Vedanta Clinics Pvt. Ltd., Pune",
    "Konkan Agro Traders, Ratnagiri",
    "Nexgen Logistics Pvt. Ltd., Thane",
    "Mahavir Builders, Nashik",
    "Greenleaf Pharma Distributors, Pune",
]

# 8 disjoint item sets (5 items each) so no two invoices share line items.
# Each entry: (description, HSN code, min rate, max rate)
ITEM_SETS = [
    [("A4 copier paper 75 GSM (ream)", "4802", 240, 320),
     ("Ballpoint pens, box of 50", "9608", 180, 260),
     ("Spiral notebooks, 200 pages", "4820", 45, 80),
     ("Heavy duty stapler", "8305", 350, 520),
     ("Whiteboard markers, pack of 4", "9608", 120, 200)],
    [("MS angle 50x50x6 mm (kg)", "7216", 72, 90),
     ("MS flat bar 25x5 mm (kg)", "7216", 68, 84),
     ("Hex bolts M12 x 60 mm (pkt 100)", "7318", 1100, 1600),
     ("Welding electrodes 3.15 mm (kg)", "8311", 180, 240),
     ("Cutting disc 4.5 inch (pc)", "6804", 45, 70)],
    [("Corrugated carton 9x7x5 in (pc)", "4819", 22, 35),
     ("BOPP tape roll 48 mm", "3919", 60, 95),
     ("Bubble wrap roll 1.2 m", "3921", 1100, 1500),
     ("Stretch film 500 mm, 20 mu", "3920", 900, 1300),
     ("Printed poly mailers (pc)", "3923", 8, 14)],
    [("Tea dust CTC 1 kg", "0902", 320, 410),
     ("Sugar M-30 1 kg", "1701", 44, 52),
     ("Surf detergent 1 kg", "3402", 180, 240),
     ("Phenyl 1 litre", "3808", 90, 130),
     ("Dish wash bar (pc)", "3401", 10, 18)],
    [("USB-C cable 1 m", "8544", 180, 260),
     ("Wireless mouse", "8471", 350, 550),
     ("Laptop bag 15.6 in", "4202", 650, 900),
     ("HDMI cable 2 m", "8544", 220, 320),
     ("Power strip 6-way", "8536", 380, 520)],
    [("Office chair, mesh back", "9401", 4200, 5600),
     ("Study table 120 cm", "9403", 5200, 6800),
     ("Filing cabinet, 2 drawer", "9403", 3100, 4300),
     ("Desk lamp, LED", "9405", 650, 900),
     ("Acrylic sheet 3 mm (sq ft)", "3920", 110, 160)],
    [("MCB 16A single pole", "8536", 210, 290),
     ("Copper wire 2.5 sq mm (m)", "8544", 45, 70),
     ("LED bulb 9 W", "8539", 75, 110),
     ("Switch board 6-module", "8538", 260, 340),
     ("PVC conduit 25 mm (3 m)", "3917", 90, 130)],
    [("Nitrile gloves, box of 100", "4015", 420, 560),
     ("3-ply surgical masks (box 50)", "6307", 180, 260),
     ("Hand sanitiser 500 ml", "3808", 150, 220),
     ("Digital thermometer", "9025", 260, 380),
     ("Cotton roll 500 g", "3005", 120, 180)],
]

INVOICE_CONFIGS = [
    dict(id="inv_01", slug="shree_clean", fmt="pdf", vendor="Shree Traders",
         inv_no="SHT/2026-27/0412", date=dt.date(2026, 10, 14), intra=True,
         po="PO-SHT-7781", buyer=BUYERS[0], set=0, two_page=True,
         expected="PASS", flag=""),
    dict(id="inv_02", slug="mehta_steel_igst", fmt="pdf", vendor="Mehta Steel Works",
         inv_no="MSW/26/1187", date=dt.date(2026, 10, 20), intra=False,
         po="PO-MSW-2210", buyer=BUYERS[1], set=1,
         expected="PASS", flag=""),
    dict(id="inv_03", slug="gstin_mismatch", fmt="jpg", vendor="Shree Traders",
         gstin="27AAFCS4471K1Z3", inv_no="SHT/2026-27/0431",
         date=dt.date(2026, 10, 25), intra=True, po=None,
         buyer=BUYERS[2], set=4,
         expected="FLAG", flag="gstin_mismatch"),
    dict(id="inv_04", slug="ravi_unregistered", fmt="pdf",
         vendor="Ravi & Sons General Store", inv_no="RSGS-0091",
         date=dt.date(2026, 11, 2), intra=True, po=None,
         buyer=BUYERS[3], set=3,
         expected="FLAG", flag="unregistered_vendor"),
    dict(id="inv_05", slug="total_mismatch", fmt="jpg", vendor="Mehta Steel Works",
         inv_no="MSW/26/1199", date=dt.date(2026, 11, 5), intra=False,
         po="PO-MSW-2236", buyer=BUYERS[4], set=6, extra=4500,
         expected="FLAG", flag="total_mismatch"),
    dict(id="inv_06", slug="duplicate_number", fmt="png", vendor="Shree Traders",
         inv_no="SHT/2026-27/0412", date=dt.date(2026, 11, 9), intra=True,
         po="PO-SHT-7790", buyer=BUYERS[5], set=5,
         expected="FLAG", flag="duplicate_invoice"),
    dict(id="inv_07", slug="apex_high_value", fmt="pdf", vendor="Apex Packaging",
         inv_no="APX/2026/0361", date=dt.date(2026, 11, 10), intra=False,
         po="PO-APX-0455", buyer=BUYERS[6], set=2, target_subtotal=66102,
         expected="REVIEW", flag="high_value"),
    dict(id="inv_08", slug="future_dated", fmt="tiff", vendor="Shree Traders",
         inv_no="SHT/2026-27/0455", date=TODAY + dt.timedelta(days=91), intra=True,
         po=None, buyer=BUYERS[7], set=7,
         expected="FLAG", flag="date_logic"),
]


def build_items(set_idx, r, target_subtotal=None):
    rows = []
    for desc, hsn, lo, hi in ITEM_SETS[set_idx]:
        qty = r.randint(1, 6)
        rate = r.randint(lo, hi)
        rows.append([desc, hsn, qty, rate, qty * rate])
    if target_subtotal is not None:
        remainder = target_subtotal - sum(row[4] for row in rows)
        rows.append(["Heat sealing machine, industrial (pc)", "8479", 1, remainder, remainder])
    return rows


def build_invoice(n, cfg):
    r = rng_for(n)
    vendor = VENDORS[cfg["vendor"]]
    gstin_printed = cfg.get("gstin") or vendor["gstin"]
    items = build_items(cfg["set"], r, cfg.get("target_subtotal"))
    sub = sum(row[4] for row in items)
    if cfg["intra"]:
        half = round(sub * 0.09)
        taxes = [("CGST @ 9%", half), ("SGST @ 9%", half)]
    else:
        taxes = [("IGST @ 18%", round(sub * 0.18))]
    grand = sub + sum(t for _, t in taxes) + cfg.get("extra", 0)

    inv_date = cfg["date"].strftime(r.choice(["%d-%m-%Y", "%d/%m/%Y", "%d %b %Y"]))
    inv_label = r.choice(["Invoice No.", "Invoice Number", "Inv #"])
    date_label = r.choice(["Invoice Date", "Date", "Dated"])
    gst_label = r.choice(["GSTIN", "GST No.", "GSTIN / UIN"])

    blocks = [
        ("h", "TAX INVOICE"),
        ("kv", [(inv_label, cfg["inv_no"]),
                (date_label, inv_date),
                ("PO Reference", cfg.get("po") or "Not quoted")]),
        ("sec", "Supplier"),
        ("kv", [("Name", cfg["vendor"]),
                ("Address", vendor["address"]),
                (gst_label, gstin_printed)]),
        ("sec", "Bill To"),
        ("kv", [("Buyer", cfg["buyer"])]),
        ("sec", "Line Items"),
        ("table", ["Sl", "Description", "HSN", "Qty", "Rate", "Amount"],
         [[str(i + 1), row[0], row[1], str(row[2]), inr(row[3]), inr(row[4])]
          for i, row in enumerate(items)]),
    ]
    if cfg.get("two_page"):
        blocks.append(("break",))
    totals = ([("Taxable value", money(sub))]
              + [(lbl, money(v)) for lbl, v in taxes]
              + [("Grand Total", money(grand))])
    blocks += [
        ("sec", "Totals"),
        ("kv", totals),
        ("p", "Payment due within 30 days of invoice date. Subject to Pune jurisdiction."),
    ]
    return make_doc(n, cfg["id"], cfg["slug"], "invoice", cfg["fmt"],
                    cfg["expected"], cfg["flag"],
                    f"Tax Invoice {cfg['inv_no']}", blocks)


# ============================================================== KYC

def pan_blocks(r, name, father, dob, pan):
    return [
        ("h", "INCOME TAX DEPARTMENT"),
        ("p", "Government of India - Permanent Account Number Card"),
        ("kv", [("Name", name),
                ("Father's Name", father),
                (r.choice(["Date of Birth", "DOB"]), dob),
                ("Permanent Account Number", pan)]),
    ]


def dl_blocks(r, name, dl_no, dob, valid_till):
    return [
        ("h", "DRIVING LICENCE"),
        ("p", "Union of India - Form LL/DL"),
        ("kv", [("DL No.", dl_no),
                ("Name", name),
                ("Date of Birth", dob),
                ("Valid Till (NT)", valid_till),
                ("Class of Vehicle", "MCWG, LMV")]),
    ]


def aadhaar_blocks(name, dob, aadhaar, address):
    return [
        ("h", "Government of India"),
        ("p", "Aadhaar - Unique Identification Authority of India"),
        ("kv", [("Name", name),
                ("Date of Birth", dob),
                ("Gender", "Female"),
                ("Aadhaar Number", aadhaar)]),
        ("kv", [("Address", address)]),
    ]


def passport_blocks(r, surname, given, dob, pno, issue, expiry):
    return [
        ("h", "REPUBLIC OF INDIA"),
        ("p", "Passport"),
        ("kv", [("Passport No.", pno),
                (r.choice(["Surname", "Last Name"]), surname),
                ("Given Name(s)", given),
                ("Date of Birth", dob),
                ("Place of Birth", "Kochi"),
                ("Date of Issue", issue),
                ("Date of Expiry", expiry)]),
    ]


def build_kyc(start_n):
    n = start_n
    docs = []

    # PAN format: 5 letters, 4 digits, 1 letter.  4th char must be in PCHFATBLJG.
    # ABCPE1234F → 4th char = P (Person) ✅  5th char = E (first letter of surname) ✅
    docs.append(make_doc(n, "kyc_01", "pan_valid", "kyc_pan", "pdf", "PASS", "",
                         "PAN Card", pan_blocks(rng_for(n), "Priya Sharma",
                                                "Ramesh Sharma", "18/04/1992",
                                                "ABCPS1234F")))
    n += 1
    docs.append(make_doc(n, "kyc_02", "pan_bad_format", "kyc_pan", "pdf", "FLAG",
                         "bad_pan_format", "PAN Card",
                         pan_blocks(rng_for(n), "Rahul Verma", "Suresh Verma",
                                    "09/11/1988", "1234ABCDE5")))
    n += 1
    docs.append(make_doc(n, "kyc_03", "dl_valid", "kyc_driving_licence", "jpg", "PASS", "",
                         "Driving Licence", dl_blocks(rng_for(n), "Anjali Desai",
                                                      "MH12 20190045678", "23/02/1990",
                                                      "14/07/2029")))
    n += 1
    docs.append(make_doc(n, "kyc_04", "dl_expired", "kyc_driving_licence", "jpg", "FLAG",
                         "expired_id", "Driving Licence",
                         dl_blocks(rng_for(n), "Sanjay Patil", "MH14 20150098321",
                                   "02/06/1984", "15/11/2022")))
    n += 1
    docs.append(make_doc(n, "kyc_05", "aadhaar_valid", "kyc_aadhaar", "png", "PASS", "",
                         "Aadhaar Card", aadhaar_blocks("Meera Iyer", "09/08/2001",
                                                        "5482 1376 9042",
                                                        "12, Lake View Road, Chennai 600034")))
    n += 1
    docs.append(make_doc(n, "kyc_06", "passport_underage", "kyc_passport", "tiff", "FLAG",
                         "underage", "Passport",
                         passport_blocks(rng_for(n), "Nair", "Arjun", "27/02/2013",
                                         "Z4812376", "10/05/2023", "09/05/2033")))
    return docs


# ============================================================== CONTRACTS

PAY_VARIANTS = [
    "Undisputed invoices shall be settled within {n} days from the date of receipt.",
    "The Client shall remit payment for each invoice in full within {n} calendar days.",
    "Payment falls due {n} days after the invoice date, as set out in the statement of work.",
]
TERM_VARIANTS = [
    "Either party may terminate this agreement by giving {n} days written notice to the other party.",
    "This agreement may be ended by either party upon {n} days' prior written notice.",
    "Termination requires {n} days of advance written intimation from the terminating party.",
]
CAP_VARIANTS = [
    "The aggregate liability of the Service Provider shall not exceed {cap} for any claim arising out of this agreement.",
    "Total liability under this agreement is capped at {cap} in aggregate.",
]
UNCAPPED_TEXT = ("The Service Provider shall bear unlimited liability for all losses, "
                 "damages and costs arising from this agreement, without any monetary cap.")
SCOPE_VARIANTS = [
    "The Service Provider shall deliver {svc} to the Client as detailed in Annexure A.",
    "Services covered: {svc}, performed as per the statement of work agreed between the parties.",
]
LAW_VARIANTS = [
    "This agreement is governed by the laws of India, and courts at {city} shall have exclusive jurisdiction.",
    "Indian law shall govern this agreement; disputes are subject to the courts of {city}.",
]


def contract_blocks(r, title, date_str, party_a, party_b, scope, pay, term, liab, city):
    return [
        ("h", title),
        ("p", f'This agreement is made on {date_str} between {party_a} (the "Client") '
              f'and {party_b} (the "Service Provider").'),
        ("sec", "1. Scope"), ("p", scope),
        ("sec", "2. Payment Terms"), ("p", pay),
        ("sec", "3. Termination"), ("p", term),
        ("sec", "4. Liability"), ("p", liab),
        ("sec", "5. Governing Law"), ("p", r.choice(LAW_VARIANTS).format(city=city)),
        ("kv", [("For the Client", party_a), ("For the Service Provider", party_b)]),
    ]


def build_contracts(start_n):
    docs = []
    n = start_n

    def svc_contract(n, doc_id, slug, fmt, a, b, svc, pay_days, term_days, cap, city,
                     date_str, expected, flag, lowres=None):
        r = rng_for(n)
        scope = r.choice(SCOPE_VARIANTS).format(svc=svc)
        pay = r.choice(PAY_VARIANTS).format(n=pay_days)
        term = r.choice(TERM_VARIANTS).format(n=term_days)
        liab = (r.choice(CAP_VARIANTS).format(cap=money(cap)) if cap else UNCAPPED_TEXT)
        blocks = contract_blocks(r, "SERVICE AGREEMENT", date_str, a, b,
                                 scope, pay, term, liab, city)
        return make_doc(n, doc_id, slug, "contract_service", fmt, expected, flag,
                        f"Service Agreement - {a} / {b}", blocks)

    docs.append(svc_contract(n, "contract_01", "clean", "pdf",
                             "Bharat Logistics Pvt. Ltd.", "Nirmal Textiles LLP",
                             "warehouse and dispatch support", 30, 30, 500000, "Pune",
                             "01 September 2026", "PASS", ""))
    n += 1
    docs.append(svc_contract(n, "contract_02", "payment_90", "pdf",
                             "Vatsal Fabrics Pvt. Ltd.", "Konark Electronics Pvt. Ltd.",
                             "contract manufacturing of printed circuit assemblies",
                             90, 30, 500000, "Mumbai", "14 September 2026",
                             "FLAG", "payment_terms"))
    n += 1
    docs.append(svc_contract(n, "contract_03", "short_notice", "pdf",
                             "Heritage Dairy Co-op Ltd.", "Saraswati Print House",
                             "offset printing and finishing", 30, 7, 500000, "Ahmedabad",
                             "22 September 2026", "FLAG", "short_notice"))
    n += 1
    docs.append(svc_contract(n, "contract_04", "uncapped_liability", "jpg",
                             "Pioneer Cold Chain LLP", "Aditya Foods Pvt. Ltd.",
                             "temperature-controlled transport", 30, 30, None, "Bengaluru",
                             "03 October 2026", "FLAG", "no_liability_cap"))
    n += 1

    # NDA: payment clause is not applicable; all terms within limits
    r = rng_for(n)
    nda_scope = ("The Receiving Party shall keep all Confidential Information strictly "
                 "confidential and use it only to evaluate the proposed business relationship.")
    nda_pay = "No fees are payable under this Agreement; payment terms are not applicable."
    nda_term = "Either party may terminate this agreement by giving 30 days written notice to the other party."
    nda_liab = r.choice(CAP_VARIANTS).format(cap=money(500000))
    blocks = contract_blocks(r, "NON-DISCLOSURE AGREEMENT", "08 October 2026",
                             "Lotus Software Labs", "Chandra Consultants",
                             nda_scope, nda_pay, nda_term, nda_liab, "Pune")
    docs.append(make_doc(n, "contract_05", "nda_standard", "contract_nda", "jpg",
                         "PASS", "", "Non-Disclosure Agreement", blocks))
    n += 1

    docs.append(svc_contract(n, "contract_06", "multi_issue", "png",
                             "Tirupati Infra Pvt. Ltd.", "Kaveri Solar Systems",
                             "EPC support for rooftop solar installations", 60, 10,
                             500000, "Chennai", "19 October 2026",
                             "FLAG", "payment_terms;short_notice"))
    return docs


# ============================================================== INSURANCE CLAIMS

POLICY_NO = "POL-77120"
POLICY_LIMIT = 150000


def claim_blocks(r, claim_no, claim_type, name, phone, loss_date, place,
                 amount, policy, desc, provider):
    return [
        ("h", "INSURANCE CLAIM FORM"),
        ("kv", [(r.choice(["Claim No.", "Claim Number"]), claim_no),
                ("Date of Claim", "18/11/2026" if False else "16/11/2026"),
                (r.choice(["Policy No.", "Policy No"]), policy),
                ("Policy Limit", money(POLICY_LIMIT)),
                ("Claim Type", claim_type)]),
        ("sec", "Claimant Details"),
        ("kv", [("Claimant Name", name),
                ("Mobile", phone),
                ("Date of Loss", loss_date),
                ("Place of Loss", place)]),
        ("sec", "Claim Details"),
        ("kv", [("Claimed Amount", money(amount)),
                ("Description", desc),
                ("Garage / Hospital", provider)]),
        ("p", "I declare that the information above is true and complete to the best of my knowledge."),
        ("kv", [("Signature of Claimant", name)]),
    ]


def build_claims(start_n):
    docs = []
    n = start_n
    docs.append(make_doc(n, "claim_01", "motor_within_limit", "insurance_claim", "pdf",
                         "PASS", "", "Insurance Claim - Motor",
                         claim_blocks(rng_for(n), "CLM/2026/10231", "Motor - Own Damage",
                                      "Vikram Rao", "+91 98220 11473", "02-10-2026",
                                      "Pune", 65000, POLICY_NO,
                                      "Rear bumper and tail light damage in a parking collision",
                                      "Sunrise Motors Service Centre, Pune")))
    n += 1
    docs.append(make_doc(n, "claim_02", "motor_over_limit", "insurance_claim", "pdf",
                         "FLAG", "over_limit", "Insurance Claim - Motor",
                         claim_blocks(rng_for(n), "CLM/2026/10244", "Motor - Own Damage",
                                      "Sunita Kulkarni", "+91 77210 63908", "07-10-2026",
                                      "Nashik Highway", 190000, POLICY_NO,
                                      "Front-end damage from road accident, repair estimate attached",
                                      "Om Auto Works, Nashik")))
    n += 1
    docs.append(make_doc(n, "claim_03", "health_valid", "insurance_claim", "jpg",
                         "PASS", "", "Insurance Claim - Health",
                         claim_blocks(rng_for(n), "CLM/2026/10257", "Health - Hospitalisation",
                                      "Farhan Shaikh", "+91 90110 27745", "29-09-2026",
                                      "Pune", 45000, POLICY_NO,
                                      "Appendectomy, 3-day inpatient stay",
                                      "Ruby Care Hospital, Pune")))
    n += 1
    docs.append(make_doc(n, "claim_04", "policy_not_found", "insurance_claim", "jpg",
                         "FLAG", "policy_not_found", "Insurance Claim - Motor",
                         claim_blocks(rng_for(n), "CLM/2026/10263", "Motor - Third Party",
                                      "Deepa Menon", "+91 99450 08812", "11-10-2026",
                                      "Kochi", 58000, "POL-99999",
                                      "Two-wheeler collision at a signal junction",
                                      "Krishna Motors, Kochi")))
    n += 1
    docs.append(make_doc(n, "claim_05", "blurry_phone_scan", "insurance_claim", "tiff",
                         "REVIEW", "low_ocr_confidence", "Insurance Claim - Motor",
                         claim_blocks(rng_for(n), "CLM/2026/10270", "Motor - Own Damage",
                                      "Govind Pillai", "+91 98903 44120", "13-10-2026",
                                      "Thane", 42000, POLICY_NO,
                                      "Side mirror and door panel dent",
                                      "Fast Track Garage, Thane"),
                         lowres=0.3, noise=16.0, blur=1.1))
    return docs


# ============================================================== ASSEMBLY

def build_all_docs():
    docs = []
    seq = itertools.count(1)
    for cfg in INVOICE_CONFIGS:
        docs.append(build_invoice(next(seq), cfg))
    docs += build_kyc(next(seq))
    docs += build_contracts(next(seq))
    docs += build_claims(next(seq))
    return docs


def write_manifest(docs, out_dir):
    path = os.path.join(out_dir, MANIFEST_NAME)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "doc_type", "file_format", "expected_outcome", "flag_code"])
        for d in docs:
            w.writerow([d.filename, d.doc_type, d.fmt, d.expected, d.flag_code])
    return path


def zip_folder(folder, zip_path):
    base = os.path.dirname(folder)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            for name in sorted(files):
                full = os.path.join(root, name)
                zf.write(full, arcname=os.path.relpath(full, base))


def print_table(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt(row):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"

    print(sep)
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))
    print(sep)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, OUT_DIR_NAME)
    zip_path = os.path.join(base_dir, ZIP_NAME)

    setup_fonts()

    for stale in (out_dir, zip_path):
        if os.path.isdir(stale):
            shutil.rmtree(stale)
        elif os.path.exists(stale):
            os.remove(stale)
    os.makedirs(out_dir)

    docs = build_all_docs()
    assert len(docs) == 25, f"expected 25 docs, got {len(docs)}"

    print(f"\nRendering {len(docs)} documents into ./{OUT_DIR_NAME}/ ...")
    for d in docs:
        render(d, os.path.join(out_dir, d.filename))
        print(f"  [ok] {d.filename}")

    write_manifest(docs, out_dir)
    zip_folder(out_dir, zip_path)
    print(f"\nZipped -> {zip_path}")

    fmt_counts = {}
    for d in docs:
        fmt_counts[d.fmt] = fmt_counts.get(d.fmt, 0) + 1
    print("Format mix:", ", ".join(f"{k.upper()}={v}" for k, v in sorted(fmt_counts.items())))

    print("\nMANIFEST\n")
    print_table(
        ["#", "filename", "doc_type", "format", "expected", "flag_code"],
        [[i + 1, d.filename, d.doc_type, d.fmt.upper(), d.expected, d.flag_code or "-"]
         for i, d in enumerate(docs)],
    )


if __name__ == "__main__":
    main()