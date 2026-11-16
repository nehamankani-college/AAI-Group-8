"""
make_demo_dataset.py — Rich 20-document demo dataset for DocAgent (PS-15)
=========================================================================
Generates synthetic but realistic Indian business documents covering
every document type, every failure mode, and every agent path.

Documents generated:
  INVOICES (8)
    01_inv_shree_clean          — perfect invoice → auto-approve
    02_inv_mehta_valid          — valid 2-page invoice with CGST+SGST → auto-approve
    03_inv_apex_highvalue       — large invoice > ₹50,000 → review (high value)
    04_inv_unregistered_vendor  — unknown vendor → flag unregistered_vendor
    05_inv_math_error           — line items don't add up → flag total_mismatch
    06_inv_duplicate            — same invoice number as #01 → flag duplicate
    07_inv_no_po                — no PO number → flag missing_po
    08_inv_future_date          — invoice dated next year → flag date_logic

  KYC DOCUMENTS (5)
    09_kyc_pan_valid            — clean PAN card → auto-approve
    10_kyc_pan_bad_format       — PAN number has wrong format → flag
    11_kyc_dl_expired           — driving licence expired 2 yrs ago → flag expired_id
    12_kyc_aadhaar_valid        — Aadhaar card → auto-approve
    13_kyc_passport_underage    — passport with DOB making holder 16 → flag underage

  CONTRACTS (4)
    14_contract_standard        — good contract → auto-approve
    15_contract_long_payment    — payment terms 90 days (> max 45) → flag
    16_contract_no_liability    — missing liability cap clause → flag
    17_contract_short_notice    — only 7 days termination notice (< 30) → flag

  CLAIMS (3)
    18_claim_valid              — valid motor claim within policy limit → auto-approve
    19_claim_over_limit         — claim amount exceeds policy limit → flag
    20_claim_blurry_scan        — degraded photo scan → low OCR confidence → review

Usage:
    python3 make_demo_dataset.py
    # writes 20 PDFs + 2 JPGs to  demo_dataset/
"""

import os, random, math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "demo_dataset")
os.makedirs(OUT, exist_ok=True)

# ── Fonts ───────────────────────────────────────────────────────────────────
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf", "arial.ttf",
]
BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf", "arialbd.ttf",
]

def _pick(cands):
    for c in cands:
        try:
            ImageFont.truetype(c, 20)
            return c
        except OSError:
            continue
    raise SystemExit("No TrueType font found. Edit FONT_CANDIDATES in make_demo_dataset.py")

FONT = _pick(FONT_CANDIDATES)
BOLD = _pick(BOLD_CANDIDATES)
W, H = 1654, 2339   # A4 @ 200 dpi

def f(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)

def page(bg=255):
    img = Image.new("L", (W, H), bg)
    return img, ImageDraw.Draw(img)

def scanify(img, angle=0.0, noise=5, seed=1, jpeg_q=None):
    rng = np.random.default_rng(seed)
    a = np.array(img)
    if abs(angle) > 0.05:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        a = cv2.warpAffine(a, M, (W, H), borderValue=235)
    a = a.astype(np.float32) + rng.normal(0, noise, a.shape)
    a = cv2.GaussianBlur(a, (3, 3), 0.5)
    img2 = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if jpeg_q:                      # simulate phone-photo JPEG compression
        from io import BytesIO
        buf = BytesIO()
        img2.convert("RGB").save(buf, "JPEG", quality=jpeg_q)
        buf.seek(0)
        img2 = Image.open(buf).convert("L")
    return img2

def hr(d, y, margin=120):
    d.line((margin, y, W - margin, y), fill=0, width=2)

def inr(n):
    return f"Rs.{n:,.2f}"

def save_pdf(name, pages, note=None):
    path = os.path.join(OUT, name + ".pdf")
    pages[0].convert("RGB").save(path, save_all=True,
        append_images=[p.convert("RGB") for p in pages[1:]], resolution=200)
    print(f"  ✓ {name}.pdf" + (f"  [{note}]" if note else ""))

def save_jpg(name, img, note=None):
    path = os.path.join(OUT, name + ".jpg")
    img.save(path, quality=72)
    print(f"  ✓ {name}.jpg" + (f"  [{note}]" if note else ""))


# ═══════════════════════════════════════════════════════════════════════════
# INVOICE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════
def _invoice_base(d, inv_no, date_str, due_str, po, vendor, gstin, vendor_addr, bill_to, page_label="1 of 1"):
    d.text((120, 100), "TAX INVOICE", font=f(68, True), fill=0)
    d.line((120, 185, W-120, 185), fill=0, width=4)
    d.text((120, 210), vendor, font=f(44, True), fill=0)
    d.text((120, 268), vendor_addr, font=f(28), fill=0)
    d.text((120, 310), f"GSTIN: {gstin}", font=f(28), fill=0)
    d.text((960, 210), f"Invoice No: {inv_no}", font=f(32), fill=0)
    d.text((960, 258), f"Invoice Date: {date_str}", font=f(32), fill=0)
    d.text((960, 306), f"Due Date: {due_str}", font=f(32), fill=0)
    if po:
        d.text((960, 354), f"PO Number: {po}", font=f(32), fill=0)
    d.text((120, 400), f"Bill To: {bill_to}", font=f(30), fill=0)
    d.line((120, 450, W-120, 450), fill=0, width=3)
    return 470  # next y

def _invoice_table(d, y, items):
    """items: list of (desc, qty, rate) — amount computed automatically."""
    headers = [("Description", 740), ("Qty", 130), ("Rate", 240), ("Amount", 280)]
    x = 120
    for h, w in headers:
        d.text((x, y), h, font=f(28, True), fill=0)
        x += w
    y += 42
    hr(d, y)
    y += 14
    total = 0
    for desc, qty, rate in items:
        amt = qty * rate
        total += amt
        vals = [desc, str(qty), inr(rate), inr(amt)]
        x = 120
        for v, (_, w) in zip(vals, headers):
            d.text((x, y), v, font=f(27), fill=0)
            x += w
        y += 46
    return y, total

def _invoice_totals(d, y, subtotal, tax_rows, total_override=None):
    hr(d, y + 10)
    y += 30
    d.text((1000, y), f"Subtotal:   {inr(subtotal)}", font=f(30), fill=0)
    y += 52
    tax_sum = 0
    for label, amt in tax_rows:
        d.text((1000, y), f"{label}:   {inr(amt)}", font=f(30), fill=0)
        y += 52
        tax_sum += amt
    total = total_override if total_override is not None else round(subtotal + tax_sum, 2)
    hr(d, y + 4)
    y += 24
    d.text((1000, y), f"Total:  {inr(total)}", font=f(36, True), fill=0)
    return y + 60

def _invoice_footer(d, y):
    d.text((120, y + 40), "Authorised Signatory: ____________________", font=f(28), fill=0)
    d.text((120, y + 100), "Terms: Goods once sold will not be accepted back.", font=f(24), fill=0)
    d.text((120, y + 150), "Bank: HDFC Bank  |  A/C: 00110234567890  |  IFSC: HDFC0001122", font=f(24), fill=0)


# ── 01 Clean Invoice → auto-approve ─────────────────────────────────────────
def inv_01_clean():
    p, d = page()
    y = _invoice_base(d, "INV-2001", "03/11/2026", "03/12/2026", "PO-2291",
                      "Shree Traders", "27ABCDE1234F1Z5",
                      "14, MG Road, Pune, Maharashtra 411001",
                      "Kalyani Engineering Pvt Ltd")
    items = [("Steel Brackets 40mm", 100, 120), ("Hex Bolts M10 (Box)", 20, 450),
             ("Mounting Plates 6mm", 50, 250), ("Packing & Freight", 1, 1800)]
    y, sub = _invoice_table(d, y, items)
    tax = [("CGST 9%", round(sub * 0.09, 2)), ("SGST 9%", round(sub * 0.09, 2))]
    y = _invoice_totals(d, y, sub, tax)
    _invoice_footer(d, y)
    save_pdf("01_inv_shree_clean", [scanify(p, 0.4, seed=11)], "PASS → auto-approve")

# ── 02 Valid 2-page Invoice with CGST+SGST ───────────────────────────────────
def inv_02_mehta():
    p1, d1 = page()
    y = _invoice_base(d1, "INV-2002", "10/11/2026", "10/12/2026", "PO-2305",
                      "Mehta Steel Works", "24AAACM5678K1ZP",
                      "78, Ring Road, Surat, Gujarat 395003",
                      "Kalyani Engineering Pvt Ltd")
    items = [("HR Sheet 2mm x 4ft x 8ft", 30, 2800), ("CR Sheet 1mm x 4ft x 8ft", 25, 3200),
             ("MS Angle 50x50x6", 40, 1100), ("MS Flat Bar 50x6", 60, 480)]
    y, sub1 = _invoice_table(d1, y, items)
    d1.text((1000, y + 20), f"Subtotal pg 1: {inr(sub1)}", font=f(28), fill=0)
    d1.text((W - 400, H - 160), "Page 1 of 2", font=f(26), fill=0)

    p2, d2 = page()
    d2.text((120, 80), "Invoice No: INV-2002 (Page 2 of 2)", font=f(32, True), fill=0)
    items2 = [("MS Round Bar 25mm dia", 20, 950), ("Welding Electrodes 7018", 50, 220),
              ("Transport & Packing", 1, 4500)]
    y2, sub2 = _invoice_table(d2, 160, items2)
    sub = sub1 + sub2
    cgst = round(sub * 0.09, 2)
    sgst = round(sub * 0.09, 2)
    y2 = _invoice_totals(d2, y2, sub, [("CGST 9%", cgst), ("SGST 9%", sgst)])
    _invoice_footer(d2, y2)
    save_pdf("02_inv_mehta_valid", [scanify(p1, 1.0, seed=12), scanify(p2, -0.6, seed=13)],
             "PASS → auto-approve (2 pages)")

# ── 03 High-value Invoice → review ──────────────────────────────────────────
def inv_03_highvalue():
    p, d = page()
    y = _invoice_base(d, "INV-2003", "12/11/2026", "12/12/2026", "PO-3310",
                      "Apex Packaging LLP", "29AAFFA9988B1Z2",
                      "45, Industrial Area, Bengaluru, Karnataka 560099",
                      "Kalyani Engineering Pvt Ltd")
    items = [("HDPE Bags 50kg (1000 pcs)", 10, 4200), ("PP Woven Sacks (500 pcs)", 5, 3800),
             ("Stretch Film Rolls (100 kg)", 20, 1200), ("Bubble Wrap 50m x 1.2m", 30, 650),
             ("Packing Machinery Rental", 1, 18000)]
    y, sub = _invoice_table(d, y, items)
    tax = [("IGST 18%", round(sub * 0.18, 2))]
    y = _invoice_totals(d, y, sub, tax)
    _invoice_footer(d, y)
    save_pdf("03_inv_apex_highvalue", [scanify(p, -0.8, seed=14)],
             "REVIEW → invoice_review_threshold exceeded")

# ── 04 Unregistered Vendor ───────────────────────────────────────────────────
def inv_04_unregistered():
    p, d = page()
    y = _invoice_base(d, "INV-2004", "14/11/2026", "14/12/2026", "PO-9999",
                      "Ravi & Sons Traders", "36BBBCS4567G1Z8",   # not in vendor master
                      "12, Gandhi Chowk, Hyderabad, Telangana 500001",
                      "Kalyani Engineering Pvt Ltd")
    items = [("Copper Wire 2.5mm (100m)", 50, 380), ("Electrical Tape (10 pcs)", 100, 45),
             ("MCB 32A (10 pcs)", 10, 420)]
    y, sub = _invoice_table(d, y, items)
    tax = [("GST 18%", round(sub * 0.18, 2))]
    y = _invoice_totals(d, y, sub, tax)
    _invoice_footer(d, y)
    save_pdf("04_inv_unregistered_vendor", [scanify(p, 1.5, seed=15)],
             "REJECT → unregistered_vendor")

# ── 05 Math Error (line items ≠ total) ──────────────────────────────────────
def inv_05_math_error():
    p, d = page()
    y = _invoice_base(d, "INV-2005", "15/11/2026", "15/12/2026", "PO-2291",
                      "Shree Traders", "27ABCDE1234F1Z5",
                      "14, MG Road, Pune, Maharashtra 411001",
                      "Kalyani Engineering Pvt Ltd")
    items = [("Threaded Rod M12 1m", 200, 85), ("Lock Nuts M12", 400, 12),
             ("Spring Washers M12", 400, 8), ("Delivery Charges", 1, 1500)]
    y, sub = _invoice_table(d, y, items)  # real sub = 200*85+400*12+400*8+1500 = 24300
    # Deliberately wrong total (inflated by 3000)
    tax = [("GST 18%", round(sub * 0.18, 2))]
    wrong_total = round(sub + sum(t for _, t in tax) + 3000, 2)
    y = _invoice_totals(d, y, sub, tax, total_override=wrong_total)
    _invoice_footer(d, y)
    save_pdf("05_inv_math_error", [scanify(p, 0.3, seed=16)],
             "REVIEW → total_mismatch (inflated by ₹3,000)")

# ── 06 Duplicate Invoice ─────────────────────────────────────────────────────
def inv_06_duplicate():
    # Same invoice number as 01 but slightly different amounts (OCR variant)
    p, d = page()
    y = _invoice_base(d, "INV-2001", "03/11/2026", "03/12/2026", "PO-2291",
                      "Shree Traders", "27ABCDE1234F1Z5",
                      "14, MG Road, Pune, Maharashtra 411001",
                      "Kalyani Engineering Pvt Ltd")
    items = [("Steel Brackets 40mm", 100, 120), ("Hex Bolts M10 (Box)", 20, 450),
             ("Mounting Plates 6mm", 50, 250), ("Packing & Freight", 1, 1800)]
    y, sub = _invoice_table(d, y, items)
    tax = [("CGST 9%", round(sub * 0.09, 2)), ("SGST 9%", round(sub * 0.09, 2))]
    y = _invoice_totals(d, y, sub, tax)
    _invoice_footer(d, y)
    save_pdf("06_inv_duplicate", [scanify(p, -1.1, noise=7, seed=17)],
             "REVIEW → duplicate_invoice (same INV-2001)")

# ── 07 Missing PO ────────────────────────────────────────────────────────────
def inv_07_no_po():
    p, d = page()
    y = _invoice_base(d, "INV-2007", "16/11/2026", "16/12/2026", None,  # no PO
                      "Mehta Steel Works", "24AAACM5678K1ZP",
                      "78, Ring Road, Surat, Gujarat 395003",
                      "Kalyani Engineering Pvt Ltd")
    items = [("Stainless Steel Pipe 1 inch (6m)", 15, 1800), ("SS Elbow 90 deg 1 inch", 30, 220),
             ("Teflon Tape 12mm", 50, 35)]
    y, sub = _invoice_table(d, y, items)
    tax = [("GST 18%", round(sub * 0.18, 2))]
    y = _invoice_totals(d, y, sub, tax)
    _invoice_footer(d, y)
    save_pdf("07_inv_no_po", [scanify(p, 0.9, seed=18)], "REVIEW → missing_po")

# ── 08 Future-dated Invoice ───────────────────────────────────────────────────
def inv_08_future_date():
    p, d = page()
    y = _invoice_base(d, "INV-2008", "15/06/2027", "15/07/2027", "PO-2291",
                      "Shree Traders", "27ABCDE1234F1Z5",
                      "14, MG Road, Pune, Maharashtra 411001",
                      "Kalyani Engineering Pvt Ltd")
    items = [("Conveyor Belt 500mm x 10m", 2, 12500), ("Belt Fasteners (box of 50)", 4, 1800)]
    y, sub = _invoice_table(d, y, items)
    tax = [("GST 18%", round(sub * 0.18, 2))]
    y = _invoice_totals(d, y, sub, tax)
    _invoice_footer(d, y)
    save_pdf("08_inv_future_date", [scanify(p, -0.5, seed=19)],
             "REVIEW → date_logic (future invoice date)")


# ═══════════════════════════════════════════════════════════════════════════
# KYC BUILDERS
# ═══════════════════════════════════════════════════════════════════════════
def _card(d, title, subtitle=None):
    d.rectangle((140, 180, W - 140, 1200), outline=0, width=6)
    d.text((200, 220), title, font=f(50, True), fill=0)
    if subtitle:
        d.text((200, 290), subtitle, font=f(32), fill=0)
    return 400

def _kv(d, y, label, value, label_size=26, val_size=40):
    d.text((200, y), label, font=f(label_size), fill=0)
    d.text((200, y + label_size + 8), value, font=f(val_size, True), fill=0)
    return y + label_size + val_size + 28

# ── 09 Valid PAN Card ────────────────────────────────────────────────────────
def kyc_09_pan_valid():
    p, d = page()
    y = _card(d, "INCOME TAX DEPARTMENT", "GOVT. OF INDIA")
    d.text((200, y), "Permanent Account Number Card", font=f(32), fill=0)
    y += 70
    y = _kv(d, y, "", "ABCPK1234F")            # valid PAN format
    y = _kv(d, y, "Name", "SUPRIYA KRISHNAN NAYAK")
    y = _kv(d, y, "Father's Name", "KRISHNAN NAYAK")
    y = _kv(d, y, "Date of Birth", "14/03/1998")
    d.text((200, y + 40), "Signature: ____________", font=f(30), fill=0)
    save_pdf("09_kyc_pan_valid", [scanify(p, -1.5, noise=7, seed=21)], "PASS → auto-approve")

# ── 10 PAN with Bad Format ────────────────────────────────────────────────────
def kyc_10_pan_bad():
    p, d = page()
    y = _card(d, "INCOME TAX DEPARTMENT", "GOVT. OF INDIA")
    d.text((200, y), "Permanent Account Number Card", font=f(32), fill=0)
    y += 70
    y = _kv(d, y, "", "1234ABCDE5")            # digits first — invalid PAN format
    y = _kv(d, y, "Name", "VIKRAM DESAI")
    y = _kv(d, y, "Father's Name", "MOHAN DESAI")
    y = _kv(d, y, "Date of Birth", "22/08/1990")
    save_pdf("10_kyc_pan_bad_format", [scanify(p, 1.0, noise=8, seed=22)],
             "REVIEW → bad PAN format")

# ── 11 Expired Driving Licence ────────────────────────────────────────────────
def kyc_11_dl_expired():
    p, d = page()
    y = _card(d, "Driving Licence", "Republic of India — Maharashtra")
    y = _kv(d, y, "DL No", "MH12 20150012345")
    y = _kv(d, y, "Name", "RAJESH DILIP PATIL")
    y = _kv(d, y, "Date of Birth", "02/07/1985")
    y = _kv(d, y, "Address", "22 Shivaji Nagar, Pune 411005")
    y = _kv(d, y, "Valid Till", "01/05/2023")     # expired
    y = _kv(d, y, "Vehicle Class", "LMV / MCWG")
    save_pdf("11_kyc_dl_expired", [scanify(p, 2.0, noise=9, seed=23)],
             "REJECT → expired_id (expired May 2023)")

# ── 12 Valid Aadhaar ──────────────────────────────────────────────────────────
def kyc_12_aadhaar_valid():
    p, d = page()
    d.rectangle((140, 180, W - 140, 1200), outline=0, width=6)
    d.text((200, 220), "भारत सरकार / Government of India", font=f(44, True), fill=0)
    d.text((200, 290), "Unique Identification Authority of India — UIDAI", font=f(28), fill=0)
    y = 400
    y = _kv(d, y, "Aadhaar No", "2345 6789 0123")
    y = _kv(d, y, "Name", "PRIYA SHARMA")
    y = _kv(d, y, "Date of Birth", "11/05/2000")
    y = _kv(d, y, "Gender", "Female")
    y = _kv(d, y, "Address", "B-204, Laxmi Nagar, Delhi 110092")
    d.text((200, y + 40), "This card is the property of UIDAI. Aadhaar is not proof of citizenship.", font=f(22), fill=0)
    save_pdf("12_kyc_aadhaar_valid", [scanify(p, -0.8, noise=6, seed=24)], "PASS → auto-approve")

# ── 13 Passport (Underage) ────────────────────────────────────────────────────
def kyc_13_passport_underage():
    p, d = page()
    y = _card(d, "REPUBLIC OF INDIA — PASSPORT", "Ministry of External Affairs")
    y = _kv(d, y, "Passport No", "P1234567")
    y = _kv(d, y, "Name", "ARJUN MEHTA")
    y = _kv(d, y, "Date of Birth", "10/03/2012")   # 14 years old in 2026
    y = _kv(d, y, "Date of Issue", "15/08/2022")
    y = _kv(d, y, "Date of Expiry", "14/08/2032")
    y = _kv(d, y, "Nationality", "INDIAN")
    save_pdf("13_kyc_passport_underage", [scanify(p, 0.7, noise=7, seed=25)],
             "REVIEW → underage (DOB 2012 → age 14)")


# ═══════════════════════════════════════════════════════════════════════════
# CONTRACT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════
def _contract_base(d, title, date, party_a, party_b):
    d.text((120, 100), title, font=f(56, True), fill=0)
    hr(d, 185)
    text = (f"This Agreement is entered into on {date} between {party_a} "
            f'("Client") and {party_b} ("Vendor").')
    d.text((120, 210), text, font=f(26), fill=0)
    hr(d, 290)
    return 320

def _clause(d, y, num, heading, body):
    d.text((120, y), f"{num}. {heading}", font=f(32, True), fill=0)
    y += 46
    # Word-wrap body at ~100 chars
    words = body.split()
    line, lines = "", []
    for w in words:
        if len(line) + len(w) + 1 > 95:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    for ln in lines:
        d.text((140, y), ln, font=f(27), fill=0)
        y += 38
    return y + 24

def _contract_footer(d, y, party_a, party_b):
    hr(d, y + 20)
    y += 60
    d.text((120, y), f"Signed for Client:", font=f(28), fill=0)
    d.text((120, y + 46), f"{party_a}", font=f(26, True), fill=0)
    d.text((120, y + 100), "Signature: ____________________   Date: ___________", font=f(26), fill=0)
    d.text((960, y), f"Signed for Vendor:", font=f(28), fill=0)
    d.text((960, y + 46), f"{party_b}", font=f(26, True), fill=0)
    d.text((960, y + 100), "Signature: ____________________   Date: ___________", font=f(26), fill=0)

# ── 14 Standard Contract → auto-approve ──────────────────────────────────────
def contract_14_standard():
    p, d = page()
    y = _contract_base(d, "SERVICE AGREEMENT", "01/11/2026",
                       "Kalyani Engineering Pvt Ltd", "Shree Traders")
    clauses = [
        ("Scope of Services", "Vendor shall supply industrial fasteners and fittings as per purchase orders raised by Client, meeting the quality standards specified in Schedule A."),
        ("Payment Terms", "Client shall make payment within 30 days of receipt of a valid tax invoice. Payments delayed beyond this period shall attract interest at 1.5% per month."),
        ("Termination", "Either party may terminate this Agreement by giving 30 days written notice. Termination for cause may be effected immediately with written justification."),
        ("Limitation of Liability", "Vendor's total liability under this Agreement shall not exceed the value of the specific order that gave rise to the claim, capped at Rs. 5,00,000."),
        ("Governing Law", "This Agreement shall be governed by and construed in accordance with the laws of India. Any disputes shall be subject to the exclusive jurisdiction of courts in Pune, Maharashtra."),
    ]
    for i, (h, b) in enumerate(clauses, 1):
        y = _clause(d, y, i, h, b)
    _contract_footer(d, y, "Kalyani Engineering Pvt Ltd", "Shree Traders")
    save_pdf("14_contract_standard", [scanify(p, 0.5, seed=31)], "PASS → auto-approve")

# ── 15 Long Payment Terms (> 45 days) ─────────────────────────────────────────
def contract_15_long_payment():
    p, d = page()
    y = _contract_base(d, "SERVICE AGREEMENT", "05/11/2026",
                       "Kalyani Engineering Pvt Ltd", "Apex Packaging LLP")
    clauses = [
        ("Scope of Services", "Vendor shall supply packaging materials including HDPE bags, PP woven sacks, and stretch film rolls as per Client's quarterly requirements."),
        ("Payment Terms", "Client shall make payment within 90 days of receipt of a valid tax invoice. No early payment discount shall be applicable."),
        ("Termination", "Either party may terminate this Agreement by giving 30 days written notice to the other party."),
        ("Limitation of Liability", "Vendor's total liability shall not exceed Rs. 2,00,000 per incident or the order value, whichever is lower."),
        ("Governing Law", "This Agreement is governed by the laws of India. Disputes shall be settled by arbitration in Bengaluru."),
    ]
    for i, (h, b) in enumerate(clauses, 1):
        y = _clause(d, y, i, h, b)
    _contract_footer(d, y, "Kalyani Engineering Pvt Ltd", "Apex Packaging LLP")
    save_pdf("15_contract_long_payment", [scanify(p, -0.9, seed=32)],
             "REVIEW → payment_terms 90 days > max 45")

# ── 16 No Liability Cap ───────────────────────────────────────────────────────
def contract_16_no_liability():
    p, d = page()
    y = _contract_base(d, "SUPPLY AGREEMENT", "08/11/2026",
                       "Kalyani Engineering Pvt Ltd", "Mehta Steel Works")
    clauses = [
        ("Scope of Services", "Vendor shall supply steel sheets, angles, and bars as per purchase orders. All materials shall conform to IS 2062 standards."),
        ("Payment Terms", "Payment shall be made within 45 days of invoice receipt. Disputes on invoice amounts must be raised within 10 days."),
        ("Termination", "Either party may terminate with 45 days written notice. Immediate termination is permitted on material breach."),
        ("Liability", "Vendor shall be liable for all losses, damages, claims, and expenses arising from defective materials supplied, without any cap or limitation."),
        ("Governing Law", "This Agreement is governed by Indian law. The courts of Surat, Gujarat shall have jurisdiction."),
    ]
    for i, (h, b) in enumerate(clauses, 1):
        y = _clause(d, y, i, h, b)
    _contract_footer(d, y, "Kalyani Engineering Pvt Ltd", "Mehta Steel Works")
    save_pdf("16_contract_no_liability_cap", [scanify(p, 1.1, seed=33)],
             "REVIEW → no liability cap (unlimited liability clause)")

# ── 17 Short Termination Notice (< 30 days) ───────────────────────────────────
def contract_17_short_notice():
    p, d = page()
    y = _contract_base(d, "SERVICES AGREEMENT", "10/11/2026",
                       "Kalyani Engineering Pvt Ltd", "Ravi & Sons Traders")
    clauses = [
        ("Scope of Services", "Vendor shall provide electrical component supply services including wiring, MCBs, and accessories as ordered by Client."),
        ("Payment Terms", "Client shall make payment within 30 days of valid invoice. Late payments attract 2% monthly interest."),
        ("Termination", "Either party may terminate this Agreement by giving 7 days written notice. No compensation shall be payable on termination."),
        ("Limitation of Liability", "Vendor's liability is limited to Rs. 1,00,000 or the order value, whichever is less."),
        ("Governing Law", "This Agreement is governed by the laws of India."),
    ]
    for i, (h, b) in enumerate(clauses, 1):
        y = _clause(d, y, i, h, b)
    _contract_footer(d, y, "Kalyani Engineering Pvt Ltd", "Ravi & Sons Traders")
    save_pdf("17_contract_short_notice", [scanify(p, -1.4, seed=34)],
             "REVIEW → termination notice only 7 days < min 30")


# ═══════════════════════════════════════════════════════════════════════════
# CLAIM BUILDERS
# ═══════════════════════════════════════════════════════════════════════════
def _claim_base(d, claim_no, policy_no, claimant, dol, claim_date, amount, description):
    d.text((120, 100), "MOTOR INSURANCE CLAIM FORM", font=f(54, True), fill=0)
    hr(d, 185)
    fields = [
        ("Claim No", claim_no), ("Policy No", policy_no), ("Claimant Name", claimant),
        ("Date of Loss", dol), ("Claim Submission Date", claim_date),
        ("Claim Amount (Rs.)", f"{amount:,.2f}"),
        ("Vehicle Reg. No", "MH14 AB 1234"), ("Description of Loss", description),
    ]
    y = 220
    for label, val in fields:
        d.text((120, y), f"{label}:", font=f(28), fill=0)
        d.text((600, y), val, font=f(30, True), fill=0)
        y += 56
    return y

# ── 18 Valid Claim → auto-approve ────────────────────────────────────────────
def claim_18_valid():
    p, d = page()
    y = _claim_base(d, "CLM-7001", "POL-77120", "Anita Deshmukh",
                    "05/10/2026", "10/10/2026", 85000.0,
                    "Rear collision damage at Pune-Mumbai Expressway km 42")
    d.text((120, y + 60), "Supporting Documents Attached:", font=f(28, True), fill=0)
    for doc in ["  ✓ FIR Copy", "  ✓ Repair Estimate (3 workshops)", "  ✓ Photos of damage (12 images)",
                "  ✓ RC Book copy", "  ✓ Driving Licence copy"]:
        y += 48
        d.text((120, y + 60), doc, font=f(26), fill=0)
    d.text((120, y + 180), "Claimant Signature: ____________", font=f(30), fill=0)
    save_pdf("18_claim_valid", [scanify(p, 0.3, seed=41)], "PASS → within policy limit Rs.1,50,000")

# ── 19 Claim Over Policy Limit ────────────────────────────────────────────────
def claim_19_over_limit():
    p, d = page()
    y = _claim_base(d, "CLM-7002", "POL-77120", "Anita Deshmukh",
                    "02/11/2026", "08/11/2026", 190000.0,     # limit is 1,50,000
                    "Total loss — vehicle destroyed in flood damage")
    d.text((120, y + 60), "Supporting Documents:", font=f(28, True), fill=0)
    for doc in ["  ✓ Surveyor Report", "  ✓ Photos (flood damage)", "  ✓ RC Book"]:
        y += 48
        d.text((120, y + 60), doc, font=f(26), fill=0)
    d.text((120, y + 180), "Claimant Signature: ____________", font=f(30), fill=0)
    save_pdf("19_claim_over_limit", [scanify(p, -0.6, seed=42)],
             "REVIEW → claim Rs.1,90,000 > policy limit Rs.1,50,000")

# ── 20 Blurry Phone-Photo Scan → low OCR confidence ──────────────────────────
def claim_20_blurry():
    p, d = page()
    y = _claim_base(d, "CLM-7003", "POL-77120", "Anita Deshmukh",
                    "14/11/2026", "16/11/2026", 42000.0,
                    "Side-mirror and door panel damage from parking lot collision")
    d.text((120, y + 180), "Claimant Signature: ____________", font=f(30), fill=0)
    # Degrade heavily — low resolution + blur + strong rotation
    a = np.array(p)
    a = cv2.resize(a, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), 1.8)
    rng = np.random.default_rng(99)
    a = np.clip(a.astype(np.float32) * 0.80 + rng.normal(0, 15, a.shape), 0, 255).astype(np.uint8)
    h2, w2 = a.shape
    M = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), 5.5, 1.0)
    a = cv2.warpAffine(a, M, (w2, h2), borderValue=180)
    img = Image.fromarray(a)
    save_jpg("20_claim_blurry_photo", img, "REVIEW → low OCR confidence (blurry scan)")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n📄 Generating DocAgent demo dataset (20 documents)...\n")

    print("── Invoices ──")
    inv_01_clean()
    inv_02_mehta()
    inv_03_highvalue()
    inv_04_unregistered()
    inv_05_math_error()
    inv_06_duplicate()
    inv_07_no_po()
    inv_08_future_date()

    print("\n── KYC Documents ──")
    kyc_09_pan_valid()
    kyc_10_pan_bad()
    kyc_11_dl_expired()
    kyc_12_aadhaar_valid()
    kyc_13_passport_underage()

    print("\n── Contracts ──")
    contract_14_standard()
    contract_15_long_payment()
    contract_16_no_liability()
    contract_17_short_notice()

    print("\n── Insurance Claims ──")
    claim_18_valid()
    claim_19_over_limit()
    claim_20_blurry()

    files = sorted(os.listdir(OUT))
    print(f"\n✅ Done! {len(files)} files written to  demo_dataset/\n")

    # Print a demo cheat-sheet
    print("=" * 70)
    print("DEMO CHEAT-SHEET — What each document triggers")
    print("=" * 70)
    scenarios = [
        ("01_inv_shree_clean.pdf",         "Invoice",   "✅ AUTO-APPROVE — clean, PO matched, math correct"),
        ("02_inv_mehta_valid.pdf",         "Invoice",   "✅ AUTO-APPROVE — 2 pages, CGST+SGST, valid"),
        ("03_inv_apex_highvalue.pdf",      "Invoice",   "⚠️  REVIEW    — amount > ₹50K threshold"),
        ("04_inv_unregistered_vendor.pdf", "Invoice",   "🔴 FLAG       — vendor not in master list"),
        ("05_inv_math_error.pdf",          "Invoice",   "🔴 FLAG       — line items ≠ printed total (₹3K gap)"),
        ("06_inv_duplicate.pdf",           "Invoice",   "⚠️  REVIEW    — same invoice number already processed"),
        ("07_inv_no_po.pdf",               "Invoice",   "⚠️  REVIEW    — PO reference missing"),
        ("08_inv_future_date.pdf",         "Invoice",   "🔴 FLAG       — invoice date is in the future"),
        ("09_kyc_pan_valid.pdf",           "KYC-PAN",  "✅ AUTO-APPROVE — valid PAN format and DOB"),
        ("10_kyc_pan_bad_format.pdf",      "KYC-PAN",  "🔴 FLAG       — PAN number format invalid"),
        ("11_kyc_dl_expired.pdf",          "KYC-DL",   "🔴 FLAG       — licence expired May 2023"),
        ("12_kyc_aadhaar_valid.pdf",       "KYC-Aadh", "✅ AUTO-APPROVE — Aadhaar, 24 y/o, valid"),
        ("13_kyc_passport_underage.pdf",   "KYC-Pass", "🔴 FLAG       — holder is 14 years old"),
        ("14_contract_standard.pdf",       "Contract", "✅ AUTO-APPROVE — all clauses within limits"),
        ("15_contract_long_payment.pdf",   "Contract", "⚠️  REVIEW    — payment 90 days > max 45"),
        ("16_contract_no_liability_cap.pdf","Contract","🔴 FLAG       — no liability cap clause"),
        ("17_contract_short_notice.pdf",   "Contract", "⚠️  REVIEW    — termination notice 7 days < min 30"),
        ("18_claim_valid.pdf",             "Claim",    "✅ AUTO-APPROVE — Rs.85K within policy limit"),
        ("19_claim_over_limit.pdf",        "Claim",    "⚠️  REVIEW    — Rs.1.9L > policy limit Rs.1.5L"),
        ("20_claim_blurry_photo.jpg",      "Claim",    "⚠️  REVIEW    — low OCR confidence, blurry scan"),
    ]
    for fname, dtype, note in scenarios:
        print(f"  {fname:<40} {dtype:<10} {note}")
    print()

