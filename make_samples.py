"""Generate synthetic sample documents (scan-like, slightly skewed + noisy) for the demo."""
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(OUT, exist_ok=True)
FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf",
                   "/System/Library/Fonts/Supplemental/Arial.ttf"]
BOLD_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf",
                   "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]


def _pick(cands):
    for c in cands:
        try:
            ImageFont.truetype(c, 20)
            return c
        except OSError:
            continue
    raise SystemExit("No usable TrueType font found; edit FONT_CANDIDATES in make_samples.py")


FONT = _pick(FONT_CANDIDATES)
BOLD = _pick(BOLD_CANDIDATES)
W, H = 1654, 2339  # A4 @ 200dpi


def f(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def page():
    img = Image.new("L", (W, H), 255)
    return img, ImageDraw.Draw(img)


def scanify(img, angle=0.0, noise=6, seed=1):
    """Skew + noise + slight blur so it looks like a scan."""
    rng = np.random.default_rng(seed)
    a = np.array(img)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    a = cv2.warpAffine(a, M, (W, H), borderValue=235)
    a = a.astype(np.float32) + rng.normal(0, noise, a.shape)
    a = cv2.GaussianBlur(a, (3, 3), 0.6)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def rows(d, x, y, items, widths, size=30, gap=52, bold=False):
    for it in items:
        cx = x
        for text, w in zip(it, widths):
            d.text((cx, y), str(text), font=f(size, bold), fill=0)
            cx += w
        y += gap
    return y


def inr(n):
    return f"{n:,.2f}"


def invoice_pages():
    # Page 1: header + items. Page 2: more items, tax, total.
    # Tax row is DUPLICATED -> printed total is inflated by Rs 2,000 (the brief's scenario).
    p1, d = page()
    d.text((120, 120), "TAX INVOICE", font=f(64, True), fill=0)
    d.text((120, 230), "Shree Traders", font=f(44, True), fill=0)
    d.text((120, 290), "14, MG Road, Pune, Maharashtra 411001", font=f(28), fill=0)
    d.text((120, 330), "GSTIN: 27ABCDE1234F1Z5", font=f(28), fill=0)
    d.text((1000, 230), "Invoice No: INV-0847", font=f(32), fill=0)
    d.text((1000, 285), "Invoice Date: 12/09/2026", font=f(32), fill=0)
    d.text((1000, 340), "Due Date: 12/10/2026", font=f(32), fill=0)
    d.text((1000, 395), "PO Number: PO-2291", font=f(32), fill=0)
    d.text((120, 440), "Bill To: Kalyani Engineering Pvt Ltd", font=f(30), fill=0)
    d.line((120, 540, W - 120, 540), fill=0, width=3)
    rows(d, 120, 560, [("Description", "Qty", "Rate", "Amount")], (760, 160, 240, 260), 30, bold=True)
    d.line((120, 610, W - 120, 610), fill=0, width=2)
    items = [("Steel Brackets 40mm", 100, inr(120), inr(12000)),
             ("Hex Bolts M10", 500, inr(18), inr(9000)),
             ("Mounting Plates", 50, inr(250), inr(12500))]
    rows(d, 120, 630, items, (760, 160, 240, 260), 30)
    d.text((W - 400, H - 160), "Page 1 of 2", font=f(24), fill=0)

    p2, d = page()
    d.text((120, 120), "Invoice No: INV-0847 (contd.)", font=f(32), fill=0)
    rows(d, 120, 220, [("Description", "Qty", "Rate", "Amount")], (760, 160, 240, 260), 30, bold=True)
    d.line((120, 270, W - 120, 270), fill=0, width=2)
    items2 = [("Washers Assorted", 1000, inr(3.5), inr(3500)),
              ("Packaging and Freight", 1, inr(5300), inr(5300))]
    y = rows(d, 120, 290, items2, (760, 160, 240, 260), 30)
    # Sum of items = 12000+9000+12500+3500+5300 = 42,300
    d.line((900, y + 20, W - 120, y + 20), fill=0, width=2)
    d.text((900, y + 50), f"Subtotal:   {inr(42300)}", font=f(30), fill=0)
    d.text((900, y + 105), f"GST:   {inr(2000)}", font=f(30), fill=0)  # (row 1 of 2)
    d.text((900, y + 160), f"GST:   {inr(2000)}", font=f(30), fill=0)
    d.text((900, y + 230), f"Total: {inr(44300)}", font=f(36, True), fill=0)
    d.text((120, H - 330), "Authorised Signatory: ____________________", font=f(28), fill=0)
    d.text((120, H - 160), "Page 2 of 2", font=f(24), fill=0)
    return [scanify(p1, 1.8, seed=1), scanify(p2, -1.2, seed=2)]


def pan_card():
    p, d = page()
    d.rectangle((150, 200, W - 150, 1100), outline=0, width=5)
    d.text((200, 230), "INCOME TAX DEPARTMENT", font=f(46, True), fill=0)
    d.text((200, 300), "GOVT. OF INDIA", font=f(34), fill=0)
    d.text((200, 400), "Permanent Account Number Card", font=f(34), fill=0)
    d.text((200, 480), "ABCPK1234F", font=f(64, True), fill=0)
    d.text((200, 600), "Name", font=f(26), fill=0)
    d.text((200, 640), "SUPRIYA KRISHNAN NAYAK", font=f(40, True), fill=0)
    d.text((200, 740), "Father's Name", font=f(26), fill=0)
    d.text((200, 780), "KRISHNAN NAYAK", font=f(36), fill=0)
    d.text((200, 880), "Date of Birth", font=f(26), fill=0)
    d.text((200, 920), "14/03/2004", font=f(40), fill=0)
    d.text((200, 1040), "Signature: ____________", font=f(28), fill=0)
    d.line([(215, 1020), (260, 995), (300, 1025), (340, 990), (390, 1022), (430, 1000)], fill=0, width=4)
    return [scanify(p, -2.4, noise=9, seed=3)]


def aadhaar_expired_licence():
    p, d = page()
    d.rectangle((150, 200, W - 150, 1150), outline=0, width=5)
    d.text((200, 230), "Driving Licence", font=f(50, True), fill=0)
    d.text((200, 310), "Union of India - Maharashtra", font=f(32), fill=0)
    d.text((200, 420), "DL No: MH12 20150012345", font=f(40, True), fill=0)
    d.text((200, 520), "Name: RAJESH PATIL", font=f(38), fill=0)
    d.text((200, 600), "DOB: 02/07/1985", font=f(38), fill=0)
    d.text((200, 680), "Address: 22 Shivaji Nagar, Pune 411005", font=f(32), fill=0)
    d.text((200, 760), "Valid Till: 01/05/2024", font=f(38), fill=0)
    return [scanify(p, 1.0, noise=8, seed=4)]


def contract():
    p, d = page()
    d.text((120, 120), "SERVICE AGREEMENT", font=f(58, True), fill=0)
    d.text((120, 220), "This Agreement is made on 01/10/2026 between Kalyani Engineering Pvt Ltd", font=f(28), fill=0)
    d.text((120, 262), '("Client") and Shree Traders ("Vendor").', font=f(28), fill=0)
    lines = [
        ("1. Scope of Services", "Vendor shall supply fasteners and fittings as per purchase orders."),
        ("2. Payment Terms", "Payment shall be made within 60 days of invoice date."),
        ("3. Termination", "Either party may terminate this Agreement with 15 days written notice."),
        ("4. Liability", "Vendor liability is unlimited for all direct and indirect losses."),
        ("5. Governing Law", "This Agreement is governed by the laws of India."),
    ]
    y = 380
    for h, t in lines:
        d.text((120, y), h, font=f(32, True), fill=0)
        d.text((120, y + 48), t, font=f(28), fill=0)
        y += 140
    d.text((120, y + 80), "Signed for Client: ____________________", font=f(28), fill=0)
    d.text((120, y + 150), "Signed for Vendor: ____________________", font=f(28), fill=0)
    return [scanify(p, 0.8, seed=5)]


def claim_form(degrade=False):
    p, d = page()
    d.text((120, 120), "MOTOR INSURANCE CLAIM FORM", font=f(54, True), fill=0)
    y = 280
    for line in ["Claim No: CLM-5521", "Policy No: POL-77120", "Claimant: Anita Deshmukh",
                 "Date of Loss: 05/10/2026", "Claim Date: 20/10/2026", "Claim Amount: 185000.00",
                 "Description: Rear collision damage to vehicle MH14 AB 1234"]:
        d.text((120, y), line, font=f(34), fill=0)
        y += 90
    d.text((120, y + 100), "Claimant Signature: ____________", font=f(30), fill=0)
    if not degrade:
        return [scanify(p, 0.5, seed=7)]
    a = np.array(p)
    a = cv2.resize(a, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), 1.3)
    a = np.clip(a.astype(np.float32) * 0.85 + np.random.default_rng(9).normal(0, 12, a.shape), 0, 255).astype(np.uint8)
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.5, 1.0)
    a = cv2.warpAffine(a, M, (w, h), borderValue=200)
    return [Image.fromarray(a)]


def hindi_note():
    # Hindi KYC-style form (needs a Devanagari font; falls back to skipping if unavailable)
    try:
        hf = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", 40)
    except Exception:
        return None
    p, d = page()
    d.text((200, 200), "आधार कार्ड", font=hf, fill=0)
    d.text((200, 320), "नाम: राजेश पाटिल", font=hf, fill=0)
    d.text((200, 420), "जन्म तिथि: 02/07/1985", font=hf, fill=0)
    d.text((200, 520), "2345 6789 0123", font=hf, fill=0)
    return [scanify(p, 1.2, seed=6)]


def save(name, pages):
    if pages is None:
        return
    pages[0].convert("RGB").save(os.path.join(OUT, name + ".pdf"), save_all=True,
                                 append_images=[x.convert("RGB") for x in pages[1:]], resolution=200)
    # a "photo" version of the first page as JPEG
    pages[0].save(os.path.join(OUT, name + ".jpg"), quality=80)


if __name__ == "__main__":
    save("Invoice_0847", invoice_pages())
    save("PAN_card_supriya", pan_card())
    save("DL_expired_rajesh", aadhaar_expired_licence())
    save("Service_Agreement_Shree", contract())
    save("Claim_CLM5521", claim_form())
    cp = claim_form(degrade=True)
    cp[0].save(os.path.join(OUT, "Claim_photo_blurry.jpg"), quality=55)
    save("Aadhaar_hindi", hindi_note())
    # keep only PDFs for most; remove stray jpgs except the PAN photo
    for n in ["Invoice_0847", "DL_expired_rajesh", "Service_Agreement_Shree", "Aadhaar_hindi", "Claim_CLM5521"]:
        p = os.path.join(OUT, n + ".jpg")
        if os.path.exists(p):
            os.remove(p)
    print(sorted(os.listdir(OUT)))
