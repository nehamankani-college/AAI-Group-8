"""Generate a verification test set (different layouts than the demo samples) + expected.csv answer key.
Documents are SYNTHETIC. Replace/add your own real scans and add rows to expected.csv to test real data."""
import csv
import json
import os

import cv2
import numpy as np
from PIL import Image

import make_samples as ms
from make_samples import W, H, f, page, scanify, inr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testset")
os.makedirs(OUT, exist_ok=True)
ROWS = []


def scribble(d, x, y):
    pts = [(x, y - 12), (x + 35, y - 22), (x + 70, y - 4), (x + 105, y - 24), (x + 150, y - 6), (x + 190, y - 18)]
    d.line(pts, fill=0, width=4)


def invoice(name, vendor, addr, gstin, no_label, no, date, due, po, billto, items, taxes, total, signed=True,
            sub_label="Subtotal", skew=1.2, photo=False, expected=None, decision="auto_approve", seed=11):
    p, d = page()
    d.text((120, 110), "TAX INVOICE", font=f(60, True), fill=0)
    d.text((120, 220), vendor, font=f(44, True), fill=0)
    d.text((120, 280), addr, font=f(28), fill=0)
    d.text((120, 320), "GSTIN: " + gstin, font=f(32), fill=0)
    d.text((1000, 220), "%s: %s" % (no_label, no), font=f(32), fill=0)
    d.text((1000, 275), "Invoice Date: %s" % date, font=f(32), fill=0)
    d.text((1000, 330), "Due Date: %s" % due, font=f(32), fill=0)
    d.text((1000, 385), "PO Number: %s" % po, font=f(32), fill=0)
    d.text((120, 440), "Bill To: %s" % billto, font=f(30), fill=0)
    d.line((120, 540, W - 120, 540), fill=0, width=3)
    ms.rows(d, 120, 560, [("Description", "Qty", "Rate", "Amount")], (760, 160, 240, 260), 30, bold=True)
    d.line((120, 610, W - 120, 610), fill=0, width=2)
    y = ms.rows(d, 120, 630, [(a, b, inr(c), inr(e)) for a, b, c, e in items], (760, 160, 240, 260), 30)
    sub = sum(e for *_, e in items)
    d.line((900, y + 20, W - 120, y + 20), fill=0, width=2)
    d.text((900, y + 50), "%s:   %s" % (sub_label, inr(sub)), font=f(30), fill=0)
    yy = y + 105
    for lab, amt in taxes:
        d.text((900, yy), "%s:   %s" % (lab, inr(amt)), font=f(30), fill=0)
        yy += 55
    d.text((900, yy + 20), "Total: %s" % inr(total), font=f(36, True), fill=0)
    sy = H - 330
    d.text((120, sy), "Authorised Signatory: ____________________", font=f(28), fill=0)
    if signed:
        scribble(d, 420, sy)
    img = photo_effect(p) if photo else scanify(p, skew, seed=seed)
    ext = ".jpg" if photo else ".pdf"
    save(img, name + ext)
    ROWS.append({"file": name + ext, "doc_type": "invoice", "decision": decision, "fields": json.dumps(expected or {})})


def photo_effect(p):
    a = np.array(p).astype(np.float32)
    h, w = a.shape
    grad = np.linspace(0.78, 1.0, w)[None, :] * np.linspace(0.9, 1.0, h)[:, None]
    a = a * grad
    a = cv2.GaussianBlur(a, (0, 0), 1.1)
    a += np.random.default_rng(5).normal(0, 7, a.shape)
    a = np.clip(a, 0, 255).astype(np.uint8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 5.5, 1.0)
    a = cv2.warpAffine(a, M, (w, h), borderValue=170)
    a = cv2.resize(a, None, fx=0.62, fy=0.62, interpolation=cv2.INTER_AREA)
    return Image.fromarray(a)


def save(img, fname):
    path = os.path.join(OUT, fname)
    if fname.endswith(".pdf"):
        img.convert("RGB").save(path, resolution=200)
    else:
        img.convert("RGB").save(path, quality=70)


def verhoeff_check_digit(num):
    d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
         [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
         [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
         [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    p = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
         [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
         [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
    inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = d[c][p[(i + 1) % 8][int(ch)]]
    return str(inv[c])


def main():
    apex = dict(vendor="Apex Packaging LLP", addr="Plot 9, Peenya Industrial Area, Bengaluru 560058",
                gstin="29AAFFA9988B1Z2", billto="Kalyani Engineering Pvt Ltd")
    ok_items = [("Corrugated Cartons", 200, 60, 12000), ("Packing Tape Rolls", 100, 40, 4000)]
    exp_ok = {"invoice_no": "APX-2026-044", "invoice_date": "2026-10-10", "vendor": "Apex Packaging LLP",
              "po_number": "PO-3310", "subtotal": 16000.0, "total": 18880.0}
    # 01 clean, fully valid -> should auto-approve
    invoice("01_inv_apex_clean", no_label="Invoice Number", no="APX-2026-044", date="10/10/2026", due="09/11/2026",
            po="PO-3310", items=ok_items, taxes=[("GST 18%", 2880)], total=18880, expected=exp_ok, **apex)
    # 02 different wording, CGST + SGST equal amounts (must NOT be flagged as duplicate), above approval threshold
    invoice("02_inv_mehta_cgst_sgst", vendor="Mehta Steel Works", addr="GIDC Sachin, Surat 394230",
            gstin="24AAACM5678K1ZP", no_label="Invoice No", no="MSW/2026/118", date="05/10/2026", due="04/11/2026",
            po="PO-2305", billto="Kalyani Engineering Pvt Ltd",
            items=[("Steel Coil 2mm", 10, 6500, 65000), ("Steel Sheet 1.5mm", 20, 1300, 26000)],
            taxes=[("CGST 9%", 8190), ("SGST 9%", 8190)], total=107380, skew=-1.5, decision="needs_review",
            expected={"invoice_no": "MSW/2026/118", "invoice_date": "2026-10-05", "vendor": "Mehta Steel Works",
                      "po_number": "PO-2305", "subtotal": 91000.0, "total": 107380.0})
    # 03 unknown vendor + PO belongs to another vendor
    invoice("03_inv_unregistered_vendor", vendor="Sunrise Stationers", addr="12 Station Road, Nashik 422001",
            gstin="27AAKCS1122M1Z9", no_label="Invoice No", no="SS-5521", date="14/10/2026", due="13/11/2026",
            po="PO-2291", billto="Kalyani Engineering Pvt Ltd", items=[("A4 Paper Reams", 50, 300, 15000)],
            taxes=[("GST 12%", 1800)], total=16800, decision="needs_review", skew=0.8,
            expected={"invoice_no": "SS-5521", "vendor": "Sunrise Stationers", "po_number": "PO-2291", "total": 16800.0})
    # 04 arithmetic error inside a line (200 x 60 printed as 13,000)
    invoice("04_inv_line_math_error", no_label="Invoice No", no="APX-2026-051", date="11/10/2026", due="10/11/2026",
            po="PO-3310", items=[("Corrugated Cartons", 200, 60, 13000), ("Packing Tape Rolls", 100, 40, 4000)],
            taxes=[("GST 18%", 3060)], total=20060, decision="needs_review", skew=-0.7,
            expected={"invoice_no": "APX-2026-051", "po_number": "PO-3310", "total": 20060.0}, **apex)
    # 05 phone photo of a valid invoice (tilted, shaded, low-res)
    invoice("05_inv_apex_photo", no_label="Invoice Number", no="APX-2026-060", date="12/10/2026", due="11/11/2026",
            po="PO-3310", items=ok_items, taxes=[("GST 18%", 2880)], total=18880, photo=True, decision="auto_approve",
            expected={"invoice_no": "APX-2026-060", "invoice_date": "2026-10-12", "vendor": "Apex Packaging LLP",
                      "po_number": "PO-3310", "subtotal": 16000.0, "total": 18880.0}, **apex)
    # 06 not a business document at all
    p, d = page()
    d.text((300, 200), "THE GOLDEN SPOON CAFE", font=f(58, True), fill=0)
    y = 360
    for n, pr in [("Masala Dosa", 120), ("Filter Coffee", 60), ("Paneer Wrap", 150), ("Mango Lassi", 90)]:
        d.text((300, y), "%s ........ Rs %d" % (n, pr), font=f(36), fill=0)
        y += 90
    save(scanify(p, 1.0, seed=21), "06_menu_unknown.jpg")
    ROWS.append({"file": "06_menu_unknown.jpg", "doc_type": "other", "decision": "reject", "fields": "{}"})
    # 07 Aadhaar-style card with a valid Verhoeff number
    base = "23456789012"
    aad = base + verhoeff_check_digit(base)
    p, d = page()
    d.rectangle((150, 200, W - 150, 1000), outline=0, width=5)
    d.text((200, 230), "Government of India", font=f(44, True), fill=0)
    d.text((200, 300), "Aadhaar - Unique Identification Authority of India", font=f(30), fill=0)
    d.text((200, 420), "Name: Anita Deshmukh", font=f(40), fill=0)
    d.text((200, 510), "DOB: 12/08/1990", font=f(38), fill=0)
    d.text((200, 640), "%s %s %s" % (aad[:4], aad[4:8], aad[8:]), font=f(68, True), fill=0)
    save(scanify(p, -2.0, noise=8, seed=22), "07_aadhaar_valid.jpg")
    ROWS.append({"file": "07_aadhaar_valid.jpg", "doc_type": "kyc", "decision": "auto_approve",
                 "fields": json.dumps({"id_number": "%s %s %s" % (aad[:4], aad[4:8], aad[8:]), "name": "Anita Deshmukh",
                                       "dob": "1990-08-12"})})
    # 08 contract that follows the standard template and is signed
    p, d = page()
    d.text((120, 120), "SERVICE AGREEMENT", font=f(58, True), fill=0)
    d.text((120, 220), "This Agreement is made on 02/10/2026 between Kalyani Engineering Pvt Ltd", font=f(28), fill=0)
    d.text((120, 262), '("Client") and Apex Packaging LLP ("Vendor").', font=f(28), fill=0)
    y = 380
    for h, t in [("1. Scope of Services", "Vendor shall supply goods and services as per purchase orders."),
                 ("2. Payment Terms", "Payment shall be made within 30 days of invoice date."),
                 ("3. Termination", "Either party may terminate this Agreement with 30 days written notice."),
                 ("4. Liability", "Vendor liability is limited to the value of the relevant purchase order."),
                 ("5. Governing Law", "This Agreement is governed by the laws of India.")]:
        d.text((120, y), h, font=f(32, True), fill=0)
        d.text((120, y + 48), t, font=f(28), fill=0)
        y += 140
    for i, who in enumerate(["Client", "Vendor"]):
        yy = y + 80 + 70 * i
        d.text((120, yy), "Signed for %s: ____________________" % who, font=f(28), fill=0)
        scribble(d, 330, yy)
    save(scanify(p, 0.9, seed=23), "08_contract_standard.pdf")
    ROWS.append({"file": "08_contract_standard.pdf", "doc_type": "contract", "decision": "auto_approve",
                 "fields": json.dumps({"effective_date": "2026-10-02", "party_b": "Apex Packaging LLP",
                                       "payment_terms_days": 30, "termination_notice_days": 30})})
    # 09 claim within limit and signed
    p, d = page()
    d.text((120, 120), "MOTOR INSURANCE CLAIM FORM", font=f(54, True), fill=0)
    y = 280
    for line in ["Claim No: CLM-6002", "Policy No: POL-77120", "Claimant: Anita Deshmukh", "Date of Loss: 05/10/2026",
                 "Claim Date: 12/10/2026", "Claim Amount: 90000.00", "Description: Windshield and bumper damage"]:
        d.text((120, y), line, font=f(34), fill=0)
        y += 90
    d.text((120, y + 100), "Claimant Signature: ____________", font=f(30), fill=0)
    scribble(d, 420, y + 100)
    save(scanify(p, -0.6, seed=24), "09_claim_ok.pdf")
    ROWS.append({"file": "09_claim_ok.pdf", "doc_type": "claim", "decision": "auto_approve",
                 "fields": json.dumps({"claim_number": "CLM-6002", "policy_number": "POL-77120",
                                       "claim_amount": 90000.0, "date_of_loss": "2026-10-05"})})
    # 10 exact duplicate of 01 (same invoice number) -> must be caught
    invoice("10_inv_apex_duplicate", no_label="Invoice Number", no="APX-2026-044", date="10/10/2026", due="09/11/2026",
            po="PO-3310", items=ok_items, taxes=[("GST 18%", 2880)], total=18880, decision="reject", skew=0.4,
            expected={"invoice_no": "APX-2026-044", "total": 18880.0}, **apex)
    ROWS.sort(key=lambda r: r["file"])
    with open(os.path.join(OUT, "expected.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["file", "doc_type", "decision", "fields"])
        w.writeheader()
        w.writerows(ROWS)
    print("wrote", len(ROWS), "documents + expected.csv to", OUT)


if __name__ == "__main__":
    main()
