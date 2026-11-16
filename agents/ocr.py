"""Agent 3 - Layout & OCR: text + bounding boxes + confidence; rows/segments preserve table structure."""
import numpy as np
import pytesseract
from PIL import Image

import os
import shutil

from .common import ascii_digits

if shutil.which("tesseract") is None:  # common Windows install locations
    for _p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break

LOW_CONF = 0.60


def _words(img, lang):
    d = pytesseract.image_to_data(Image.fromarray(img), lang=lang, config="--psm 6",
                                  output_type=pytesseract.Output.DICT)
    out = []
    for i, t in enumerate(d["text"]):
        t = ascii_digits(t.strip())
        c = float(d["conf"][i])
        if not t or c < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        out.append({"text": t, "bbox": [x, y, x + w, y + h], "conf": round(c / 100, 3)})
    return out


def _rows(words):
    rows = []
    for w in sorted(words, key=lambda w: (w["bbox"][1] + w["bbox"][3]) / 2):
        yc = (w["bbox"][1] + w["bbox"][3]) / 2
        h = max(10, w["bbox"][3] - w["bbox"][1])
        if rows and abs(yc - rows[-1]["yc"]) < 0.6 * h:
            rows[-1]["w"].append(w)
            rows[-1]["yc"] = np.mean([(x["bbox"][1] + x["bbox"][3]) / 2 for x in rows[-1]["w"]])
        else:
            rows.append({"yc": yc, "w": [w]})
    return [sorted(r["w"], key=lambda w: w["bbox"][0]) for r in rows]


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def ocr_page(page, lang="eng"):
    img = page["image"]
    gap = 0.05 * img.shape[1]
    rows = _rows(_words(img, lang))
    segments, words_all, row_index, text_parts, offset = [], [], [], [], 0
    for r_i, row in enumerate(rows):
        segs, cur = [], [row[0]]
        for w in row[1:]:
            if w["bbox"][0] - cur[-1]["bbox"][2] > gap:
                segs.append(cur)
                cur = [w]
            else:
                cur.append(w)
        segs.append(cur)
        idxs = []
        for sw in segs:
            text, pos = "", 0
            for w in sw:
                if text:
                    text += " "
                start = offset + len(text)
                text += w["text"]
                words_all.append({"text": w["text"], "bbox": w["bbox"], "conf": w["conf"], "start": start,
                                  "end": start + len(w["text"]), "seg": len(segments)})
            conf = float(np.mean([w["conf"] for w in sw]))
            segments.append({"idx": len(segments), "row": r_i, "text": text, "bbox": _union([w["bbox"] for w in sw]),
                             "conf": round(conf, 3), "low_confidence": conf < LOW_CONF})
            idxs.append(len(segments) - 1)
            text_parts.append(text)
            offset += len(text) + 1  # newline separator
        row_index.append(idxs)
    mean = float(np.mean([w["conf"] for w in words_all])) if words_all else 0.0
    return {"page": page["page"], "width": page["width"], "height": page["height"], "text": "\n".join(text_parts),
            "words": words_all, "segments": segments, "rows": row_index, "mean_conf": round(mean, 3)}


def run_ocr(ingested, lang="eng"):
    pages = [ocr_page(p, lang) for p in ingested["pages"]]
    allw = [w["conf"] for p in pages for w in p["words"]]
    return {"lang": lang, "pages": pages, "mean_conf": round(float(np.mean(allw)), 3) if allw else 0.0,
            "word_count": len(allw),
            "low_conf_segments": sum(1 for p in pages for s in p["segments"] if s["low_confidence"])}
