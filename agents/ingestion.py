"""Agent 1 - Document Ingestion: load PDF/image/photo, split pages, deskew, denoise, quality score."""
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

TARGET_WIDTH = 1654  # ~A4 @ 200 dpi; photos are scaled to this so OCR sees consistent text size


def estimate_skew(gray, limit=6.0, step=0.25):
    """Projection-profile skew estimate: the angle that makes text rows sharpest."""
    small = cv2.resize(gray, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    binary = (cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0).astype(np.float32)
    h, w = binary.shape
    best, best_angle = -1, 0.0
    for a in np.arange(-limit, limit + step, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST)
        v = np.var(r.sum(axis=1))
        if v > best:
            best, best_angle = v, a
    return float(best_angle)


def quality_score(gray):
    sharp = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 300.0)
    contrast = min(1.0, float(gray.std()) / 60.0)
    return round(0.65 * sharp + 0.35 * contrast, 2)


def load_pages(path, dpi=200):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            try:
                import pymupdf as fitz  # PyMuPDF: pure pip install, no Poppler needed
            except ImportError:
                import fitz
            doc = fitz.open(path)
            pages = []
            for pg in doc:
                pix = pg.get_pixmap(dpi=dpi)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            return pages, "pdf"
        except ImportError:
            from pdf2image import convert_from_path  # fallback (needs Poppler)
            return convert_from_path(path, dpi=dpi), "pdf"
    img = ImageOps.exif_transpose(Image.open(path))
    return [img], "image"


def ingest(path, dpi=200, upscale=1.0):
    """Return {pages:[{page,image(np gray),width,height,deskew_applied,angle,quality_score}], ...}."""
    raw, kind = load_pages(path, dpi)
    pages = []
    for i, im in enumerate(raw, 1):
        gray = np.array(im.convert("L"))
        h, w = gray.shape
        if kind == "image" or w < 1200:
            s = TARGET_WIDTH / w
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        if upscale != 1.0:
            gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        angle = estimate_skew(gray)
        deskewed = False
        if abs(angle) >= 0.3:
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=int(np.median(gray)))
            deskewed = True
        clean = cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=15)
        pages.append({"page": i, "image": clean, "width": clean.shape[1], "height": clean.shape[0],
                      "deskew_applied": deskewed, "skew_angle": round(angle, 2),
                      "quality_score": quality_score(clean)})
    return {"kind": kind, "dpi": dpi, "upscale": upscale, "pages": pages}


def summary(ing):
    return {"pages": len(ing["pages"]), "source_kind": ing["kind"], "dpi": ing["dpi"], "upscale": ing["upscale"],
            "deskew_applied": any(p["deskew_applied"] for p in ing["pages"]),
            "quality_score": round(float(np.mean([p["quality_score"] for p in ing["pages"]])), 2),
            "per_page": [{k: p[k] for k in ("page", "width", "height", "deskew_applied", "skew_angle", "quality_score")}
                         for p in ing["pages"]]}
