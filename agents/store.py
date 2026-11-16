"""Tiny file-based record store: out/<doc_id>/record.json + page images + crops + outputs."""
import json
import os
import shutil

import cv2

from .common import OUT


def doc_dir(doc_id):
    return os.path.join(OUT, doc_id)


def save(record):
    d = doc_dir(record["id"])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "record.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, ensure_ascii=False, default=str)


def load(doc_id):
    p = os.path.join(doc_dir(doc_id), "record.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def list_records():
    recs = []
    for name in os.listdir(OUT):
        r = load(name) if os.path.isdir(os.path.join(OUT, name)) else None
        if r:
            recs.append(r)
    return sorted(recs, key=lambda r: r["created_at"], reverse=True)


def save_page_images(doc_id, images):
    d = doc_dir(doc_id)
    os.makedirs(d, exist_ok=True)
    for i, img in enumerate(images, 1):
        cv2.imwrite(os.path.join(d, "page%d.png" % i), img)


def load_page_images(doc_id, n):
    d = doc_dir(doc_id)
    return [cv2.imread(os.path.join(d, "page%d.png" % i), cv2.IMREAD_GRAYSCALE) for i in range(1, n + 1)]


def clear_all():
    for name in os.listdir(OUT):
        p = os.path.join(OUT, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
