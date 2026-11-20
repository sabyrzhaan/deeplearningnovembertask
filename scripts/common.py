"""Общие утилиты: поиск изображений, пути к разметке, чтение YOLO-разметки."""
import os
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(src):
    """src: папка с картинками, .txt со списком путей или список таких элементов."""
    if isinstance(src, (list, tuple)):
        out = []
        for s in src:
            out += list_images(s)
        return out
    p = Path(src)
    if p.is_file() and p.suffix == ".txt":
        return [Path(l.strip()) for l in p.read_text().splitlines() if l.strip()]
    return sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)


def label_path(img):
    """Путь к .txt-разметке по пути к картинке (та же логика, что в Ultralytics:
    последняя папка /images/ заменяется на /labels/)."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    s = str(img)
    s = sb.join(s.rsplit(sa, 1))
    return Path(s).with_suffix(".txt")


def read_yolo_labels(path):
    """Читает YOLO-разметку: строки `class x_center y_center width height` (0..1).
    Возвращает (rows, issues): rows = [(cls, x, y, w, h), ...], issues = [описания проблем]."""
    rows, issues = [], []
    path = Path(path)
    if not path.exists():
        return rows, ["missing_label_file"]
    for k, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            issues.append(f"line {k}: ожидалось 5 чисел, получено {len(parts)}")
            continue
        try:
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            issues.append(f"line {k}: не число")
            continue
        rows.append((c, x, y, w, h))
    return rows, issues


def load_data_cfg(path="configs/data.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
