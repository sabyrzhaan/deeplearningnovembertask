"""Визуализация: разметка (ground truth), предсказания, кривые обучения, случаи ошибок."""
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import label_path, read_yolo_labels  # noqa: E402

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]


def draw_gt_image(img_path, names, thickness=3):
    """Картинка с нарисованными ground-truth боксами (numpy RGB)."""
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    rows, _ = read_yolo_labels(label_path(img_path))
    for (c, x, y, bw, bh) in rows:
        col = PALETTE[c % len(PALETTE)]
        x1, y1, x2, y2 = (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h
        d.rectangle([x1, y1, x2, y2], outline=col, width=thickness)
        label = names[c] if 0 <= c < len(names) else f"?{c}"
        tw = d.textlength(label)
        d.rectangle([x1, y1, x1 + tw + 6, y1 + 14], fill=col)
        d.text((x1 + 3, y1 + 1), label, fill=(255, 255, 255))
    return np.asarray(im)


def _grid(arrays, titles, cols, save, figsize_per=(4.2, 3.4)):
    rows = int(np.ceil(len(arrays) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, a, t in zip(axes, arrays, titles):
        ax.imshow(a)
        ax.set_title(t, fontsize=8)
    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()


def show_ground_truth(images, names, n=8, seed=42, cols=4, save=None):
    """Случайные n картинок (seed фиксирован) с GT-боксами."""
    pick = random.Random(seed).sample(list(images), min(n, len(images)))
    _grid([draw_gt_image(p, names) for p in pick], [Path(p).name for p in pick], cols, save)
    return pick


def show_predictions(model, images, n=8, seed=42, conf=0.25, iou=0.7, imgsz=640, cols=4, save=None):
    """Случайные n картинок с предсказаниями (рамка, класс, confidence)."""
    pick = random.Random(seed).sample(list(images), min(n, len(images)))
    res = model.predict(source=[str(p) for p in pick], conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    arrays = [r.plot()[:, :, ::-1] for r in res]  # BGR -> RGB
    _grid(arrays, [f"{Path(p).name} | предсказано: {len(r.boxes)}" for p, r in zip(pick, res)], cols, save)
    return pick


def show_failures(model, records, names, conf=0.25, iou=0.7, imgsz=640, save=None):
    """Для каждой записи: слева разметка, справа предсказание модели."""
    arrays, titles = [], []
    for rec in records:
        r = model.predict(source=rec["image"], conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        arrays += [draw_gt_image(rec["image"], names), r.plot()[:, :, ::-1]]
        info = (f"tp={rec['tp']} fn={rec['fn']} fp_bg={rec['fp_bg']} fp_class={rec['fp_class']} "
                f"fp_loc={rec['fp_loc']} fp_dup={rec['fp_dup']}")
        titles += [f"РАЗМЕТКА: {Path(rec['image']).name}", f"ПРЕДСКАЗАНИЕ ({info})"]
    _grid(arrays, titles, cols=2, save=save, figsize_per=(6, 4.5))


def plot_curves(run_dirs, save=None):
    """Кривые обучения по results.csv: суммарные потери, mAP@0.50, precision/recall по эпохам.
    run_dirs: {имя_запуска: папка_запуска}."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    for name, d in run_dirs.items():
        df = pd.read_csv(Path(d) / "results.csv")
        df.columns = [c.strip() for c in df.columns]
        tr = [c for c in df.columns if c.startswith("train/") and "loss" in c]
        va = [c for c in df.columns if c.startswith("val/") and "loss" in c]
        ax[0].plot(df["epoch"], df[tr].sum(axis=1), label=name)
        ax[1].plot(df["epoch"], df[va].sum(axis=1), label=name)
        ax[2].plot(df["epoch"], df["metrics/mAP50(B)"], label=name)
        ax[3].plot(df["epoch"], df["metrics/precision(B)"], label=f"{name} P")
        ax[3].plot(df["epoch"], df["metrics/recall(B)"], "--", label=f"{name} R")
    for a, t in zip(ax, ["train loss (сумма)", "val loss (сумма)", "val mAP@0.50", "val precision / recall"]):
        a.set_title(t)
        a.set_xlabel("эпоха")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()
