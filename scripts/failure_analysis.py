"""Сопоставление предсказаний с разметкой при ФИКСИРОВАННОМ пороге уверенности.

Зачем: официальные P/R из Ultralytics считаются в точке максимума F1 (порог подбирается по самой выборке),
а здесь порог задан заранее (conf=0.25) — это честная «рабочая точка» модели. Плюс разбор типов ошибок:
  tp        — верная детекция (класс совпал, IoU >= порога)
  fp_class  — рамка на объекте, но класс неверный
  fp_loc    — класс верный, но рамка неточная (0.1 <= IoU < порога)
  fp_dup    — повторная рамка на уже найденный объект
  fp_bg     — ложное срабатывание на фоне
  fn        — пропущенный объект
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import label_path, read_yolo_labels  # noqa: E402


def xywhn_to_xyxy(rows, w, h):
    """[(cls, xc, yc, bw, bh)] нормированные -> (классы, боксы xyxy в пикселях)."""
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows, dtype=float)
    cls = a[:, 0].astype(int)
    xc, yc, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return cls, np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], axis=1)


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(gt_cls, gt_xyxy, pr_cls, pr_xyxy, pr_conf, iou_thr=0.5, loc_thr=0.1):
    """Жадное сопоставление: предсказания по убыванию уверенности."""
    n_gt, n_pr = len(gt_cls), len(pr_cls)
    c = dict(tp=0, fp_class=0, fp_loc=0, fp_dup=0, fp_bg=0, fn=0)
    gt_taken = np.zeros(n_gt, bool)
    if n_pr:
        ious = iou_matrix(pr_xyxy, gt_xyxy) if n_gt else np.zeros((n_pr, 0))
        for p in np.argsort(-np.asarray(pr_conf)):
            if n_gt == 0:
                c["fp_bg"] += 1
                continue
            row = ious[p]
            same = gt_cls == pr_cls[p]
            cand = np.where(same & ~gt_taken & (row >= iou_thr), row, -1.0)
            j = int(cand.argmax())
            if cand[j] >= 0:
                gt_taken[j] = True
                c["tp"] += 1
            elif np.any(same & gt_taken & (row >= iou_thr)):
                c["fp_dup"] += 1
            elif np.any(~same & (row >= iou_thr)):
                c["fp_class"] += 1
            elif np.any(same & (row >= loc_thr)):
                c["fp_loc"] += 1
            else:
                c["fp_bg"] += 1
    c["fn"] = int((~gt_taken).sum())
    return c


def evaluate_images(model, images, conf=0.25, nms_iou=0.7, imgsz=640, match_iou=0.5):
    """Прогон модели по списку картинок -> (список записей по картинкам, итоговые метрики)."""
    records = []
    stream = model.predict(source=[str(p) for p in images], conf=conf, iou=nms_iou,
                           imgsz=imgsz, stream=True, verbose=False)
    for r in stream:
        h, w = r.orig_shape
        rows, _ = read_yolo_labels(label_path(r.path))
        gt_cls, gt_xyxy = xywhn_to_xyxy(rows, w, h)
        b = r.boxes
        pr_xyxy = b.xyxy.cpu().numpy()
        pr_cls = b.cls.cpu().numpy().astype(int)
        pr_conf = b.conf.cpu().numpy()
        c = match_image(gt_cls, gt_xyxy, pr_cls, pr_xyxy, pr_conf, iou_thr=match_iou)
        rec = dict(image=str(r.path), n_gt=len(gt_cls), n_pred=len(pr_cls), **c)
        rec["errors"] = c["fn"] + c["fp_class"] + c["fp_loc"] + c["fp_dup"] + c["fp_bg"]
        records.append(rec)
    return records, summarize(records, conf, nms_iou, match_iou)


def summarize(records, conf, nms_iou, match_iou):
    tot = {k: int(sum(r[k] for r in records)) for k in
           ["tp", "fp_class", "fp_loc", "fp_dup", "fp_bg", "fn", "n_gt", "n_pred"]}
    fp = tot["fp_class"] + tot["fp_loc"] + tot["fp_dup"] + tot["fp_bg"]
    prec = tot["tp"] / (tot["tp"] + fp) if (tot["tp"] + fp) else 0.0
    rec = tot["tp"] / (tot["tp"] + tot["fn"]) if (tot["tp"] + tot["fn"]) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, conf=conf, nms_iou=nms_iou,
                match_iou=match_iou, **tot)


def worst_images(records, k=3):
    """k картинок с наибольшим числом ошибок (при равенстве — где больше объектов)."""
    return sorted(records, key=lambda r: (r["errors"], r["n_gt"]), reverse=True)[:k]


if __name__ == "__main__":  # мини-самопроверка логики сопоставления (без модели)
    gt_c = np.array([0, 1])
    gt_b = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], float)
    pr_c = np.array([0, 0, 2, 1])
    pr_b = np.array([[2, 2, 98, 98], [3, 3, 97, 97], [500, 500, 550, 550], [200, 200, 300, 300]], float)
    pr_s = np.array([0.9, 0.8, 0.7, 0.6])
    print(match_image(gt_c, gt_b, pr_c, pr_b, pr_s))
