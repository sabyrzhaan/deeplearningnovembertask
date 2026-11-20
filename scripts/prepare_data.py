"""Скачивание датасета, проверка разметки, поиск дубликатов между выборками,
запись configs/data.yaml и results/dataset_report.json.

Запуск:  python scripts/prepare_data.py
Свой датасет в YOLO-формате:  python scripts/prepare_data.py --root /path/to/ds --names a,b,c
(ожидается структура root/images/{train,val,test} и root/labels/{train,val,test})
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import label_path, list_images, read_yolo_labels  # noqa: E402

SPLITS = ["train", "val", "test"]
RANK = {"train": 0, "val": 1, "test": 2}  # при утечке удаляем картинку из выборки с МЕНЬШИМ рангом


def dhash_bits(path, size=8):
    """dHash: картинка -> 64-битный «отпечаток» по градиентам яркости.
    Похожие картинки дают близкие отпечатки (малое расстояние Хэмминга)."""
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).reshape(-1)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def analyze_split(images, names):
    """Статистика и проверка разметки для одной выборки."""
    per_class = Counter()
    areas = []
    problems = Counter()
    examples = []
    n_empty = n_missing = 0
    for img in images:
        rows, issues = read_yolo_labels(label_path(img))
        if "missing_label_file" in issues:
            n_missing += 1
            issues = []
        elif not rows and not issues:
            n_empty += 1
        for it in issues:
            problems["malformed_line"] += 1
            examples.append(f"{Path(img).name}: {it}")
        seen = set()
        for (c, x, y, w, h) in rows:
            if not (0 <= c < len(names)):
                problems["class_out_of_range"] += 1
                examples.append(f"{Path(img).name}: класс {c} вне диапазона")
                continue
            per_class[names[c]] += 1
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                problems["coords_out_of_0_1"] += 1
                examples.append(f"{Path(img).name}: координаты вне [0,1]: {(x, y, w, h)}")
            if w <= 0 or h <= 0:
                problems["zero_size_box"] += 1
                examples.append(f"{Path(img).name}: нулевой размер бокса")
            if w * h < 1e-4:
                problems["tiny_box_lt_0.01pct_area"] += 1
            key = (c, round(x, 4), round(y, 4), round(w, 4), round(h, 4))
            if key in seen:
                problems["duplicate_box_in_file"] += 1
                examples.append(f"{Path(img).name}: повторяющийся бокс")
            seen.add(key)
            areas.append(w * h)
    return {
        "images": len(images),
        "objects": int(sum(per_class.values())),
        "per_class": {n: int(per_class.get(n, 0)) for n in names},
        "background_images_empty_label": n_empty,
        "images_without_label_file": n_missing,
        "median_box_area_fraction": float(np.median(areas)) if areas else None,
        "label_problems": dict(problems),
        "problem_examples": examples[:20],
    }


def find_cross_split_leaks(split_images, thr):
    """Ищем (почти) одинаковые картинки в РАЗНЫХ выборках: расстояние Хэмминга dHash <= thr."""
    items = [(s, p) for s in SPLITS for p in split_images.get(s, [])]
    if len(items) < 2:
        return [], 0
    H = np.stack([dhash_bits(p) for _, p in items]).astype(np.float32)
    D = H @ (1 - H).T + (1 - H) @ H.T  # матрица расстояний Хэмминга
    iu = np.triu_indices(len(items), k=1)
    mask = D[iu] <= thr
    pairs, same_split = [], 0
    for i, j in zip(iu[0][mask], iu[1][mask]):
        if items[i][0] == items[j][0]:
            same_split += 1  # внутри одной выборки это допустимо
        else:
            pairs.append((items[i][0], str(items[i][1]), items[j][0], str(items[j][1]), int(D[i, j])))
    return pairs, same_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="african-wildlife.yaml", help="встроенный yaml Ultralytics")
    ap.add_argument("--root", default=None, help="корень готового датасета (пропустить скачивание)")
    ap.add_argument("--names", default=None, help="имена классов через запятую (вместе с --root)")
    ap.add_argument("--hash-thr", type=int, default=4, help="порог Хэмминга dHash для 'почти дубликата'")
    ap.add_argument("--no-dedup", action="store_true", help="не удалять утечки, только отчитаться")
    ap.add_argument("--out-yaml", default="configs/data.yaml")
    ap.add_argument("--out-report", default="results/dataset_report.json")
    args = ap.parse_args()

    if args.root:
        root = Path(args.root)
        names = args.names.split(",")
        split_dirs = {s: str(root / "images" / s) for s in SPLITS}
    else:
        from ultralytics.data.utils import check_det_dataset  # скачает ~100 МБ при первом запуске

        info = check_det_dataset(args.data)
        root = Path(info["path"])
        names = [info["names"][i] for i in sorted(info["names"])]
        split_dirs = {s: info[s] for s in SPLITS if info.get(s)}

    split_images = {s: list_images(d) for s, d in split_dirs.items()}
    print("Классы:", names)

    raw = {s: analyze_split(imgs, names) for s, imgs in split_images.items()}

    # --- дубликаты между выборками: точные (md5) и почти-дубликаты (dHash) ---
    leaks, same_split = find_cross_split_leaks(split_images, args.hash_thr)
    md5_map = {}
    for s in SPLITS:
        for p in split_images.get(s, []):
            md5_map.setdefault(md5(p), set()).add(s)
    exact_cross = sum(1 for v in md5_map.values() if len(v) > 1)

    dropped = {s: set() for s in SPLITS}
    if not args.no_dedup:
        for (sa, pa, sb, pb, d) in leaks:
            if RANK[sa] < RANK[sb]:
                dropped[sa].add(pa)
            else:
                dropped[sb].add(pb)

    out_dir = Path(args.out_yaml).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"path": str(root)}
    final_images = {}
    for s in split_images:
        keep = [p for p in split_images[s] if str(p) not in dropped[s]]
        final_images[s] = keep
        if dropped[s]:  # Ultralytics умеет читать список путей из .txt
            txt = out_dir / f"{s}_dedup.txt"
            txt.write_text("\n".join(str(p) for p in keep) + "\n")
            cfg[s] = str(txt.resolve())
        else:
            cfg[s] = str(split_dirs[s])
    cfg["names"] = {i: n for i, n in enumerate(names)}
    Path(args.out_yaml).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    final = {s: analyze_split(imgs, names) for s, imgs in final_images.items()}
    report = {
        "dataset": args.data if not args.root else str(root),
        "classes": names,
        "annotation_format": "YOLO txt: class x_center y_center width height (нормировано 0..1)",
        "split_method": "готовое разбиение датасета (не менялось); затем удаление утечек (см. leaks_cross_split)",
        "near_duplicate_hash": {"algorithm": "dHash 8x8", "hamming_threshold": args.hash_thr},
        "exact_duplicates_across_splits": exact_cross,
        "near_duplicate_pairs_same_split_allowed": same_split,
        "leaks_cross_split_count": len(leaks),
        "leaks_cross_split_examples": leaks[:50],
        "removed_images": {s: len(v) for s, v in dropped.items()},
        "splits_raw": raw,
        "splits_final": final,
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\nПары-утечки между выборками: {len(leaks)} | точные дубликаты между выборками: {exact_cross}")
    print("Удалено из выборок:", report["removed_images"])
    print("\n| split | images | objects | " + " | ".join(names) + " | empty | label problems |")
    for s, st in final.items():
        pc = " | ".join(str(st["per_class"][n]) for n in names)
        print(f"| {s} | {st['images']} | {st['objects']} | {pc} | "
              f"{st['background_images_empty_label']} | {sum(st['label_problems'].values())} |")
    print(f"\nЗаписано: {args.out_yaml}, {args.out_report}")


if __name__ == "__main__":
    main()
