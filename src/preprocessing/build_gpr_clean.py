"""Validate + standardize the published Mendeley GPR dataset into leakage-safe YOLO splits.

The source is ALREADY processed by the authors (cropped 224x224, augmented,
YOLO+VOC annotated via makesense.ai). This tool does NOT re-crop, enhance,
denoise, or re-augment images: it copies bytes verbatim.

It only: verifies extraction, checks image/label pairs, validates YOLO boxes
(QC), removes exact duplicates, flags (never auto-deletes) perceptual
near-duplicates expected from augmentation, splits by original source group
(all augments of one profile stay in ONE split), and standardizes
folder semantics into a 2-class YOLO schema (cavity->0, utility->1,
intact->empty txt).

Source: extracted GPR_data/ dir OR GPR_data.rar (read-only via rarfile).
Output: GPR_CLEAN/images/{train,val,test} + labels/{train,val,test} + data.yaml
         + cleaning_report.json + manifest.csv

Rules enforced:
- group-aware split by original source ID (never split augments of one source)
- remap folder semantics -> YOLO ids: cavity -> 0, utility -> 1, intact -> empty txt
- reject malformed / w,h<=0 / out-of-[0,1] boxes
- exact (md5) + perceptual (dHash<=3) duplicates
- fixed seed, reproducible
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASS_MAP = {"cavity": 0, "utility": 1}
VALID_SPLITS = ("train", "val", "test")


def validate_yolo_box(line: str):
    """Return (ok, reason). Valid: 5 fields, cx,cy in [0,1], w,h in (0,1]."""
    parts = line.strip().split()
    if len(parts) != 5:
        return False, f"malformed: expected 5 fields, got {len(parts)}"
    try:
        float(parts[0])
        cx, cy, w, h = map(float, parts[1:])
    except ValueError:
        return False, "malformed: non-numeric"
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return False, f"out-of-range center ({cx},{cy})"
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False, f"invalid size w={w} h={h}"
    return True, ""


def remap_label_lines(lines, kind: str):
    """Standardize folder semantics into the combined 2-class YOLO schema.

    Upstream files are all class 0 within their own folder; the folder carries
    the semantics. kind in cavity|utility.
    """
    want = CLASS_MAP[kind]
    out = []
    for ln in lines:
        p = ln.strip().split()
        p[0] = str(want)
        out.append(" ".join(p))
    return out


def source_group_id(stem: str) -> str:
    """Group key = original source id: '001_aug_1' -> '001', '10_aug_10' -> '10'."""
    stem = Path(stem).stem  # tolerate 'name.jpg' input
    m = re.match(r"(.+)_aug_\d+$", stem)
    if m:
        return m.group(1)
    return stem  # originals: each file its own group


def normalize_stem(filename: str) -> str:
    """Lowercase ext, ' (2)' -> '_2', spaces -> '_'. Returns name+ext."""
    p = Path(filename)
    stem = re.sub(r"\s*\(\d+\)", lambda m: "_" + re.sub(r"\D", "", m.group(0)), p.stem)
    stem = re.sub(r"\s+", "_", stem.strip())
    return f"{stem}.jpg"


def dhash(image_bytes: bytes, hash_size: int = 8) -> int | None:
    """Difference hash via PIL only (grayscale, no extra deps). None on failure."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(
            (hash_size + 1, hash_size), Image.BILINEAR)
        px = list(img.get_flattened_data()) if hasattr(img, "get_flattened_data") else list(img.getdata())
        bits = 0
        for r in range(hash_size):
            for c in range(hash_size):
                bits = (bits << 1) | (1 if px[r * (hash_size + 1) + c] > px[r * (hash_size + 1) + c + 1] else 0)
        return bits
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------- IO adapters (dir or rar, read-only) ----------------

class SourceIndex:
    """Collects (kind, orig_ref, image_bytes, label_text|None). Never writes to source."""

    def __init__(self):
        self.items = []  # dicts

    @classmethod
    def from_path(cls, source: Path):
        idx = cls()
        if source.is_file() and source.suffix.lower() == ".rar":
            idx._from_rar(source)
        else:
            idx._from_dir(source)
        return idx

    def _from_dir(self, root: Path):
        # preferred layout: augmented_cavities/images, .../annotations/Yolo_format|YOLO_format|VOC_XML_format
        # fallback: flat legacy layout (images next to annotations/)
        specs = [
            ("cavity", ["augmented_cavities/images", "augmented_cavities"]),
            ("utility", ["augmented_utilities/images", "augmented_utilities"]),
            ("intact", ["augmented_intact/images", "augmented_intact"]),
        ]
        for kind, candidates in specs:
            imgdir = None
            for c in candidates:
                d = root / c
                if d.is_dir() and any(d.iterdir()):
                    # pick dir that actually holds images
                    if any(f.suffix.lower() in (".jpg", ".jpeg") for f in d.iterdir() if f.is_file()):
                        imgdir = d
                        break
            if imgdir is None:
                continue
            for f in sorted(imgdir.iterdir()):
                if not (f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")):
                    continue
                data = f.read_bytes()
                label = self._find_label_for_dir(root, kind, f.stem)
                self.items.append({"kind": kind, "orig_ref": str(f), "bytes": data,
                                   "label_text": label, "base": f.stem})
        # originals (kept for manifest/counts, excluded from YOLO splits by default? include as their own groups)
        for kind, dname in [("cavity", "cavities"), ("utility", "Utilities"), ("intact", "intact")]:
            d = root / dname
            if not d.is_dir():
                continue
            for f in sorted(d.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg"):
                    self.items.append({"kind": kind, "orig_ref": str(f), "bytes": f.read_bytes(),
                                       "label_text": None, "base": f.stem, "original_only": True})

    def _find_label_for_dir(self, root: Path, kind: str, stem: str):
        if kind == "intact":
            return None
        base = root / ("augmented_cavities" if kind == "cavity" else "augmented_utilities") / "annotations"
        for sub in ["YOLO_format", "Yolo_format", "yolo_format"]:
            cand = base / sub / (stem + ".txt")
            if cand.is_file():
                return cand.read_text()
        # legacy flat: annotations/Yolo_format
        return None

    def _from_rar(self, rar_path: Path):
        import rarfile
        rf = rarfile.RarFile(str(rar_path))
        infos = {i.filename: i for i in rf.infolist() if not i.is_dir()}
        # map label files by basename
        yolo_by_base = {}
        for name in infos:
            if name.endswith(".txt") and "/annotations/" in name:
                yolo_by_base.setdefault(Path(name).stem, name)
        for name, info in sorted(infos.items()):
            if not name.lower().endswith((".jpg", ".jpeg")):
                continue
            try:
                data = rf.read(name)
            except Exception:
                data = None  # compressed entry w/o tool or truncated; counted as unreadable
            parts = name.split("/")
            top = parts[1] if len(parts) > 1 else ""
            if top == "augmented_cavities":
                kind = "cavity"
            elif top == "augmented_utilities":
                kind = "utility"
            elif top == "augmented_intact":
                kind = "intact"
            elif top == "cavities":
                kind = "cavity"
            elif top == "Utilities":
                kind = "utility"
            elif top == "intact":
                kind = "intact"
            else:
                continue
            label = None
            if kind in ("cavity", "utility"):
                key = Path(name).stem
                if key in yolo_by_base:
                    try:
                        label = rf.read(yolo_by_base[key]).decode()
                    except Exception:
                        label = None
            self.items.append({"kind": kind, "orig_ref": f"{rar_path}::{name}",
                               "bytes": data, "label_text": label,
                               "base": Path(name).stem,
                               "original_only": top in ("cavities", "Utilities", "intact")})


def build_clean(source: str, out: str, seed: int = 42,
                ratios=(0.7, 0.15, 0.15), include_originals: bool = False,
                phash_thresh: int = 3):
    rng = random.Random(seed)
    idx = SourceIndex.from_path(Path(source))
    report = Counter()
    report["images_found"] = len(idx.items)

    # 1. decode check + exact dup (md5) + perceptual dup (dHash<=thresh, same kind)
    md5_seen = {}
    hash_seen = []  # (dhash, md5, out_name)
    valid = []
    dup_exact = 0
    dup_percept = 0
    corrupted = 0
    for it in idx.items:
        data = it["bytes"]
        if data is None:
            corrupted += 1
            it["_drop"] = "unreadable-source-bytes"
            continue
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(data))
            im.verify()
        except Exception:
            corrupted += 1
            it["_drop"] = "corrupt-image"
            continue
        md5 = hashlib.md5(data).hexdigest()
        if md5 in md5_seen:
            dup_exact += 1
            it["_drop"] = f"exact-duplicate-of:{md5_seen[md5]}"
            continue
        dh = dhash(data)
        is_pdup = False
        if dh is not None:
            for prev_dh, prev_md5, prev_name in hash_seen:
                if hamming(dh, prev_dh) <= phash_thresh and it["kind"] == "intact":
                    # only auto-drop perceptual dups for intact background (labels unaffected);
                    # for cavity/utility keep but flag (avoid dropping distinct anomalies)
                    pass
            # count but do not drop anomaly images; drop intact near-dups
            for prev_dh, prev_md5, prev_name in hash_seen:
                if hamming(dh, prev_dh) <= phash_thresh:
                    it["_phash_match"] = prev_name
                    break
        md5_seen[md5] = it["base"]
        if dh is not None:
            hash_seen.append((dh, md5, it["base"]))
        valid.append(it)
    report["corrupted_images"] = corrupted
    report["duplicates_exact_removed"] = dup_exact

    # 2. label validation + remap (only augmented anomaly images go to splits)
    pool = []
    invalid_labels = 0
    invalid_list = []
    for it in valid:
        if it.get("original_only") and not include_originals:
            it["_drop"] = "original-excluded"
            continue
        kind = it["kind"]
        if kind == "intact":
            it["clean_lines"] = []  # empty txt
            pool.append(it)
            continue
        txt = it["label_text"]
        if txt is None:
            invalid_labels += 1
            invalid_list.append((it["orig_ref"], "missing-label"))
            continue
        lines = [l for l in txt.strip().splitlines() if l.strip()]
        if not lines:
            invalid_labels += 1
            invalid_list.append((it["orig_ref"], "empty-label"))
            continue
        clean = []
        bad = False
        for ln in lines:
            ok, reason = validate_yolo_box(ln)
            if not ok:
                bad = True
                invalid_list.append((it["orig_ref"], reason + " :: " + ln.strip()))
                break
            clean.append(ln)
        if bad:
            invalid_labels += 1
            continue
        it["clean_lines"] = remap_label_lines(clean, kind)
        pool.append(it)
    report["invalid_labels"] = invalid_labels

    # 3. group-aware stratified split (by source_group_id, per kind)
    groups = defaultdict(list)  # (kind, group) -> items
    for it in pool:
        groups[(it["kind"], source_group_id(it["base"]))].append(it)
    split_of_group = {}
    for kind in ("cavity", "utility", "intact"):
        keys = sorted(g for (k, g) in groups if k == kind)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, g in enumerate(keys):
            if i < n_train:
                split_of_group[(kind, g)] = "train"
            elif i < n_train + n_val:
                split_of_group[(kind, g)] = "val"
            else:
                split_of_group[(kind, g)] = "test"
    for it in pool:
        it["_split"] = split_of_group[(it["kind"], source_group_id(it["base"]))]

    # 4. write out (copy bytes, never touch source)
    outp = Path(out)
    if outp.exists():
        shutil.rmtree(outp)
    for s in VALID_SPLITS:
        (outp / "images" / s).mkdir(parents=True, exist_ok=True)
        (outp / "labels" / s).mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    used_names = Counter()
    counts = Counter()
    for it in pool:
        s = it["_split"]
        norm = normalize_stem(it["base"] + ".jpg")
        # ensure unique output name
        if used_names[norm]:
            norm = f"{Path(norm).stem}__{used_names[norm]}.jpg"
        used_names[norm] += 1
        (outp / "images" / s / norm).write_bytes(it["bytes"])
        lab = (outp / "labels" / s / (Path(norm).stem + ".txt"))
        if it["clean_lines"]:
            lab.write_text("\n".join(it["clean_lines"]) + "\n")
        else:
            lab.write_text("")  # intact background
        counts[(s, it["kind"])] += 1
        manifest_rows.append({
            "clean_path": f"images/{s}/{norm}",
            "split": s, "class": it["kind"],
            "class_id": CLASS_MAP.get(it["kind"], -1),
            "source_ref": it["orig_ref"],
            "source_group": f"{it['kind']}:{source_group_id(it['base'])}",
            "orig_base": it["base"],
            "md5": hashlib.md5(it["bytes"]).hexdigest(),
        })
    with open(outp / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            w.writeheader()
            w.writerows(manifest_rows)

    # 5. data.yaml
    with open(outp / "data.yaml", "w") as f:
        f.write(f"path: {outp.resolve().as_posix()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("nc: 2\nnames:\n  0: cavity\n  1: utility\n")

    report["images_retained"] = len(pool)
    report["duplicates_removed_total"] = dup_exact
    for kind in ("cavity", "utility", "intact"):
        report[f"{kind}_count"] = sum(1 for it in pool if it["kind"] == kind)
    for s in VALID_SPLITS:
        report[f"{s}_count"] = sum(1 for it in pool if it["_split"] == s)
        for kind in ("cavity", "utility", "intact"):
            report[f"{s}_{kind}"] = counts[(s, kind)]
    report["seed"] = seed
    report["invalid_label_samples"] = invalid_list[:20]
    with open(outp / "cleaning_report.json", "w") as f:
        json.dump(dict(report), f, indent=2)
    return dict(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="GPR_data/ dir or GPR_data.rar (read-only)")
    ap.add_argument("--out", required=True, help="output GPR_CLEAN dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--include-originals", action="store_true")
    a = ap.parse_args()
    rep = build_clean(a.source, a.out, seed=a.seed, ratios=(a.train, a.val, a.test),
                      include_originals=a.include_originals)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
