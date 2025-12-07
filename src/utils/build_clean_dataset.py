#!/usr/bin/env python3
"""
Build a leakage-free, deduplicated, class-balanced dataset split from the raw /data tree.

- Pre/Post pairs at identical coords (paper §3.2) put opposite labels on near-identical backgrounds.
- Random image-level split (paper §3.5) lets one view of a coord land in train and its pair in val/test.
- Spatial duplicates across captures (paper §3.3) inflate metrics.

What this script does:
1) Scan data_root/{train,validation,test}/{class} for images.
2) Drop byte-identical duplicates (keeps first occurrence).
3) Group by normalized coordinates (configurable rounding).
4) Optionally merge coordinate groups within a distance threshold (meters) to catch near-duplicates.
5) Drop any coordinate groups that contain conflicting labels (damage vs. no_damage).
6) Assign each remaining coordinate group to exactly one split, preserving original split ratios *and* approximately balancing classes per split.
7) Copy images to a clean output_root/{split}/{class} tree with normalized filenames.

Usage:
    python src/utils/build_clean_dataset.py --data-root data --output-root data_clean
    python src/utils/build_clean_dataset.py --data-root data --output-root data_clean \\
        --coord-round 3 --merge-distance-m 50 --no-drop-conflicts
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, DefaultDict, Set

CoordKey = str
SplitName = str
LabelName = str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean, leakage-free dataset split.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Existing dataset root.")
    parser.add_argument("--output-root", type=Path, default=Path("data_clean"), help="Destination root for the cleaned dataset.")
    parser.add_argument("--exts", nargs="+", default=[".jpeg", ".jpg", ".png"], help="Image extensions to include.")
    parser.add_argument("--coord-round", type=int, default=3, help="Decimal places to round lon/lat for coord keys.")
    parser.add_argument("--merge-distance-m", type=float, default=50.0, help="Merge coord groups whose centers are within this distance (meters).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split assignment.")
    parser.add_argument("--drop-conflicts", action="store_true", help="Drop coords with conflicting labels (default: keep them in a single split).")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Desired share of samples in train split (will be normalized with val/test).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Desired share of samples in validation split (will be normalized with train/test).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Desired share of samples in test split (will be normalized with train/val).",
    )
    parser.add_argument(
        "--balance-multiplier",
        type=float,
        default=0.0,
        help="Downsample majority classes toward (minority_total * multiplier). Set 0 or <0 to disable downsampling.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print stats; do not write output.")
    return parser.parse_args()


def coord_key_from_path(path: Path, coord_round: int) -> Tuple[CoordKey, Tuple[float, float] | None]:
    import re

    stem = path.stem
    match = re.search(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)", stem)
    if not match:
        return stem, None
    lon = round(float(match.group(1)), 6)
    lat = round(float(match.group(2)), 6)
    lon_key = round(lon, coord_round)
    lat_key = round(lat, coord_round)
    key = f"{lon_key:.{coord_round}f}_{lat_key:.{coord_round}f}"
    return key, (lon, lat)


def hash_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_images(data_root: Path, exts: Iterable[str], coord_round: int):
    exts_lower = {e.lower() for e in exts}
    images = []
    hash_seen = {}
    hash_dups = 0
    for split_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        split = split_dir.name
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = label_dir.name
            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() not in exts_lower:
                    continue
                h = hash_file(img_path)
                if h in hash_seen:
                    hash_dups += 1
                    continue
                hash_seen[h] = img_path
                key, lonlat = coord_key_from_path(img_path, coord_round)
                images.append(
                    {
                        "split": split,
                        "label": label,
                        "path": img_path,
                        "coord_key": key,
                        "lonlat": lonlat,
                    }
                )
    return images, hash_dups


def group_by_coord(images: List[Dict]) -> Dict[CoordKey, Dict]:
    groups: Dict[CoordKey, Dict] = {}
    for img in images:
        key = img["coord_key"]
        info = groups.setdefault(key, {"items": [], "labels": set(), "splits": set(), "lonlat": img["lonlat"]})
        info["items"].append(img)
        info["labels"].add(img["label"])
        info["splits"].add(img["split"])
        if info["lonlat"] is None and img["lonlat"] is not None:
            info["lonlat"] = img["lonlat"]
    return groups


def primary_label(info: Dict) -> LabelName:
    counts = Counter(i["label"] for i in info["items"])
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def merge_close_groups(groups: Dict[CoordKey, Dict], threshold_m: float) -> Dict[CoordKey, Dict]:
    if threshold_m <= 0:
        return groups
    # Prepare coordinates
    coords = {k: v["lonlat"] for k, v in groups.items() if v["lonlat"] is not None}
    if not coords:
        return groups
    cell_size = threshold_m / 111_320.0
    grid: Dict[Tuple[int, int], List[CoordKey]] = {}
    for key, (lon, lat) in coords.items():
        cell = (int(lon / cell_size), int(lat / cell_size))
        grid.setdefault(cell, []).append(key)

    parent: Dict[CoordKey, CoordKey] = {k: k for k in groups}

    def find(x: CoordKey) -> CoordKey:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: CoordKey, b: CoordKey) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if len(groups[ra]["items"]) < len(groups[rb]["items"]):
            ra, rb = rb, ra
        parent[rb] = ra

    neighbors = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for cell, keys in grid.items():
        for key in keys:
            lon1, lat1 = coords[key]
            for dx, dy in neighbors:
                nk_cell = (cell[0] + dx, cell[1] + dy)
                for other in grid.get(nk_cell, []):
                    if key >= other:
                        continue
                    lon2, lat2 = coords[other]
                    d = haversine_m(lon1, lat1, lon2, lat2)
                    if d <= threshold_m:
                        union(key, other)

    merged: Dict[CoordKey, Dict] = {}
    for key, info in groups.items():
        root = find(key)
        out = merged.setdefault(root, {"items": [], "labels": set(), "splits": set(), "lonlat": info["lonlat"]})
        out["items"].extend(info["items"])
        out["labels"].update(info["labels"])
        out["splits"].update(info["splits"])
        if out["lonlat"] is None and info["lonlat"] is not None:
            out["lonlat"] = info["lonlat"]
    return merged


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _allocate_targets(total: int, split_ratios: Dict[SplitName, float]) -> Dict[SplitName, int]:
    """
    Allocate integer targets per split given split ratios, making the sum exact.
    """
    if total == 0:
        return {s: 0 for s in split_ratios}
    raw = {s: total * r for s, r in split_ratios.items()}
    allocated = {s: int(math.floor(v)) for s, v in raw.items()}
    remainder = total - sum(allocated.values())
    if remainder != 0:
        # Distribute remainder to splits with largest fractional parts.
        frac_order = sorted(split_ratios.keys(), key=lambda s: raw[s] - math.floor(raw[s]), reverse=True)
        idx = 0
        step = 1 if remainder > 0 else -1
        while remainder != 0:
            s = frac_order[idx % len(frac_order)]
            # Avoid negative allocations
            if allocated[s] + step >= 0:
                allocated[s] += step
                remainder -= step
            idx += 1
    return allocated


def downsample_majority(groups: Dict[CoordKey, Dict], balance_multiplier: float, seed: int) -> Dict[CoordKey, Dict]:
    """
    Downsample majority labels to be closer to the minority label count.
    balance_multiplier allows keeping more than exact parity (e.g., 1.5x minority).
    """
    if balance_multiplier <= 0:
        print("Downsampling disabled (balance_multiplier <= 0).")
        return groups
    label_totals: Counter = Counter()
    for info in groups.values():
        for item in info["items"]:
            label_totals[item["label"]] += 1
    if not label_totals:
        return groups
    minority = min(label_totals.values())
    target_per_label = {lbl: min(cnt, int(math.floor(minority * balance_multiplier))) for lbl, cnt in label_totals.items()}

    rng = random.Random(seed)
    keep_keys: Set[CoordKey] = set()
    for lbl, target in target_per_label.items():
        # Collect groups by primary label
        lbl_groups = [(k, v) for k, v in groups.items() if primary_label(v) == lbl]
        rng.shuffle(lbl_groups)
        total = 0
        for key, info in lbl_groups:
            count = len(info["items"])
            if target == 0 or total >= target:
                continue
            if total + count > target and len(lbl_groups) > 1:
                # Skip this group to avoid overshooting badly; may underfill slightly.
                continue
            keep_keys.add(key)
            total += count
    # If a label was underfilled because of coarse group sizes, keep smallest remaining groups until target met.
    for lbl, target in target_per_label.items():
        lbl_groups = [(k, v) for k, v in groups.items() if primary_label(v) == lbl and k not in keep_keys]
        lbl_groups.sort(key=lambda kv: len(kv[1]["items"]))
        total = sum(len(groups[k]["items"]) for k in keep_keys if primary_label(groups[k]) == lbl)
        for key, info in lbl_groups:
            if total >= target:
                break
            keep_keys.add(key)
            total += len(info["items"])

    if not keep_keys:
        return groups
    pruned = {k: v for k, v in groups.items() if k in keep_keys}
    before = sum(len(v["items"]) for v in groups.values())
    after = sum(len(v["items"]) for v in pruned.values())
    print(f"Downsampled majority: {before} -> {after} images. Targets per label: {target_per_label}")
    return pruned


def assign_groups_balanced(
    groups: Dict[CoordKey, Dict],
    targets_total: Dict[SplitName, int],
    targets_per_label: Dict[LabelName, Dict[SplitName, int]],
    seed: int,
) -> Tuple[Dict[CoordKey, SplitName], Counter, Dict[LabelName, Counter]]:
    coords = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(coords)
    assigned: Dict[CoordKey, SplitName] = {}
    current_total: Counter = Counter()
    current_label: Dict[LabelName, Counter] = {lbl: Counter() for lbl in targets_per_label}
    valid_splits = list(targets_total.keys())
    if not valid_splits:
        raise RuntimeError("No non-empty splits detected in source data.")

    for coord in coords:
        info = groups[coord]
        label = primary_label(info)
        label_targets = targets_per_label.get(label, {s: 0 for s in valid_splits})
        has_label_targets = any(v > 0 for v in label_targets.values())

        def score(split: SplitName) -> Tuple[float, float]:
            lt = label_targets.get(split, 0)
            if has_label_targets and lt > 0:
                lbl_score = current_label.get(label, Counter())[split] / lt
            elif has_label_targets and lt == 0:
                lbl_score = float("inf")
            else:
                lbl_score = 0.0
            tt = targets_total.get(split, 0)
            tot_score = current_total[split] / tt if tt > 0 else float("inf")
            return (lbl_score, tot_score)

        split = min(valid_splits, key=score)
        count = len(info["items"])
        assigned[coord] = split
        current_total[split] += count
        if label not in current_label:
            current_label[label] = Counter()
        current_label[label][split] += count
    return assigned, current_total, current_label


def materialize(groups: Dict[CoordKey, Dict], assignments: Dict[CoordKey, SplitName], output_root: Path) -> Counter:
    """
    Copy images to destination with coord+orig-stem filenames to retain metadata.
    """
    counts = Counter()
    per_coord_counter: Dict[Tuple[str, str], int] = {}
    for coord, info in groups.items():
        split = assignments[coord]
        for img in info["items"]:
            label = img["label"]
            dest_dir = output_root / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            key = (split, coord)
            idx = per_coord_counter.get(key, 0)
            per_coord_counter[key] = idx + 1
            suffix = img["path"].suffix
            base = f"{coord}__{img['path'].stem}"
            dest_name = f"{base}{suffix}" if idx == 0 else f"{base}_{idx:04d}{suffix}"
            dest_path = dest_dir / dest_name
            while dest_path.exists():
                idx += 1
                per_coord_counter[key] = idx + 1
                dest_name = f"{base}_{idx:04d}{suffix}"
                dest_path = dest_dir / dest_name
            shutil.copy2(img["path"], dest_path)
            counts[split] += 1
    return counts


def main() -> None:
    args = parse_args()
    drop_conflicts = args.drop_conflicts

    images, hash_dups = collect_images(args.data_root, args.exts, args.coord_round)
    print(f"Scanned images: {len(images)} (dropped byte-identical duplicates: {hash_dups})")

    groups = group_by_coord(images)
    conflict_coords = [k for k, v in groups.items() if len(v["labels"]) > 1]
    multisplit_coords = [k for k, v in groups.items() if len(v["splits"]) > 1]
    print(f"Initial coord groups: {len(groups)} | conflicting-label coords: {len(conflict_coords)} | multi-split coords: {len(multisplit_coords)}")

    if drop_conflicts:
        for k in conflict_coords:
            groups.pop(k, None)
        print(f"Dropped {len(conflict_coords)} coords with conflicting labels.")

    if args.merge_distance_m > 0:
        groups = merge_close_groups(groups, args.merge_distance_m)
        conflict_coords = [k for k, v in groups.items() if len(v["labels"]) > 1]
        multisplit_coords = [k for k, v in groups.items() if len(v["splits"]) > 1]
        print(
            f"After merging within {args.merge_distance_m} m: {len(groups)} coords | conflicting-label coords: {len(conflict_coords)} | multi-split coords: {len(multisplit_coords)}"
        )

    # Downsample majority classes toward minority to improve balance.
    groups = downsample_majority(groups, args.balance_multiplier, args.seed)

    # Compute desired totals and per-class targets using provided split ratios.
    ratios_raw = {"train": args.train_ratio, "validation": args.val_ratio, "test": args.test_ratio}
    ratio_sum = sum(ratios_raw.values())
    if ratio_sum <= 0:
        raise RuntimeError("Split ratios must sum to > 0")
    split_ratios = {k: v / ratio_sum for k, v in ratios_raw.items()}

    total_clean = sum(len(v["items"]) for v in groups.values())
    targets_total = _allocate_targets(total_clean, split_ratios)

    label_totals: Counter = Counter()
    for info in groups.values():
        for item in info["items"]:
            label_totals[item["label"]] += 1
    targets_per_label: Dict[LabelName, Dict[SplitName, int]] = {}
    for lbl, lbl_total in label_totals.items():
        targets_per_label[lbl] = _allocate_targets(lbl_total, split_ratios)

    assignments, filled_total, filled_label = assign_groups_balanced(groups, targets_total, targets_per_label, args.seed)
    print(f"Target total sizes: {targets_total}")
    print(f"Planned total sizes: {dict(filled_total)}")
    print("Planned per-class sizes:")
    for lbl, tgt in targets_per_label.items():
        realized = {s: filled_label.get(lbl, Counter()).get(s, 0) for s in split_ratios}
        print(f"  {lbl}: target={tgt}, realized={realized}")

    if args.dry_run:
        print("Dry run requested; not writing output.")
        return

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    realized = materialize(groups, assignments, args.output_root)
    print(f"Wrote cleaned dataset to: {args.output_root}")
    print(f"Final split sizes: {dict(realized)}")


if __name__ == "__main__":
    main()
