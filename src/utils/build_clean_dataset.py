#!/usr/bin/env python3
"""
Build a leakage-free, deduplicated dataset split from the raw /data tree.

- Pre/Post pairs at identical coords (paper §3.2) put opposite labels on near-identical backgrounds.
- Random image-level split (paper §3.5) lets one view of a coord land in train and its pair in val/test.
- Spatial duplicates across captures (paper §3.3) inflate metrics.

What this script does:
1) Scan data_root/{train,validation,test}/{class} for images.
2) Drop byte-identical duplicates (keeps first occurrence).
3) Group by normalized coordinates (configurable rounding).
4) Optionally merge coordinate groups within a distance threshold (meters) to catch near-duplicates.
5) Drop any coordinate groups that contain conflicting labels (damage vs. no_damage).
6) Assign each remaining coordinate group to exactly one split, preserving the original split size ratios.
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
from typing import Dict, Iterable, List, Tuple

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
    parser.add_argument("--no-drop-conflicts", action="store_true", help="Keep coords with conflicting labels (defaults to dropping them).")
    parser.add_argument("--dry-run", action="store_true", help="Only print stats; do not write output.")
    return parser.parse_args()


def coord_key_from_path(path: Path, coord_round: int) -> Tuple[CoordKey, Tuple[float, float] | None]:
    stem = path.stem
    try:
        lon_str, lat_str = stem.split("_", 1)
        lon = float(lon_str)
        lat = float(lat_str)
        lon_key = round(lon, coord_round)
        lat_key = round(lat, coord_round)
        key = f"{lon_key:.{coord_round}f}_{lat_key:.{coord_round}f}"
        return key, (lon, lat)
    except Exception:
        return stem, None


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


def assign_groups(groups: Dict[CoordKey, Dict], targets: Dict[SplitName, int], seed: int) -> Tuple[Dict[CoordKey, SplitName], Counter]:
    coords = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(coords)
    assigned: Dict[CoordKey, SplitName] = {}
    current: Counter = Counter()
    valid_splits = list(targets.keys())
    if not valid_splits:
        raise RuntimeError("No non-empty splits detected in source data.")

    def score(split: SplitName) -> float:
        tgt = targets.get(split, 0)
        if tgt == 0:
            return float("inf")
        return current[split] / tgt

    for coord in coords:
        split = min(valid_splits, key=score)
        count = len(groups[coord]["items"])
        assigned[coord] = split
        current[split] += count
    return assigned, current


def materialize(groups: Dict[CoordKey, Dict], assignments: Dict[CoordKey, SplitName], output_root: Path) -> Counter:
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
            dest_name = f"{coord}{suffix}" if idx == 0 else f"{coord}_{idx:04d}{suffix}"
            dest_path = dest_dir / dest_name
            while dest_path.exists():
                idx += 1
                per_coord_counter[key] = idx + 1
                dest_name = f"{coord}_{idx:04d}{suffix}"
                dest_path = dest_dir / dest_name
            shutil.copy2(img["path"], dest_path)
            counts[split] += 1
    return counts


def main() -> None:
    args = parse_args()
    drop_conflicts = not args.no_drop_conflicts

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

    # Targets from original split proportions
    split_counts = Counter(img["split"] for img in images)
    total = sum(split_counts.values())
    targets = {s: int(split_counts[s] / total * sum(len(v["items"]) for v in groups.values())) for s in split_counts}
    assignments, filled = assign_groups(groups, targets, args.seed)
    print(f"Target sizes: {targets}")
    print(f"Planned sizes: {dict(filled)}")

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
