#!/usr/bin/env python3
"""

The raw dataset holds before/after images of the same buildings (damage/no_damage)
across different splits. That means identical locations (same lon/lat in the
filename) can appear in train, validation, and test simultaneously, sometimes
with conflicting labels. This leaks information between splits and causes the
validation curves to swing depending on whether the model is seeing an already
memorized building. Grouping every coordinate into exactly one split fixes that
leakage while keeping the before/after pairs intact inside the split. The script
also merges coordinates that are very close together (configurable distance and
decimal rounding) so near-duplicate tiles do not straddle splits.

What it does:
* Scans data_root/{train,validation,test}/{class} for image files.
* Groups files by coordinate key derived from the filename stem (lon_lat).
* Assigns each coordinate group to a single split, aiming to preserve the
  original split sizes as much as possible.
* Writes a new folder tree (output_root/{split}/{class}) using hardlinks by
  default (or copies if --copy is set) so the original data remains untouched.

Usage examples:
    python src/utils/build_coordinate_split.py --data-root data --output-root data_resampled --dry-run
    python src/utils/build_coordinate_split.py --data-root data --output-root data_resampled

After running, point the datamodule to the new root to train/evaluate without cross-split leakage.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CoordKey = str
SplitName = str
LabelName = str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group dataset splits by coordinate to avoid leakage.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Existing dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_resampled"),
        help="Destination root for the grouped, leakage-free splits.",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpeg", ".jpg", ".png"],
        help="Image extensions to include.",
    )
    parser.add_argument(
        "--coord-round",
        type=int,
        default=3,
        help="Decimal places to round lon/lat when deriving coordinate keys (coarser keeps nearby images together).",
    )
    parser.add_argument(
        "--merge-distance-m",
        type=float,
        default=150.0,
        help="Merge coordinate groups whose centers are within this many meters (0 to disable).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for group assignment.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned assignment and stats; do not write files.",
    )
    return parser.parse_args()


def coord_key_from_path(path: Path, coord_round: int) -> Tuple[CoordKey, Tuple[float, float] | None]:
    """
    Extract the coordinate key from a filename.
    Parse lon/lat and round to reduce float-noise variants mapping to different keys.
    Returns the key and the raw lon/lat if parsing succeeds.
    """
    stem = path.stem
    try:
        lon_str, lat_str = stem.split("_", 1)
        lon = round(float(lon_str), 6)
        lat = round(float(lat_str), 6)
        lon_key = round(lon, coord_round)
        lat_key = round(lat, coord_round)
        key = f"{lon_key:.{coord_round}f}_{lat_key:.{coord_round}f}"
        return key, (lon, lat)
    except Exception:
        # Fallback: keep original stem so we do not drop the sample if parsing fails.
        return stem, None


def collect_groups(data_root: Path, exts: Iterable[str], coord_round: int) -> Tuple[
    Dict[CoordKey, Dict],
    Counter,
    Counter,
    List[SplitName],
    List[LabelName],
    int,
    int,
    Dict[CoordKey, Tuple[float, float] | None],
]:
    groups: Dict[CoordKey, Dict] = {}
    split_counts: Counter = Counter()
    class_counts: Counter = Counter()
    split_names: List[SplitName] = []
    label_names: List[LabelName] = []
    conflict_coords = 0
    multisplit_coords = 0
    coord_lookup: Dict[CoordKey, Tuple[float, float] | None] = {}

    exts_lower = {e.lower() for e in exts}

    for split_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        split = split_dir.name
        split_names.append(split)
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = label_dir.name
            if label not in label_names:
                label_names.append(label)
            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() not in exts_lower:
                    continue
                key, lonlat = coord_key_from_path(img_path, coord_round)
                info = groups.setdefault(key, {"items": [], "labels": set(), "splits": set(), "coord": None})
                if lonlat is not None:
                    coord_lookup.setdefault(key, lonlat)
                    if info["coord"] is None:
                        info["coord"] = lonlat
                info["items"].append((img_path, label, split))
                info["labels"].add(label)
                info["splits"].add(split)
                split_counts[split] += 1
                class_counts[label] += 1

    for info in groups.values():
        if len(info["labels"]) > 1:
            conflict_coords += 1
        if len(info["splits"]) > 1:
            multisplit_coords += 1

    split_names.sort()
    label_names.sort()
    return groups, split_counts, class_counts, split_names, label_names, conflict_coords, multisplit_coords, coord_lookup


def summarize_groups(groups: Dict[CoordKey, Dict]) -> Tuple[int, int]:
    conflict_coords = 0
    multisplit_coords = 0
    for info in groups.values():
        if len(info["labels"]) > 1:
            conflict_coords += 1
        if len(info["splits"]) > 1:
            multisplit_coords += 1
    return conflict_coords, multisplit_coords


def compute_targets(split_counts: Counter) -> Dict[SplitName, int]:
    """
    Preserve the original split sizes as the target allocation for grouped data.
    If a split was empty, it will not receive any groups.
    """
    return {split: count for split, count in split_counts.items() if count > 0}


def merge_close_groups(groups: Dict[CoordKey, Dict], coord_lookup: Dict[CoordKey, Tuple[float, float]], threshold_m: float) -> Dict[CoordKey, Dict]:
    """
    Merge coordinate groups whose centers fall within threshold_m meters.
    This is conservative against leakage from nearby (not just identical) coordinates.
    """
    if threshold_m <= 0 or not coord_lookup:
        return groups

    import math
    from collections import defaultdict

    def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        # Standard haversine distance in meters.
        R = 6_371_000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = phi2 - phi1
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    # Roughly convert meters to degrees (lat-based; lon will be tighter at higher latitudes).
    cell_size = threshold_m / 111_320.0
    grid: Dict[Tuple[int, int], List[CoordKey]] = defaultdict(list)
    for key, (lon, lat) in coord_lookup.items():
        cell = (int(lon // cell_size), int(lat // cell_size))
        grid[cell].append(key)

    parent: Dict[CoordKey, CoordKey] = {k: k for k in groups.keys()}

    def find(x: CoordKey) -> CoordKey:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: CoordKey, b: CoordKey) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Keep the larger group as root to reduce tree depth.
        if len(groups[ra]["items"]) < len(groups[rb]["items"]):
            ra, rb = rb, ra
        parent[rb] = ra

    neighbors = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for cell, keys in grid.items():
        for key in keys:
            lon1, lat1 = coord_lookup[key]
            for dx, dy in neighbors:
                nk_cell = (cell[0] + dx, cell[1] + dy)
                for other in grid.get(nk_cell, []):
                    if key >= other:
                        # Enforce an ordering to avoid duplicate checks.
                        continue
                    lon2, lat2 = coord_lookup[other]
                    if haversine_m(lon1, lat1, lon2, lat2) <= threshold_m:
                        union(key, other)

    merged: Dict[CoordKey, Dict] = {}
    for key, info in groups.items():
        root = find(key)
        out = merged.setdefault(
            root,
            {
                "items": [],
                "labels": set(),
                "splits": set(),
                "coord": groups[root].get("coord") or coord_lookup.get(root),
            },
        )
        out["items"].extend(info["items"])
        out["labels"].update(info["labels"])
        out["splits"].update(info["splits"])
        if out["coord"] is None and info.get("coord") is not None:
            out["coord"] = info["coord"]
    return merged


def assign_groups(groups: Dict[CoordKey, Dict], targets: Dict[SplitName, int], seed: int) -> Tuple[Dict[CoordKey, SplitName], Counter]:
    """
    Assign each coordinate group to one split, keeping sizes close to the original.
    We shuffle for randomness (but deterministically with seed) and greedily fill
    the split that is furthest from its target.
    """
    coords = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(coords)

    assigned: Dict[CoordKey, SplitName] = {}
    current: Counter = Counter()

    def split_score(split: SplitName) -> float:
        target = targets.get(split, 0)
        if target == 0:
            return float("inf")
        return current[split] / target

    valid_splits = list(targets.keys())
    if not valid_splits:
        raise RuntimeError("No non-empty splits detected in source data.")

    for coord in coords:
        # Pick the least-filled split relative to its target.
        split = min(valid_splits, key=split_score)
        count = len(groups[coord]["items"])
        assigned[coord] = split
        current[split] += count

    return assigned, current


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_destination(dest_dir: Path, src_name: str) -> Path:
    """
    Avoid collisions when multiple identical filenames land in the same class.
    """
    dest = dest_dir / src_name
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    idx = 1
    while True:
        candidate = dest_dir / f"{stem}__dup{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def materialize(
    groups: Dict[CoordKey, Dict],
    assignments: Dict[CoordKey, SplitName],
    output_root: Path,
) -> Counter:
    """
    Write the grouped dataset to disk using copies (no hardlinks to avoid surprises).
    Destination filenames are rewritten to the normalized coord key to make the
    renaming explicit and avoid keeping noisy float precision in stems.
    """
    counts = Counter()
    per_coord_counter: Dict[Tuple[str, str, str], int] = {}
    for coord, info in groups.items():
        split = assignments[coord]
        for src, label, _orig_split in info["items"]:
            dest_dir = output_root / split / label
            ensure_dir(dest_dir)
            key = (split, coord, label)
            idx = per_coord_counter.get(key, 0)
            per_coord_counter[key] = idx + 1
            suffix = src.suffix
            # First file for a coord uses bare coord; subsequent get an index suffix.
            dest_name = f"{coord}{suffix}" if idx == 0 else f"{coord}_{idx:04d}{suffix}"
            dest_path = dest_dir / dest_name
            # Ensure uniqueness in case of unexpected collisions.
            while dest_path.exists():
                idx += 1
                per_coord_counter[key] = idx + 1
                dest_name = f"{coord}_{idx:04d}{suffix}"
                dest_path = dest_dir / dest_name
            shutil.copy2(src, dest_path)
            counts[split] += 1
    return counts


def main() -> None:
    args = parse_args()

    groups, split_counts, class_counts, split_names, label_names, conflict_coords, multisplit_coords, coord_lookup = collect_groups(args.data_root, args.exts, args.coord_round)

    print(f"Found {len(groups)} unique coordinates across splits (rounded to {args.coord_round} decimals).")
    print(f"Class counts: {dict(class_counts)}")
    print(f"Original split sizes: {dict(split_counts)}")
    print(f"Coords with conflicting labels: {conflict_coords}")
    print(f"Coords appearing in multiple splits: {multisplit_coords}")

    if args.merge_distance_m > 0:
        groups = merge_close_groups(groups, {k: v for k, v in coord_lookup.items() if v is not None}, args.merge_distance_m)
        conflict_coords, multisplit_coords = summarize_groups(groups)
        print(
            f"After merging coords within {args.merge_distance_m} meters: {len(groups)} unique coordinates; "
            f"conflicting-label coords={conflict_coords}, multisplit coords={multisplit_coords}"
        )

    targets = compute_targets(split_counts)
    assignments, filled_counts = assign_groups(groups, targets, args.seed)
    print(f"Target split sizes (images): {targets}")
    print(f"Planned grouped sizes: {dict(filled_counts)}")

    if args.dry_run:
        print("Dry run requested; not writing any files.")
        return

    # Clean output_root before writing to avoid mixing old/new assignments (prevents leakage via leftovers).
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)
    realized = materialize(groups, assignments, args.output_root)
    print(f"Wrote grouped dataset to: {args.output_root}")
    print(f"Final split sizes: {dict(realized)}")
    print("Tip: point TrainConfig.data_root to this new directory to train without leakage.")


if __name__ == "__main__":
    main()
