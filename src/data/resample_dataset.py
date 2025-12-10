import os
import re
import shutil
import random
import uuid
import argparse
from pathlib import Path
from collections import defaultdict, Counter


def get_args():
    parser = argparse.ArgumentParser(description="Leakage-Free Resampler (Keep One Side)")
    parser.add_argument("--data-root", type=str, required=True, help="Path to current dataset")
    parser.add_argument("--output-root", type=str, required=True, help="Path to save cleaned dataset")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    # Precision 1 = ~11km chunks for visualization
    parser.add_argument("--precision", type=int, default=1)
    return parser.parse_args()


def extract_coords(filename):
    matches = re.findall(r"(-?\d+\.\d+)", filename)
    if len(matches) >= 2:
        return float(matches[-2]), float(matches[-1])
    return None


def main():
    args = get_args()
    random.seed(args.seed)
    src_path = Path(args.data_root)
    dest_path = Path(args.output_root)

    if dest_path.exists():
        shutil.rmtree(dest_path)

    print(f"Scanning data (Grouping by {args.precision} decimal places)...")

    # 1. Gather all images
    # We need to track them by EXACT coordinate first to resolve conflicts
    coords_to_images = defaultdict(list)

    for root, _, files in os.walk(src_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = Path(root) / file
                label = full_path.parent.name
                if label not in ["damage", "no_damage"]:
                    continue

                coords = extract_coords(file)
                if coords:
                    item = {"path": full_path, "label": label, "coords": coords}
                    coords_to_images[coords].append(item)

    print(f"Found {len(coords_to_images)} unique coordinates.")

    # 2. Strict De-duplication (Always Keep Exactly One)

    resolved_images = []

    for coords, items in coords_to_images.items():
        # Always pick exactly one image per coordinate
        # handles both conflicting labels AND same-label duplicates.
        winner = random.choice(items)
        resolved_images.append(winner)

    print(f"Total unique images retained: {len(resolved_images)}")

    # 3. Group by Spatial Block (for Split Assignment)
    blocks = defaultdict(list)
    for item in resolved_images:
        lat, lon = item["coords"]
        # 11km blocking for visualization
        key = f"{round(lat, args.precision)}_{round(lon, args.precision)}"
        blocks[key].append(item)

    # 4. Split
    block_keys = list(blocks.keys())
    random.shuffle(block_keys)

    n_blocks = len(block_keys)
    n_val = int(n_blocks * args.val_split)
    n_test = int(n_blocks * args.test_split)

    # Safety for small datasets
    if n_val == 0 and n_blocks > 2:
        n_val = 1
    if n_test == 0 and n_blocks > 2:
        n_test = 1

    splits = {"test": block_keys[:n_test], "validation": block_keys[n_test : n_test + n_val], "train": block_keys[n_test + n_val :]}

    # 5. Write & Rename
    print("\nWriting dataset...")
    final_counts = Counter()
    split_coords_registry = defaultdict(set)

    for split_name, keys in splits.items():
        for key in keys:
            for item in blocks[key]:
                unique_id = uuid.uuid4().hex[:6]
                lat, lon = item["coords"]
                new_filename = f"{split_name}_{item['label']}_{lat}_{lon}_{unique_id}{item['path'].suffix}"

                dest = dest_path / split_name / item["label"] / new_filename
                dest.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(item["path"], dest)
                final_counts[f"{split_name}/{item['label']}"] += 1
                split_coords_registry[split_name].add(item["coords"])

    # 6. Verification
    print("\n--- STATS ---")
    for k, v in sorted(final_counts.items()):
        print(f"{k}: {v}")

    print(f"\nSaved to: {dest_path}")


if __name__ == "__main__":
    main()
