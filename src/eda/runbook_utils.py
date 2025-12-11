from __future__ import annotations

import hashlib
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import art3d
from PIL import Image, ImageEnhance, ImageOps
from matplotlib.ticker import MaxNLocator

from IPython.display import display

try:
    import geopandas as gpd
    from shapely.geometry import box
    import matplotlib.patheffects as pe
except ImportError:
    gpd = None
    box = None
    pe = None


@dataclass
class RunbookContext:
    project_root: Path
    plots_dir: Path
    include_splits: List[str]
    roundings: List[int]
    merge_distance_m: float
    stat_sample_size: int | None
    size_sample_size: int | None
    split_stat_sample_size: int | None
    duplicate_sample: int | None
    quality_sample_size: int | None
    photo_sample_size: int | None
    grid_samples: int
    seed: int
    label_palette: Dict[str, str]


# ---------- Generic helpers ----------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(ctx: RunbookContext, fig, dataset_tag: str, name: str) -> Path:
    ensure_dir(ctx.plots_dir)
    path = ctx.plots_dir / f"eda_{dataset_tag}_{name}.pdf"
    fig.savefig(path, bbox_inches="tight", format="pdf")
    print(f"Saved figure to {path.relative_to(ctx.project_root)}")
    return path


def run_shell(cmd: List[str], cwd: Path) -> None:
    print("Running:", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)
    result.check_returncode()


def coord_from_stem(stem: str):
    match = re.search(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)", stem)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def coord_key(lon: float, lat: float, rounding: int) -> Tuple[float, float]:
    return round(lon, rounding), round(lat, rounding)


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_image_dataframe(ctx: RunbookContext, data_dir: Path) -> pd.DataFrame:
    rows = []
    exts = (".jpeg", ".jpg", ".png")
    for split in ctx.include_splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = label_dir.name
            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() not in exts:
                    continue
                lon, lat = coord_from_stem(img_path.stem)
                rows.append(
                    {
                        "split": split,
                        "label": label,
                        "path": img_path,
                        "filename": img_path.name,
                        "stem": img_path.stem,
                        "lon": lon,
                        "lat": lat,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["split", "label", "filename"]).reset_index(drop=True)


# ---------- Plotting helpers ----------


def plot_class_balance(ctx: RunbookContext, df: pd.DataFrame, dataset_tag: str):
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0).assign(total=lambda d: d.sum(axis=1))
    counts.loc["all"] = counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts_no_total = counts.drop(index="all") if "all" in counts.index else counts
    counts_no_total[["damage", "no_damage"]].plot(
        kind="bar",
        ax=axes[0],
        stacked=False,
        color=[ctx.label_palette["damage"], ctx.label_palette["no_damage"]],
    )
    axes[0].set_title("Image count by split and class")
    axes[0].set_ylabel("Images [count]")
    axes[0].set_xlabel("Split")
    axes[0].legend(title="Label")

    ratios = (counts_no_total["damage"] / (counts_no_total["damage"] + counts_no_total["no_damage"])).rename("damage_ratio")
    ratios.plot(kind="bar", ax=axes[1], color="#56b4e9")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Damage class share per split")
    axes[1].set_ylabel("Damage share [fraction]")
    axes[1].set_xlabel("Split")
    axes[1].axhline(ratios.mean(), color="gray", linestyle="--", label="Mean share")
    axes[1].legend()
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "class_counts_ratios")
    plt.show()
    return counts


def plot_spatial_scatter(ctx: RunbookContext, valid_coords: pd.DataFrame, dataset_tag: str):
    if valid_coords.empty:
        print("No coordinates to plot.")
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for label, color in [("damage", ctx.label_palette["damage"]), ("no_damage", ctx.label_palette["no_damage"])]:
        subset = valid_coords[valid_coords["label"] == label]
        ax.scatter(subset["lon"], subset["lat"], s=10, alpha=0.8, label=label, c=color, edgecolors="none")
    ax.legend(title="Label", loc="upper right")
    ax.set_title("Spatial distribution encoded in filenames")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "lon_lat_scatter")
    plt.show()


def plot_spatial_density(ctx: RunbookContext, valid_coords: pd.DataFrame, dataset_tag: str):
    if valid_coords.empty:
        print("No coordinates to plot density.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, label in zip(axes, ["damage", "no_damage"]):
        subset = valid_coords[valid_coords["label"] == label]
        if subset.empty:
            ax.set_title(f"No data for {label}")
            continue
        ax.scatter(subset["lon"], subset["lat"], s=4, alpha=0.15, color=ctx.label_palette[label], label="points")
        sns.kdeplot(data=subset, x="lon", y="lat", fill=True, levels=10, alpha=0.3, color=ctx.label_palette[label], ax=ax)
        ax.set_title(f"Spatial density: {label}")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "spatial_density_kde")
    plt.show()


def plot_map(ctx: RunbookContext, valid_coords: pd.DataFrame, dataset_tag: str):
    if gpd is None or box is None:
        print("Geopandas/shapely not available; map skipped.")
        return
    if valid_coords.empty:
        print("No coordinates to map.")
        return
    try:
        url_land = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
        url_states = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
        url_cities = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
        land = gpd.read_file(url_land)
        states = gpd.read_file(url_states)
        cities = gpd.read_file(url_cities)

        usa_states = states[states["adm0_a3"] == "USA"]
        major_cities = cities[cities["SCALERANK"] < 8]

        gdf = gpd.GeoDataFrame(
            valid_coords[["split", "label"]],
            geometry=gpd.points_from_xy(valid_coords["lon"], valid_coords["lat"]),
            crs="EPSG:4326",
        )
        minx, miny, maxx, maxy = gdf.total_bounds
        fig_w, fig_h = 12, 7
        aspect_ratio = fig_w / fig_h
        pad_y = 0.2
        height_span = (maxy - miny) + (pad_y * 2)
        width_span = height_span * aspect_ratio
        ymid = (miny + maxy) / 2
        xmid = (minx + maxx) / 2
        xlims = (xmid - width_span / 2, xmid + width_span / 2)
        ylims = (ymid - height_span / 2, ymid + height_span / 2)
        view_box = box(xlims[0], ylims[0], xlims[1], ylims[1])

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_facecolor("#e0f3f8")

        land_clip = gpd.clip(land, view_box)
        land_clip.plot(ax=ax, color="#f9f9f9", edgecolor="#b0b0b0", linewidth=0.3)

        states_clip = gpd.clip(usa_states, view_box)
        states_clip.boundary.plot(ax=ax, color="#888888", linewidth=0.7, linestyle="-")

        cities_clip = gpd.clip(major_cities, view_box)
        cities_clip.plot(ax=ax, color="black", markersize=10, alpha=0.6, zorder=3)
        for x, y, label in zip(cities_clip.geometry.x, cities_clip.geometry.y, cities_clip["NAME"]):
            txt = ax.text(x, y + 0.03, label, fontsize=10, ha="center", va="bottom", color="#222222", zorder=4)
            if pe:
                txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])

        gdf[gdf["label"] == "damage"].plot(ax=ax, markersize=25, color=ctx.label_palette["damage"], alpha=0.9, label="Damage", edgecolor="white", linewidth=0.6, zorder=5)
        gdf[gdf["label"] == "no_damage"].plot(ax=ax, markersize=25, color=ctx.label_palette["no_damage"], alpha=0.8, label="No Damage", edgecolor="white", linewidth=0.6, zorder=5)

        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_h = []
        uniq_l = []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq_h.append(h)
                uniq_l.append(l)
                seen.add(l)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.grid(True, linestyle="--", alpha=0.4, color="#555555")
        ax.set_title("Spatial distribution on map (EPSG:4326)", fontsize=16, pad=15, weight="bold")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        leg = ax.legend(uniq_h, uniq_l, loc="upper right", frameon=True, framealpha=0.9, fancybox=False, edgecolor="black", title="Label")
        leg.get_frame().set_linewidth(0.5)
        plt.tight_layout()
        save_fig(ctx, fig, dataset_tag, "map_lon_lat")
        plt.show()
    except Exception as e:
        print(f"Error rendering map: {e}")


def coord_overlap_by_rounding(valid_coords: pd.DataFrame, roundings: List[int]):
    results = {}
    for r in roundings:
        buckets = defaultdict(set)
        for _, row in valid_coords.iterrows():
            buckets[coord_key(row["lon"], row["lat"], r)].add(row["split"])
        overlaps = sum(1 for v in buckets.values() if len(v) > 1)
        results[r] = {"coord_groups": len(buckets), "cross_split_coords": overlaps}
    return results


def near_neighbor_pairs(valid_coords: pd.DataFrame, distance_m: float):
    if distance_m <= 0 or valid_coords.empty:
        return []
    cell_size = distance_m / 111_320.0
    coords = valid_coords.reset_index(drop=True)
    grid = defaultdict(list)
    for idx, row in coords.iterrows():
        cell = (int(row["lon"] / cell_size), int(row["lat"] / cell_size))
        grid[cell].append(idx)
    pairs = []
    for idx, row in coords.iterrows():
        cell = (int(row["lon"] / cell_size), int(row["lat"] / cell_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cell[0] + dx, cell[1] + dy), []):
                    if j <= idx:
                        continue
                    other = coords.iloc[j]
                    if row["split"] == other["split"]:
                        continue
                    d = haversine_m(row["lon"], row["lat"], other["lon"], other["lat"])
                    if d <= distance_m:
                        pairs.append((row, other, d))
    return pairs


def plot_coord_conflicts(ctx: RunbookContext, valid_coords: pd.DataFrame, dataset_tag: str):
    coord_counts = (
        valid_coords.groupby(["lon", "lat"])
        .agg(
            count=("label", "size"),
            labels=("label", lambda x: set(x)),
            splits=("split", lambda x: set(x)),
            paths=("path", lambda x: list(x)),
        )
        .reset_index()
    )
    conflict_coords = coord_counts[coord_counts["labels"].apply(lambda s: len(s) > 1)]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(valid_coords["lon"], valid_coords["lat"], s=8, alpha=0.12, color="#bbbbbb", label="Unique coords")
    conflict_mask = valid_coords[["lon", "lat"]].apply(tuple, axis=1).isin(set(zip(conflict_coords["lon"], conflict_coords["lat"])))
    conflict_points = valid_coords[conflict_mask]
    ax.scatter(conflict_points["lon"], conflict_points["lat"], s=18, alpha=0.9, color="#d55e00", label="Conflicting labels")
    ax.set_title("Coordinates with multiple labels highlighted")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.legend()
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "coords_conflicts")
    plt.show()

    return coord_counts, conflict_coords


def plot_images_per_coord_hist(ctx: RunbookContext, coord_counts: pd.DataFrame, dataset_tag: str):
    if coord_counts.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(coord_counts["count"], bins=30, color="#029e73", alpha=0.8)
    ax.set_title("Redundancy: images per coordinate")
    ax.set_xlabel("Images per coordinate")
    ax.set_ylabel("Count of coordinates")
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "images_per_coord_hist")
    plt.show()


def plot_coord_overlap_heatmap(ctx: RunbookContext, valid_coords: pd.DataFrame, dataset_tag: str):
    coord_sets = {s: set(zip(df["lon"], df["lat"])) for s, df in valid_coords.groupby("split")}
    splits = sorted(coord_sets.keys())
    if not splits:
        return
    mat = pd.DataFrame(index=splits, columns=splits, dtype=float)
    for a in splits:
        for b in splits:
            mat.loc[a, b] = len(coord_sets[a] & coord_sets[b])
    mat = mat.fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(mat, annot=True, fmt="d", cmap="crest", ax=ax)
    ax.set_title("Coord overlap across splits (exact match)")
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "coord_overlap_heatmap")
    plt.show()


def summarize_coord_splits(valid_coords: pd.DataFrame):
    coord_map = defaultdict(list)
    for _, row in valid_coords.iterrows():
        coord_map[(row["lon"], row["lat"])].append(row)
    coord_summaries = []
    for coord, rows in coord_map.items():
        splits = {r["split"] for r in rows}
        labels = {r["label"] for r in rows}
        coord_summaries.append({"coord": coord, "splits": splits, "labels": labels, "count": len(rows), "rows": rows})
    return coord_map, coord_summaries


def plot_multisplit_bar(ctx: RunbookContext, coord_summaries: List[Dict], dataset_tag: str):
    if not coord_summaries:
        return
    split_counts = Counter()
    for c in coord_summaries:
        split_counts[len(c["splits"])] += 1
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = sorted(split_counts.keys())
    ax.bar([str(k) for k in keys], [split_counts[k] for k in keys], color="#cc78bc")
    ax.set_xlabel("Number of splits a coordinate appears in")
    ax.set_ylabel("Coordinates [count]")
    ax.set_title("Cross-split leakage from coord reuse")
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "coord_multisplit_counts")
    plt.show()


def plot_split_pair_counts(ctx: RunbookContext, coord_summaries: List[Dict], dataset_tag: str):
    pair_counts = Counter()
    for c in coord_summaries:
        splits = sorted(list(c["splits"]))
        if len(splits) <= 1:
            continue
        for a, b in combinations(splits, 2):
            pair_counts[(a, b)] += 1
    if not pair_counts:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{a}->{b}" for (a, b) in pair_counts.keys()]
    ax.bar(labels, list(pair_counts.values()), color="#fbafe4")
    ax.set_ylabel("Coordinates [count]")
    ax.set_title("Same coordinate landing in multiple splits (exact match)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "coord_split_pair_counts")
    plt.show()


def sample_same_coord_grids(ctx: RunbookContext, coord_counts: pd.DataFrame, valid_coords: pd.DataFrame, dataset_tag: str, sample_n: int = 3, images_per_coord: int = 6):
    dup_coords = coord_counts[coord_counts["count"] > 1]
    if dup_coords.empty:
        print("No duplicated coordinates to plot.")
        return
    sample_coords = dup_coords.sample(min(sample_n, len(dup_coords)), random_state=ctx.seed)
    for idx, row in sample_coords.iterrows():
        paths = row["paths"][:images_per_coord]
        coord = (row["lon"], row["lat"])
        cols = min(3, len(paths))
        rows = math.ceil(len(paths) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, path in zip(axes, paths):
            with Image.open(path) as img:
                ax.imshow(img)
            info = valid_coords[valid_coords["path"] == path].iloc[0]
            ax.set_title(f"{info['label']} | {info['split']},{Path(path).name}", fontsize=9)
            ax.axis("off")
        for ax in axes[len(paths) :]:
            ax.axis("off")
        fig.suptitle(f"Images at coord lon={coord[0]:.4f}, lat={coord[1]:.4f}", fontsize=12)
        plt.tight_layout()
        save_fig(ctx, fig, dataset_tag, f"same_coord_{idx}")
        plt.show()


def cross_split_hash_overlap(ctx: RunbookContext, df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    sample = df if ctx.duplicate_sample is None else df.sample(min(ctx.duplicate_sample, len(df)), random_state=ctx.seed)
    hash_map = defaultdict(list)
    for _, row in sample.iterrows():
        digest = hashlib.md5(Path(row["path"]).read_bytes()).hexdigest()
        hash_map[digest].append(row)
    overlap_rows = []
    for h, rows in hash_map.items():
        splits = {r["split"] for r in rows}
        if len(splits) > 1:
            for r in rows:
                overlap_rows.append({**r, "hash": h})
    return pd.DataFrame(overlap_rows)


def collect_image_sizes(ctx: RunbookContext, df: pd.DataFrame) -> pd.DataFrame:
    sample = df if ctx.size_sample_size is None else df.sample(min(ctx.size_sample_size, len(df)), random_state=ctx.seed)
    records = []
    for path in sample["path"]:
        with Image.open(path) as img:
            w, h = img.size
            records.append({"path": path, "width": w, "height": h, "area": w * h, "split": path.parent.parent.name, "label": path.parent.name})
    return pd.DataFrame(records)


def plot_size_distributions(ctx: RunbookContext, size_df: pd.DataFrame, dataset_tag: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(data=size_df, x="width", ax=axes[0], color="#56b4e9")
    axes[0].set_title("Width frequency")
    axes[0].set_xlabel("Width [px]")
    axes[0].set_ylabel("Images [count]")
    axes[0].tick_params(axis="x", rotation=45)

    sns.countplot(data=size_df, x="height", ax=axes[1], color="#de8f05")
    axes[1].set_title("Height frequency")
    axes[1].set_xlabel("Height [px]")
    axes[1].set_ylabel("Images [count]")
    axes[1].tick_params(axis="x", rotation=45)

    sns.histplot(data=size_df, x="area", bins=20, ax=axes[2], color="#029e73")
    axes[2].set_title("Pixel area distribution")
    axes[2].set_xlabel("Area [px^2]")
    axes[2].set_ylabel("Images [count]")
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "size_distributions")
    plt.show()


def compute_channel_stats(ctx: RunbookContext, df: pd.DataFrame, sample_size: int | None = None) -> Dict[str, np.ndarray]:
    sample = df if sample_size is None else df.sample(min(sample_size, len(df)), random_state=ctx.seed)
    totals = np.zeros(3, dtype=np.float64)
    totals_sq = np.zeros(3, dtype=np.float64)
    counts = 0
    hist = np.zeros((3, 256), dtype=np.int64)

    for path in sample["path"]:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        pixels = arr.reshape(-1, 3)
        totals += pixels.sum(axis=0)
        totals_sq += (pixels.astype(np.float64) ** 2).sum(axis=0)
        counts += pixels.shape[0]
        for c in range(3):
            hist[c] += np.bincount(pixels[:, c], minlength=256)

    means = totals / counts
    stds = np.sqrt(totals_sq / counts - means**2)
    return {"means": means, "stds": stds, "hist": hist, "counts": counts, "sample_size": len(sample)}


def plot_channel_histograms(ctx: RunbookContext, channel_stats: Dict[str, np.ndarray], dataset_tag: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = ["#ff0000", "#2ca02c", "#0173b2"]
    labels = ["Red", "Green", "Blue"]
    for c, ax, label in zip(range(3), axes, labels):
        hist = channel_stats["hist"][c]
        ax.plot(range(256), hist / hist.sum(), color=colors[c])
        ax.set_title(f"{label} histogram (normalized)")
        ax.set_xlabel("Pixel value [0-255]")
    axes[0].set_ylabel("Density [fraction]")
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "channel_histograms")
    plt.show()


def brightness_contrast(ctx: RunbookContext, df: pd.DataFrame):
    sample = df if ctx.photo_sample_size is None else df.sample(min(ctx.photo_sample_size, len(df)), random_state=ctx.seed)
    records = []
    for _, row in sample.iterrows():
        with Image.open(row["path"]) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32)
        records.append({"split": row["split"], "label": row["label"], "brightness": gray.mean(), "contrast": gray.std()})
    return pd.DataFrame(records)


def plot_brightness_contrast(ctx: RunbookContext, photo_stats: pd.DataFrame, dataset_tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=photo_stats, x="split", y="brightness", palette="colorblind", ax=axes[0], hue="split")
    axes[0].set_title("Brightness by split")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Brightness [0-255]")

    sns.boxplot(data=photo_stats, x="split", y="contrast", palette="colorblind", ax=axes[1], hue="split")
    axes[1].set_title("Contrast by split")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Contrast [0-255]")
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "brightness_contrast_box")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.kdeplot(data=photo_stats, x="brightness", hue="split", fill=True, common_norm=False, palette="colorblind", ax=axes[0])
    axes[0].set_title("Brightness density by split")
    axes[0].set_xlabel("Brightness [0-255]")
    axes[0].set_ylabel("Density [fraction]")

    sns.kdeplot(data=photo_stats, x="contrast", hue="split", fill=True, common_norm=False, palette="colorblind", ax=axes[1])
    axes[1].set_title("Contrast density by split")
    axes[1].set_xlabel("Contrast [0-255]")
    axes[1].set_ylabel("Density [fraction]")
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "brightness_contrast_density")
    plt.show()


def plot_image_grid(ctx: RunbookContext, df: pd.DataFrame, dataset_tag: str, split: str, label: str):
    subset = df[(df["split"] == split) & (df["label"] == label)]
    if subset.empty:
        print(f"No images for {split}/{label}")
        return
    sample = subset.sample(min(ctx.grid_samples, len(subset)), random_state=ctx.seed)
    cols = 3
    rows = math.ceil(len(sample) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for ax, (_, row) in zip(axes, sample.iterrows()):
        with Image.open(row["path"]) as img:
            ax.imshow(img)
        ax.set_title(f"{split}/{label},{row['filename']}", fontsize=9)
        ax.axis("off")
    for ax in axes[len(sample) :]:
        ax.axis("off")
    fig.suptitle(f"Random {label} samples from {split}", fontsize=14)
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, f"grid_{split}_{label}")
    plt.show()


def check_corrupted(ctx: RunbookContext, df: pd.DataFrame):
    sample = df if ctx.quality_sample_size is None else df.sample(min(ctx.quality_sample_size, len(df)), random_state=ctx.seed)
    bad = []
    for path in sample["path"]:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            bad.append((path, str(e)))
    return bad


def find_duplicates(ctx: RunbookContext, df: pd.DataFrame):
    sample = df if ctx.duplicate_sample is None else df.sample(min(ctx.duplicate_sample, len(df)), random_state=ctx.seed)
    hashes = {}
    dups = []
    for path in sample["path"]:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in hashes:
            dups.append((hashes[digest], path))
        else:
            hashes[digest] = path
    return dups


def plot_duplicate_example(ctx: RunbookContext, dups, dataset_tag: str):
    if not dups:
        print("No duplicate pairs in sample.")
        return
    a, b = dups[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(Image.open(a))
    axes[0].set_title(a.name, fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(Image.open(b))
    axes[1].set_title(b.name, fontsize=9)
    axes[1].axis("off")
    fig.suptitle("Duplicate check", fontsize=14)
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "duplicate_check")
    plt.show()


def apply_random_aug(img: Image.Image, seed: int | None = None) -> Image.Image:
    params = {"max_rotate_deg": 20, "hflip_p": 0.5, "brightness": (0.8, 1.2), "contrast": (0.85, 1.15), "saturation": (0.85, 1.15)}
    rng_local = random.Random(seed)
    out = img
    if rng_local.random() < params["hflip_p"]:
        out = ImageOps.mirror(out)
    angle = rng_local.uniform(-params["max_rotate_deg"], params["max_rotate_deg"])
    out = out.rotate(angle, resample=Image.BILINEAR)
    b_min, b_max = params["brightness"]
    c_min, c_max = params["contrast"]
    s_min, s_max = params["saturation"]
    out = ImageEnhance.Brightness(out).enhance(rng_local.uniform(b_min, b_max))
    out = ImageEnhance.Contrast(out).enhance(rng_local.uniform(c_min, c_max))
    out = ImageEnhance.Color(out).enhance(rng_local.uniform(s_min, s_max))
    return out


def plot_augmentation_preview(ctx: RunbookContext, df: pd.DataFrame, dataset_tag: str):
    if df.empty:
        return
    sample_path = df.iloc[0]["path"]
    with Image.open(sample_path) as base_img:
        base_img = base_img.convert("RGB")
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(base_img)
        axes[0].set_title("Original")
        for i in range(1, 4):
            aug_img = apply_random_aug(base_img, seed=ctx.seed + i)
            axes[i].imshow(aug_img)
            axes[i].set_title(f"Aug {i}")
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Lightweight augmentation preview", fontsize=13)
        plt.tight_layout()
        save_fig(ctx, fig, dataset_tag, "augmentation_preview")
        plt.show()


# ---------- Leakage visualizers ----------


def show_label_conflict_examples(ctx: RunbookContext, conflict_coords: pd.DataFrame, valid_coords: pd.DataFrame, dataset_tag: str, max_pairs: int = 3):
    if conflict_coords.empty:
        print("No conflicting-label coords to visualize.")
        return
    shown = 0
    for _, row in conflict_coords.iterrows():
        paths = row["paths"]
        entries = []
        for p in paths:
            match = valid_coords[valid_coords["path"] == p]
            if not match.empty:
                entries.append(match.iloc[0])
        labels_seen = {}
        for e in entries:
            lbl = e["label"]
            if lbl not in labels_seen:
                labels_seen[lbl] = e
            if len(labels_seen) >= 2:
                break
        if len(labels_seen) < 2:
            continue
        items = list(labels_seen.values())[:2]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"Same coord, opposite labels (leakage risk)\n{items[0]['label']} vs {items[1]['label']}")
        for ax, e in zip(axes, items):
            ax.imshow(Image.open(e["path"]))
            ax.set_title(f"{e['split']} / {e['label']}\n{Path(e['path']).name}", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        save_fig(ctx, fig, dataset_tag, f"conflicting_pair_{shown}")
        plt.show()
        shown += 1
        if shown >= max_pairs:
            break


def show_cross_split_same_coord(ctx: RunbookContext, valid_coords: pd.DataFrame, dataset_tag: str, max_pairs: int = 3):
    coord_groups = valid_coords.groupby(["lon", "lat"])
    pairs_shown = 0
    for (lon, lat), group in coord_groups:
        splits = group["split"].unique()
        if len(splits) <= 1:
            continue
        group = group.sort_values("split")
        combos = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group.iloc[i]["split"] == group.iloc[j]["split"]:
                    continue
                combos.append((group.iloc[i], group.iloc[j]))
        if not combos:
            continue
        a, b = combos[0]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"Same coord in multiple splits\n({lon:.6f}, {lat:.6f})")
        for ax, e in zip(axes, [a, b]):
            ax.imshow(Image.open(e["path"]))
            ax.set_title(f"{e['split']} / {e['label']}\n{Path(e['path']).name}", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        save_fig(ctx, fig, dataset_tag, f"cross_split_same_coord_{pairs_shown}")
        plt.show()
        pairs_shown += 1
        if pairs_shown >= max_pairs:
            break


def collect_duplicate_stems(df: pd.DataFrame):
    dup_counts = {}
    for split, group in df.groupby("split"):
        stems = group["stem"].value_counts()
        dup_counts[split] = int((stems > 1).sum())
    return dup_counts


def conflicts_per_split(valid_coords: pd.DataFrame, rounding: int = 6):
    conflicts = {}
    for split, group in valid_coords.groupby("split"):
        coord_labels = group.groupby(group.apply(lambda r: coord_key(r["lon"], r["lat"], rounding), axis=1))["label"].nunique()
        conflicts[split] = int((coord_labels > 1).sum())
    return conflicts


# ---------- 3D leakage visualization ----------


def plot_split_stack_3d(ctx: RunbookContext, df: pd.DataFrame, dataset_tag: str, elev: float = 25.0, azim: float = -60.0):
    valid = df.dropna(subset=["lon", "lat"])
    if valid.empty:
        print("No coordinates to plot in 3D.")
        return
    split_order = sorted(valid["split"].unique())
    split_to_z = {s: i for i, s in enumerate(split_order)}
    colors = valid["label"].map(ctx.label_palette).fillna("#888888")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(valid["lon"], valid["lat"], valid["split"].map(split_to_z), c=colors, s=18, alpha=0.75, edgecolors="none")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Split")
    ax.set_zticks(list(split_to_z.values()))
    ax.set_zticklabels(split_order)
    ax.set_title("Coord stack across splits (hue = label)")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    save_fig(ctx, fig, dataset_tag, "split_stack_3d")
    plt.show()


def _resolve_zoom_box(
    dfs: List[pd.DataFrame],
    zoom_box: Tuple[Tuple[float, float], Tuple[float, float]] | None,
    pad: float = 0.0,
    auto_bins: int = 50,
    auto_expand_bins: int = 1,
    peak_quantile: float = 0.05,
    rel_pad: float = 0.01,
    min_span: float = 0.01,
):
    if zoom_box is not None:
        (lon_min, lon_max), (lat_min, lat_max) = zoom_box
        lon_center = (lon_min + lon_max) / 2
        lat_center = (lat_min + lat_max) / 2
        lon_span = max(abs(lon_max - lon_min), min_span)
        lat_span = max(abs(lat_max - lat_min), min_span)
        lon_min = lon_center - lon_span / 2
        lon_max = lon_center + lon_span / 2
        lat_min = lat_center - lat_span / 2
        lat_max = lat_center + lat_span / 2
    else:
        combined = pd.concat([df.dropna(subset=["lon", "lat"])[["lon", "lat"]] for df in dfs], ignore_index=True)
        if combined.empty:
            return None
        lon_vals = combined["lon"].to_numpy()
        lat_vals = combined["lat"].to_numpy()
        hist, lon_edges, lat_edges = np.histogram2d(lon_vals, lat_vals, bins=auto_bins)
        max_idx = np.unravel_index(np.argmax(hist), hist.shape)
        lon_lower_edge = lon_edges[max(max_idx[0] - auto_expand_bins, 0)]
        lon_upper_edge = lon_edges[min(max_idx[0] + auto_expand_bins + 1, len(lon_edges) - 1)]
        lat_lower_edge = lat_edges[max(max_idx[1] - auto_expand_bins, 0)]
        lat_upper_edge = lat_edges[min(max_idx[1] + auto_expand_bins + 1, len(lat_edges) - 1)]

        in_peak = (lon_vals >= lon_lower_edge) & (lon_vals <= lon_upper_edge) & (lat_vals >= lat_lower_edge) & (lat_vals <= lat_upper_edge)
        peak_points = combined[in_peak]
        if peak_points.empty:
            lon_min, lon_max = combined["lon"].quantile([peak_quantile, 1 - peak_quantile])
            lat_min, lat_max = combined["lat"].quantile([peak_quantile, 1 - peak_quantile])
        else:
            lon_min, lon_max = peak_points["lon"].quantile([peak_quantile, 1 - peak_quantile])
            lat_min, lat_max = peak_points["lat"].quantile([peak_quantile, 1 - peak_quantile])

        lon_span = max(float(lon_max) - float(lon_min), min_span)
        lat_span = max(float(lat_max) - float(lat_min), min_span)
        lon_pad = max(pad, lon_span * rel_pad)
        lat_pad = max(pad, lat_span * rel_pad)
        lon_min, lon_max = float(lon_min) - lon_pad, float(lon_max) + lon_pad
        lat_min, lat_max = float(lat_min) - lat_pad, float(lat_max) + lat_pad
        return ((lon_min, lon_max), (lat_min, lat_max))
    lon_min, lon_max = sorted((float(lon_min), float(lon_max)))
    lat_min, lat_max = sorted((float(lat_min), float(lat_max)))
    return ((lon_min - pad, lon_max + pad), (lat_min - pad, lat_max + pad))


def _add_zoom_box(ax, zoom_box: Tuple[Tuple[float, float], Tuple[float, float]], z_level: float, color: str = "#c54a6b", linewidth: float = 2.0, fill_alpha: float | None = None):
    (lon_min, lon_max), (lat_min, lat_max) = zoom_box
    rect = Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        fill=fill_alpha is not None,
        lw=linewidth,
        ec=color,
        fc=color if fill_alpha is not None else "none",
        alpha=fill_alpha,
    )
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=z_level, zdir="z")


def _apply_zoom_limits(ax, zoom_box: Tuple[Tuple[float, float], Tuple[float, float]]):
    (lon_min, lon_max), (lat_min, lat_max) = zoom_box
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)


def _set_box_aspect_from_limits(ax, z_scale: float = 1.0, min_span: float = 0.02, equalize: bool = True):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    spans = [
        max(abs(xlim[1] - xlim[0]), min_span),
        max(abs(ylim[1] - ylim[0]), min_span),
        max(abs(zlim[1] - zlim[0]) * z_scale, min_span),
    ]
    if equalize:
        max_span = max(spans)
        spans = [max_span, max_span, max_span]
    ax.set_box_aspect(tuple(spans))


def _apply_tick_locators(ax, n_ticks: int = 4):
    ax.xaxis.set_major_locator(MaxNLocator(n_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(n_ticks))


def _scatter_split_stack(ax, df: pd.DataFrame, label_palette: Dict[str, str], zoom_box: Tuple[Tuple[float, float], Tuple[float, float]] | None = None, clip_to_zoom: bool = False):
    valid = df.dropna(subset=["lon", "lat"])
    split_order = sorted(valid["split"].unique())
    split_to_z = {s: i for i, s in enumerate(split_order)}
    plot_df = valid
    if clip_to_zoom and zoom_box is not None:
        (lon_min, lon_max), (lat_min, lat_max) = zoom_box
        plot_df = valid[
            (valid["lon"] >= lon_min)
            & (valid["lon"] <= lon_max)
            & (valid["lat"] >= lat_min)
            & (valid["lat"] <= lat_max)
        ]
    colors = plot_df["label"].map(label_palette).fillna("#888888")
    ax.scatter(plot_df["lon"], plot_df["lat"], plot_df["split"].map(split_to_z), c=colors, s=14, alpha=0.75, edgecolors="none")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Split")
    ax.set_zticks(list(split_to_z.values()))
    ax.set_zticklabels(split_order)
    return split_to_z


def plot_split_stack_3d_pair(ctx: RunbookContext, left_df: pd.DataFrame, right_df: pd.DataFrame, left_tag: str, right_tag: str, elev: float = 25.0, azim: float = -60.0, include_zoom: bool = True, zoom_box: Tuple[Tuple[float, float], Tuple[float, float]] | None = None, zoom_padding: float = 0.0, box_color: str = "#c54a6b", auto_zoom_bins: int = 140, auto_expand_bins: int = 0, auto_peak_quantile: float = 0.01, auto_rel_pad: float = 0.01):
    if left_df.dropna(subset=["lon", "lat"]).empty or right_df.dropna(subset=["lon", "lat"]).empty:
        print("Not enough coordinates in one of the datasets to plot 3D comparison.")
        return

    resolved_zoom = _resolve_zoom_box(
        [left_df, right_df],
        zoom_box,
        pad=zoom_padding,
        auto_bins=auto_zoom_bins,
        auto_expand_bins=auto_expand_bins,
        peak_quantile=auto_peak_quantile,
        rel_pad=auto_rel_pad,
    ) if include_zoom else None
    if include_zoom and resolved_zoom is not None:
        fig = plt.figure(figsize=(14, 12))
        grid = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)
        ax1 = fig.add_subplot(grid[0, 0], projection="3d")
        ax2 = fig.add_subplot(grid[0, 1], projection="3d")
        ax3 = fig.add_subplot(grid[1, 0], projection="3d")
        ax4 = fig.add_subplot(grid[1, 1], projection="3d")

        left_split_to_z = _scatter_split_stack(ax1, left_df, ctx.label_palette, resolved_zoom)
        left_box_z = min(left_split_to_z.values()) - 0.15 if left_split_to_z else -0.1
        _add_zoom_box(ax1, resolved_zoom, left_box_z, color=box_color, fill_alpha=0.08)
        ax1.set_title(f"{left_tag}: coord stack")
        ax1.view_init(elev=elev, azim=azim)
        _set_box_aspect_from_limits(ax1, z_scale=0.7, min_span=0.01)
        _apply_tick_locators(ax1, n_ticks=4)

        right_split_to_z = _scatter_split_stack(ax2, right_df, ctx.label_palette, resolved_zoom)
        right_box_z = min(right_split_to_z.values()) - 0.15 if right_split_to_z else -0.1
        _add_zoom_box(ax2, resolved_zoom, right_box_z, color=box_color, fill_alpha=0.08)
        ax2.set_title(f"{right_tag}: coord stack")
        ax2.view_init(elev=elev, azim=azim)
        _set_box_aspect_from_limits(ax2, z_scale=0.7, min_span=0.01)
        _apply_tick_locators(ax2, n_ticks=4)

        _scatter_split_stack(ax3, left_df, ctx.label_palette, resolved_zoom, clip_to_zoom=True)
        _apply_zoom_limits(ax3, resolved_zoom)
        _set_box_aspect_from_limits(ax3, z_scale=0.7, min_span=0.01)
        _apply_tick_locators(ax3, n_ticks=4)
        ax3.set_title(f"{left_tag}: zoom")
        ax3.view_init(elev=elev, azim=azim)

        _scatter_split_stack(ax4, right_df, ctx.label_palette, resolved_zoom, clip_to_zoom=True)
        _apply_zoom_limits(ax4, resolved_zoom)
        _set_box_aspect_from_limits(ax4, z_scale=0.7, min_span=0.01)
        _apply_tick_locators(ax4, n_ticks=4)
        ax4.set_title(f"{right_tag}: zoom")
        ax4.view_init(elev=elev, azim=azim)

        fig.suptitle(f"{left_tag} vs {right_tag}: coord stack + zoom")
        fig.subplots_adjust(top=0.93, hspace=0.2, wspace=0.15)
        save_fig(ctx, fig, "compare", "split_stack_3d_side_by_side_zoom")
        plt.show()
        return

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    _scatter_split_stack(ax1, left_df, ctx.label_palette)
    ax1.set_title(f"{left_tag}: coord stack")
    ax1.view_init(elev=elev, azim=azim)
    _set_box_aspect_from_limits(ax1, z_scale=0.7, min_span=0.01)
    _apply_tick_locators(ax1, n_ticks=4)
    _scatter_split_stack(ax2, right_df, ctx.label_palette)
    ax2.set_title(f"{right_tag}: coord stack")
    ax2.view_init(elev=elev, azim=azim)
    _set_box_aspect_from_limits(ax2, z_scale=0.7, min_span=0.01)
    _apply_tick_locators(ax2, n_ticks=4)
    plt.tight_layout()
    save_fig(ctx, fig, "compare", "split_stack_3d_side_by_side")
    plt.show()


# ---------- Dataset profiling ----------


def analyze_dataset(ctx: RunbookContext, data_dir: Path, dataset_tag: str):
    print(f"\n=== Profiling {dataset_tag} @ {data_dir} ===")
    df = load_image_dataframe(ctx, data_dir)
    if df.empty:
        print("No images found.")
        return {"dataset": dataset_tag, "root": str(data_dir), "images": 0}, df, None, None

    summary: Dict[str, object] = {
        "dataset": dataset_tag,
        "root": str(data_dir),
        "images": len(df),
    }

    counts = plot_class_balance(ctx, df, dataset_tag)
    summary["class_counts"] = counts

    valid_coords = df.dropna(subset=["lon", "lat"])
    summary["unique_coords"] = valid_coords[["lon", "lat"]].drop_duplicates().shape[0]

    coord_counts, conflict_coords = plot_coord_conflicts(ctx, valid_coords, dataset_tag)
    summary["coords_with_multiple_images"] = int((coord_counts["count"] > 1).sum())
    summary["coords_with_conflicts"] = len(conflict_coords)
    plot_images_per_coord_hist(ctx, coord_counts, dataset_tag)
    plot_spatial_scatter(ctx, valid_coords, dataset_tag)
    plot_spatial_density(ctx, valid_coords, dataset_tag)
    plot_map(ctx, valid_coords, dataset_tag)

    coord_map, coord_summaries = summarize_coord_splits(valid_coords)
    summary["coords_multi_split"] = sum(1 for c in coord_summaries if len(c["splits"]) > 1)
    plot_multisplit_bar(ctx, coord_summaries, dataset_tag)
    plot_split_pair_counts(ctx, coord_summaries, dataset_tag)
    summary["coord_overlap_by_rounding"] = coord_overlap_by_rounding(valid_coords, ctx.roundings)
    plot_coord_overlap_heatmap(ctx, valid_coords, dataset_tag)

    near_pairs = near_neighbor_pairs(valid_coords, ctx.merge_distance_m)
    summary["cross_split_pairs_within_m"] = len(near_pairs)
    if near_pairs:
        print(f"Cross-split pairs within {ctx.merge_distance_m} m (showing first 5):")
        for a, b, d in near_pairs[:5]:
            print("-", (a["split"], a["label"], Path(a["path"]).name), "<->", (b["split"], b["label"], Path(b["path"]).name), f"| {d:.2f} m")

    overlap = cross_split_hash_overlap(ctx, df)
    summary["cross_split_duplicate_hashes"] = overlap["hash"].nunique() if not overlap.empty else 0
    if not overlap.empty:
        combo_counts = overlap.groupby("hash")["split"].apply(lambda s: ", ".join(sorted(s.unique()))).value_counts()
        print("Overlap split combinations:\n", combo_counts)
        display(overlap.head())

    size_df = collect_image_sizes(ctx, df)
    summary["size_summary"] = size_df[["width", "height", "area"]].describe().to_dict()
    plot_size_distributions(ctx, size_df, dataset_tag)

    channel_stats = compute_channel_stats(ctx, df, sample_size=ctx.stat_sample_size)
    summary["pixel_mean"] = channel_stats["means"].tolist()
    summary["pixel_std"] = channel_stats["stds"].tolist()
    summary["pixel_sample_size"] = channel_stats["sample_size"]
    plot_channel_histograms(ctx, channel_stats, dataset_tag)

    photo_stats = brightness_contrast(ctx, df)
    summary["brightness_mean"] = float(photo_stats["brightness"].mean())
    summary["contrast_mean"] = float(photo_stats["contrast"].mean())
    plot_brightness_contrast(ctx, photo_stats, dataset_tag)

    for split, label in product(["train", "validation", "test"], ["damage", "no_damage"]):
        plot_image_grid(ctx, df, dataset_tag, split, label)

    corrupted = check_corrupted(ctx, df)
    summary["corrupted_files"] = len(corrupted)
    if corrupted:
        print("Corrupted files:")
        for path, err in corrupted[:5]:
            print(path, err)

    duplicates = find_duplicates(ctx, df)
    summary["duplicate_pairs_sampled"] = len(duplicates)
    plot_duplicate_example(ctx, duplicates, dataset_tag)

    show_label_conflict_examples(ctx, conflict_coords, valid_coords, dataset_tag, max_pairs=3)
    show_cross_split_same_coord(ctx, valid_coords, dataset_tag, max_pairs=3)
    sample_same_coord_grids(ctx, coord_counts, valid_coords, dataset_tag, sample_n=3, images_per_coord=6)

    summary["conflicts_by_split"] = conflicts_per_split(valid_coords)
    summary["duplicate_stems"] = collect_duplicate_stems(df)

    plot_augmentation_preview(ctx, df, dataset_tag)

    plot_split_stack_3d(ctx, df, dataset_tag)

    return summary, df, channel_stats, photo_stats
