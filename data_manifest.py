"""
data_manifest.py - Manifest creation, pairing, and split generation.

Produces:
  manifests/ir_manifest.csv
  manifests/rgb_manifest.csv
  manifests/split_assignments.csv
  manifests/paired_scene_eval.csv

Manifest schema (both modalities):
  modality, class_name, label_id, patch_path, source_frame_path,
  annotation_id, bbox_xywh, video_id, frame_index, split
"""

import os
import re
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

# ── Class definitions ──────────────────────────────────────────────────────
CLASSES = {'bike': 0, 'bus': 1, 'car': 2, 'person': 3, 'sign': 4,
           'motor': 5, 'light': 6, 'truck': 7}
NUM_CLASSES = len(CLASSES)

# ── Paths (relative to project root) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
IR_PATCHES_DIR = PROJECT_ROOT / "train_thermal_patches"
IR_FRAMES_DIR = PROJECT_ROOT / "data"
RGB_PATCHES_DIR = PROJECT_ROOT / "train_rgb_patches"
RGB_FRAMES_DIR = PROJECT_ROOT / "rgb_full_frames"
COCO_JSON = PROJECT_ROOT / "coco.json"
INDEX_JSON = PROJECT_ROOT / "index.json"
RGB_CSV = PROJECT_ROOT / "patches_rgb_train_info.csv"

# Filename pattern: video-<VID>-frame-<FINDEX>-<FRAMEID>_<ANNOT_ID>.jpg
PATCH_RE = re.compile(
    r"video-(?P<video_id>[^-]+)-frame-(?P<frame_index>\d+)-(?P<frame_id>[^_]+)_(?P<annot_id>\d+)\.jpg"
)
# Source frame pattern: video-<VID>-frame-<FINDEX>-<FRAMEID>.jpg (possibly inside data/ prefix)
FRAME_RE = re.compile(
    r"(?:data/)?video-(?P<video_id>[^-]+)-frame-(?P<frame_index>\d+)-(?P<frame_id>[^.]+)\.jpg"
)


def _deterministic_split(video_ids, val_fraction=0.2, seed=42):
    """Assign video_ids to 'train' or 'val' deterministically using hash."""
    assignments = {}
    # Sort for reproducibility, then use hash-based split
    rng = np.random.RandomState(seed)
    ids_sorted = sorted(video_ids)
    rng.shuffle(ids_sorted)
    n_val = max(1, int(len(ids_sorted) * val_fraction))
    val_set = set(ids_sorted[:n_val])
    for vid in video_ids:
        assignments[vid] = "val" if vid in val_set else "train"
    return assignments


def build_ir_manifest():
    """
    Reconstruct IR manifest from train_thermal_patches/ + coco.json.
    Maps each patch filename back to its annotation in coco.json via the
    annotation_id suffix in the filename.
    """
    print("Building IR manifest...")

    # Load COCO annotations
    with open(COCO_JSON, encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookup maps
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}
    ann_map = {ann["id"]: ann for ann in coco["annotations"]}

    rows = []
    for class_name in sorted(os.listdir(IR_PATCHES_DIR)):
        class_dir = IR_PATCHES_DIR / class_name
        if not class_dir.is_dir() or class_name not in CLASSES:
            continue

        for patch_file in sorted(os.listdir(class_dir)):
            m = PATCH_RE.match(patch_file)
            if not m:
                continue

            video_id = m.group("video_id")
            frame_index = int(m.group("frame_index"))
            frame_id = m.group("frame_id")
            annot_id = int(m.group("annot_id"))

            # Look up annotation
            ann = ann_map.get(annot_id)
            if ann is None:
                continue

            img = img_map.get(ann["image_id"])
            bbox_str = str(ann["bbox"])  # [x, y, w, h]
            source_frame_path = str(IR_FRAMES_DIR / f"video-{video_id}-frame-{frame_index:06d}-{frame_id}.jpg")

            rows.append({
                "modality": "ir",
                "class_name": class_name,
                "label_id": CLASSES[class_name],
                "patch_path": str(class_dir / patch_file),
                "source_frame_path": source_frame_path,
                "annotation_id": annot_id,
                "bbox_xywh": bbox_str,
                "video_id": video_id,
                "frame_index": frame_index,
                "split": "",  # filled later
            })

    df = pd.DataFrame(rows)
    print(f"  IR patches found: {len(df)}")
    print(f"  Class distribution:\n{df['class_name'].value_counts().to_string()}")
    return df


def build_rgb_manifest():
    """
    Build RGB manifest from patches_rgb_train_info.csv with enriched fields.
    Parses video_id and frame_index from filenames.
    """
    print("Building RGB manifest...")

    if not RGB_CSV.exists():
        raise FileNotFoundError(f"RGB CSV not found: {RGB_CSV}")

    df_raw = pd.read_csv(RGB_CSV)

    # Parse video_id and frame_index from original_filename
    parsed = df_raw["original_filename"].str.extract(
        r"video-(?P<video_id>[^-]+)-frame-(?P<frame_index>\d+)-"
    )

    # Parse annotation_id from patch_filename suffix
    annot_parsed = df_raw["patch_filename"].str.extract(r"_(\d+)\.jpg$")

    rows = []
    for i, raw_row in df_raw.iterrows():
        class_name = raw_row["class"]
        if class_name not in CLASSES:
            continue

        video_id = parsed.loc[i, "video_id"]
        frame_index = int(parsed.loc[i, "frame_index"])
        annot_id = int(annot_parsed.loc[i, 0]) if pd.notna(annot_parsed.loc[i, 0]) else -1

        patch_path = str(RGB_PATCHES_DIR / class_name / raw_row["patch_filename"])
        source_frame = raw_row["original_filename"]  # e.g., data/video-...-frame-...-....jpg
        # Map to local rgb_full_frames/ path
        frame_basename = os.path.basename(source_frame)
        source_frame_path = str(RGB_FRAMES_DIR / frame_basename)

        rows.append({
            "modality": "rgb",
            "class_name": class_name,
            "label_id": CLASSES[class_name],
            "patch_path": patch_path,
            "source_frame_path": source_frame_path,
            "annotation_id": annot_id,
            "bbox_xywh": str(raw_row["bbox"]),
            "video_id": video_id,
            "frame_index": frame_index,
            "split": "",
        })

    df = pd.DataFrame(rows)
    print(f"  RGB patches found: {len(df)}")
    print(f"  Class distribution:\n{df['class_name'].value_counts().to_string()}")
    return df


def assign_splits(ir_df, rgb_df, val_fraction=0.2, seed=42):
    """
    Assign 80/20 train/val split by video_id independently per modality.
    Returns updated DataFrames and a split_assignments DataFrame.
    """
    print("Assigning video-level splits...")

    ir_vids = ir_df["video_id"].unique()
    rgb_vids = rgb_df["video_id"].unique()

    ir_split = _deterministic_split(ir_vids, val_fraction, seed)
    rgb_split = _deterministic_split(rgb_vids, val_fraction, seed)

    ir_df = ir_df.copy()
    rgb_df = rgb_df.copy()
    ir_df["split"] = ir_df["video_id"].map(ir_split)
    rgb_df["split"] = rgb_df["video_id"].map(rgb_split)

    # Build split_assignments.csv
    split_rows = []
    for vid, split in ir_split.items():
        split_rows.append({"modality": "ir", "video_id": vid, "split": split})
    for vid, split in rgb_split.items():
        split_rows.append({"modality": "rgb", "video_id": vid, "split": split})
    split_df = pd.DataFrame(split_rows)

    # Stats
    for name, df in [("IR", ir_df), ("RGB", rgb_df)]:
        train_n = (df["split"] == "train").sum()
        val_n = (df["split"] == "val").sum()
        train_vids = df[df["split"] == "train"]["video_id"].nunique()
        val_vids = df[df["split"] == "val"]["video_id"].nunique()
        print(f"  {name}: {train_n} train ({train_vids} vids), {val_n} val ({val_vids} vids)")

    return ir_df, rgb_df, split_df


def build_paired_scene_eval(ir_df, rgb_df):
    """
    Build paired_scene_eval.csv: matched (rgb_video_id, thermal_video_id, frame_index)
    where both the RGB and IR video_ids are in their respective validation splits.
    Uses the thermal-to-RGB video pairing from index.json.
    """
    print("Building paired scene eval set...")

    with open(INDEX_JSON, encoding="utf-8") as f:
        idx = json.load(f)

    # Extract thermal -> RGB video pairings
    thermal_to_rgb = {}
    for v in idx["videos"]:
        m = re.search(r'\{"RGB"\s*:\s*"([^"]+)"\}', v.get("description", ""))
        if m:
            thermal_to_rgb[v["id"]] = m.group(1)

    rgb_to_thermal = {v: k for k, v in thermal_to_rgb.items()}

    # Get validation video_ids for each modality
    ir_val_vids = set(ir_df[ir_df["split"] == "val"]["video_id"].unique())
    rgb_val_vids = set(rgb_df[rgb_df["split"] == "val"]["video_id"].unique())

    # Find paired scenes where both are in validation
    ir_val_frames = ir_df[ir_df["split"] == "val"].groupby(
        ["video_id", "frame_index"]
    ).size().reset_index(name="ir_patch_count")

    rgb_val_frames = rgb_df[rgb_df["split"] == "val"].groupby(
        ["video_id", "frame_index"]
    ).size().reset_index(name="rgb_patch_count")

    paired_rows = []
    for _, rgb_row in rgb_val_frames.iterrows():
        rgb_vid = rgb_row["video_id"]
        fi = rgb_row["frame_index"]

        if rgb_vid not in rgb_to_thermal:
            continue
        thermal_vid = rgb_to_thermal[rgb_vid]
        if thermal_vid not in ir_val_vids:
            continue

        # Check if this thermal frame exists in IR val
        ir_match = ir_val_frames[
            (ir_val_frames["video_id"] == thermal_vid) &
            (ir_val_frames["frame_index"] == fi)
        ]
        if len(ir_match) > 0:
            paired_rows.append({
                "rgb_video_id": rgb_vid,
                "thermal_video_id": thermal_vid,
                "frame_index": fi,
                "rgb_patch_count": int(rgb_row["rgb_patch_count"]),
                "ir_patch_count": int(ir_match.iloc[0]["ir_patch_count"]),
            })

    paired_df = pd.DataFrame(paired_rows)
    print(f"  Paired validation scenes: {len(paired_df)}")
    return paired_df


def generate_all(val_fraction=0.2, seed=42):
    """Generate all manifests and save to manifests/ directory."""
    MANIFESTS_DIR.mkdir(exist_ok=True)

    ir_df = build_ir_manifest()
    rgb_df = build_rgb_manifest()
    ir_df, rgb_df, split_df = assign_splits(ir_df, rgb_df, val_fraction, seed)
    paired_df = build_paired_scene_eval(ir_df, rgb_df)

    # Save
    ir_path = MANIFESTS_DIR / "ir_manifest.csv"
    rgb_path = MANIFESTS_DIR / "rgb_manifest.csv"
    split_path = MANIFESTS_DIR / "split_assignments.csv"
    paired_path = MANIFESTS_DIR / "paired_scene_eval.csv"

    ir_df.to_csv(ir_path, index=False)
    rgb_df.to_csv(rgb_path, index=False)
    split_df.to_csv(split_path, index=False)
    paired_df.to_csv(paired_path, index=False)

    print(f"\nSaved:")
    print(f"  {ir_path} ({len(ir_df)} rows)")
    print(f"  {rgb_path} ({len(rgb_df)} rows)")
    print(f"  {split_path} ({len(split_df)} rows)")
    print(f"  {paired_path} ({len(paired_df)} rows)")

    return ir_df, rgb_df, split_df, paired_df


if __name__ == "__main__":
    generate_all()
