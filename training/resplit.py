"""
VoiceShield — Re-split metadata_clean.csv with proper stratification.
Speaker-level split ensuring both classes in every split.

Usage: python -m training.resplit
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = Path(r"C:\Users\dusha\Desktop\voiceshield\dataset")
METADATA_CSV = DATASET_ROOT / "metadata_clean.csv"
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def resplit():
    np.random.seed(SEED)

    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded {len(df)} rows")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    # --- Split REAL speakers (label=0) ---
    real_df = df[df["label"] == 0].copy()
    real_speakers = real_df["speaker_id"].unique()
    np.random.shuffle(real_speakers)

    n_real = len(real_speakers)
    n_train_r = int(n_real * TRAIN_RATIO)
    n_val_r = int(n_real * VAL_RATIO)

    train_spk_real = set(real_speakers[:n_train_r])
    val_spk_real = set(real_speakers[n_train_r:n_train_r + n_val_r])
    test_spk_real = set(real_speakers[n_train_r + n_val_r:])

    print(f"REAL speakers: {n_real} total → train={len(train_spk_real)}, val={len(val_spk_real)}, test={len(test_spk_real)}")

    # --- Split FAKE speakers (label=1) ---
    fake_df = df[df["label"] == 1].copy()
    fake_speakers = fake_df["speaker_id"].unique()
    np.random.shuffle(fake_speakers)

    n_fake = len(fake_speakers)
    n_train_f = int(n_fake * TRAIN_RATIO)
    n_val_f = int(n_fake * VAL_RATIO)

    train_spk_fake = set(fake_speakers[:n_train_f])
    val_spk_fake = set(fake_speakers[n_train_f:n_train_f + n_val_f])
    test_spk_fake = set(fake_speakers[n_train_f + n_val_f:])

    print(f"FAKE speakers: {n_fake} total → train={len(train_spk_fake)}, val={len(val_spk_fake)}, test={len(test_spk_fake)}")

    # --- Assign splits ---
    def assign_split(row):
        sid = row["speaker_id"]
        lab = row["label"]
        if lab == 0:
            if sid in train_spk_real:
                return "train"
            elif sid in val_spk_real:
                return "val"
            else:
                return "test"
        else:
            if sid in train_spk_fake:
                return "train"
            elif sid in val_spk_fake:
                return "val"
            else:
                return "test"

    df["split"] = df.apply(assign_split, axis=1)

    # --- Update file_path to match new split directories ---
    # The file_path currently has the old split dir (e.g. train/real/xxx.wav)
    # We need to update it to the new split
    def update_path(row):
        old_path = row["file_path"]
        parts = old_path.replace("\\", "/").split("/")
        # parts: [old_split, class_dir, filename]
        class_dir = parts[1] if len(parts) >= 3 else ("real" if row["label"] == 0 else "fake")
        filename = parts[-1]
        return f"{row['split']}/{class_dir}/{filename}"

    df["file_path"] = df.apply(update_path, axis=1)

    # --- Verify no speaker leakage ---
    for label_name, label_val in [("REAL", 0), ("FAKE", 1)]:
        sub = df[df["label"] == label_val]
        splits = {}
        for s in ["train", "val", "test"]:
            splits[s] = set(sub[sub["split"] == s]["speaker_id"].unique())
        for s1 in ["train", "val", "test"]:
            for s2 in ["train", "val", "test"]:
                if s1 >= s2:
                    continue
                overlap = splits[s1] & splits[s2]
                if overlap:
                    print(f"  ⚠ LEAKAGE {label_name} {s1}↔{s2}: {len(overlap)} shared speakers")
                else:
                    print(f"  ✓ No leakage {label_name} {s1}↔{s2}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  RESPLIT SUMMARY")
    print(f"{'='*60}")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        n_r = (sub["label"] == 0).sum()
        n_f = (sub["label"] == 1).sum()
        n_spk = sub["speaker_id"].nunique()
        print(f"  {split:5s}: {len(sub):>6d} samples  (R={n_r}, F={n_f})  speakers={n_spk}")
    print(f"  Total: {len(df):>6d}")
    print(f"{'='*60}")

    # --- Save ---
    df.to_csv(METADATA_CSV, index=False)
    print(f"\nSaved to {METADATA_CSV}")

    # --- IMPORTANT: Warn about physical file locations ---
    print(f"\n⚠  NOTE: file_path column now reflects new splits, but the actual")
    print(f"   .wav files are still in their ORIGINAL directories on disk.")
    print(f"   The dataset loader resolves paths via DATASET_ROOT / file_path.")
    print(f"   If files are not found, the old paths still work — we'll fix the")
    print(f"   loader to search in original locations if needed.")

    return df


if __name__ == "__main__":
    resplit()
