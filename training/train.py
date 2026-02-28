"""
VoiceShield v2 — Main Training Script
Usage: python -m training.train
"""

import os
import sys
import json
import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path

from training.config import (
    MODEL_DIR, LOG_DIR, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, LR_PATIENCE, EARLY_STOP_PATIENCE,
    LABEL_SMOOTHING, SEED,
)
from training.model import build_resnet
from training.dataset import build_dataset, get_steps
from training.metrics import evaluate


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("=" * 64)
    print("  VOICESHIELD v2 — ResNet + Log-Mel Spectrogram Training")
    print("=" * 64)

    set_seed()

    # ── Directories ───────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / f"resnet_v2_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    # ── Datasets ──────────────────────────────────────────────────
    print("\nBuilding datasets...")
    train_ds = build_dataset("train", batch_size=BATCH_SIZE, shuffle=True)
    val_ds = build_dataset("val", batch_size=BATCH_SIZE, shuffle=False)
    test_ds = build_dataset("test", batch_size=BATCH_SIZE, shuffle=False)

    train_steps = get_steps("train", BATCH_SIZE)
    val_steps = get_steps("val", BATCH_SIZE)
    test_steps = get_steps("test", BATCH_SIZE)

    print(f"  Train steps/epoch: {train_steps}")
    print(f"  Val   steps/epoch: {val_steps}")
    print(f"  Test  steps:       {test_steps}")

    # ── Model ─────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_resnet()
    model.summary(print_fn=lambda s: print(f"  {s}"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    # ── Callbacks ─────────────────────────────────────────────────
    callbacks = [
        # Save best model by val AUC
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # Reduce LR on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOG_DIR / f"resnet_v2_{timestamp}"),
            histogram_freq=0,
        ),
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            str(run_dir / "training_log.csv"),
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────
    print("\nStarting training...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model as well
    model.save(str(run_dir / "final_model.keras"))
    print(f"\nFinal model saved to {run_dir / 'final_model.keras'}")

    # Save training history
    hist_dict = {k: [float(v) for v in vals]
                 for k, vals in history.history.items()}
    with open(run_dir / "history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    # ── Evaluate on test set ──────────────────────────────────────
    print("\n" + "=" * 64)
    print("  TEST SET EVALUATION")
    print("=" * 64)

    # Load best model
    best_model = tf.keras.models.load_model(
        str(run_dir / "best_model.keras"),
        custom_objects={"SpecAugment": __import__("training.spec_augment", fromlist=["SpecAugment"]).SpecAugment},
    )

    y_true = []
    y_prob = []
    for batch_x, batch_y in test_ds:
        preds = best_model.predict(batch_x, verbose=0)
        y_prob.extend(preds.flatten().tolist())
        y_true.extend(batch_y.numpy().flatten().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    results = evaluate(y_true, y_prob, threshold=0.5, print_report=True)

    # Save results
    results_save = {
        "auc": results["auc"],
        "eer": results["eer"],
        "eer_threshold": results["eer_threshold"],
        "accuracy": results["accuracy"],
        "confusion_matrix": results["cm"].tolist(),
    }
    with open(run_dir / "test_results.json", "w") as f:
        json.dump(results_save, f, indent=2)
    print(f"\nResults saved to {run_dir / 'test_results.json'}")

    print("\n" + "=" * 64)
    print("  TRAINING COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
