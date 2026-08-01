"""
visualize.py — Generate result plots for the QM9 HOMO-LUMO gap MPNN.

Produces three figures, saved to results/:
  1. training_curves.png   — train/val loss and val MAE (eV) vs. epoch
  2. predicted_vs_true.png — scatter plot of predicted vs. true HOMO-LUMO
                              gap on the test set, with the ideal y=x line
  3. error_distribution.png — histogram of prediction errors (eV), to check
                              whether errors are roughly symmetric/unbiased
                              or show systematic skew for certain gap ranges

Run this file directly after both train.py and evaluate.py have completed
(it reads results/training_history.json and results/test_results.json).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/script use
import matplotlib.pyplot as plt
import numpy as np

import config


def plot_training_curves(history_path, save_path):
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train loss (normalized MSE)")
    axes[0].plot(epochs, history["val_loss"], label="Val loss (normalized MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (standardized units)")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_mae_ev"], color="darkorange", label="Val MAE (eV)")
    axes[1].axhline(0.117, color="gray", linestyle="--", alpha=0.6,
                     label="Gilmer et al. 2017 MPNN (~117 meV)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (eV)")
    axes[1].set_title("Validation MAE (HOMO-LUMO gap)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_predicted_vs_true(results_path, save_path):
    with open(results_path, "r") as f:
        results = json.load(f)

    preds = np.array(results["predictions_ev"])
    true = np.array(results["ground_truth_ev"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true, preds, s=4, alpha=0.25, color="steelblue")

    lo = min(true.min(), preds.min())
    hi = max(true.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], color="crimson", linestyle="--", label="Ideal (y = x)")

    ax.set_xlabel("True HOMO-LUMO gap (eV)")
    ax.set_ylabel("Predicted HOMO-LUMO gap (eV)")
    ax.set_title(f"Predicted vs. True HOMO-LUMO Gap (Test Set, n={len(true)})\n"
                 f"MAE = {results['mae_ev']:.4f} eV | RMSE = {results['rmse_ev']:.4f} eV")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_error_distribution(results_path, save_path):
    with open(results_path, "r") as f:
        results = json.load(f)

    preds = np.array(results["predictions_ev"])
    true = np.array(results["ground_truth_ev"])
    errors = preds - true

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=80, color="mediumseagreen", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="crimson", linestyle="--", label="Zero error")
    ax.axvline(errors.mean(), color="darkorange", linestyle="-",
               label=f"Mean error: {errors.mean():.4f} eV")

    ax.set_xlabel("Prediction error (Predicted - True, eV)")
    ax.set_ylabel("Count")
    ax.set_title("Test Set Error Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def visualize():
    history_path = config.TRAINING_HISTORY_PATH
    results_path = os.path.join(config.RESULTS_DIR, "test_results.json")

    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"{history_path} not found — run train.py first."
        )
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"{results_path} not found — run evaluate.py first."
        )

    print("=" * 60)
    print("Generating plots...")
    print("=" * 60)

    plot_training_curves(
        history_path, os.path.join(config.RESULTS_DIR, "training_curves.png")
    )
    plot_predicted_vs_true(
        results_path, os.path.join(config.RESULTS_DIR, "predicted_vs_true.png")
    )
    plot_error_distribution(
        results_path, os.path.join(config.RESULTS_DIR, "error_distribution.png")
    )

    print("=" * 60)
    print("All plots generated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    visualize()