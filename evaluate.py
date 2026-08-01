"""
evaluate.py — Test-set evaluation for the trained QM9 MPNN.

Responsibilities:
  1. Load the best checkpoint saved by train.py (weights + normalization
     stats — loading the stats alongside the weights, rather than
     recomputing them, guarantees evaluation uses the exact same
     train-set-derived mean/std the model was trained against).
  2. Run inference over the held-out test set (never seen during training
     or model selection).
  3. Report MAE and RMSE in eV, plus the same metrics converted to Hartree
     and meV for direct comparison against published QM9 benchmarks:
       - Gilmer et al. 2017 (MPNN):      ~0.0043 Hartree (~117 meV)
       - Wu et al. 2018 (MoleculeNet):   varies by method, ~60-100 meV
       - Klicpera et al. 2020 (DimeNet): ~33 meV (specialized architecture)
  4. Save per-molecule predictions vs. ground truth to a JSON file for
     visualize.py to plot.

Run this file directly after train.py has produced checkpoints/best_model.pt.
"""

import json
import os

import torch
import torch.nn.functional as F

import config
from dataset import get_dataloaders
from model import MPNN
from train import denormalize


def evaluate():
    print("=" * 60)
    print("Loading best checkpoint...")
    print("=" * 60)

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE, weights_only=False)
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    print(f"Checkpoint from epoch {checkpoint['epoch']} "
          f"(val MAE at save time: {checkpoint['val_mae_ev']:.4f} eV)")
    print(f"Normalization — mean: {mean:.4f} eV, std: {std:.4f} eV")

    model = MPNN().to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("-" * 60)
    print("Loading test set...")
    _, _, test_loader, dataset_mean, dataset_std = get_dataloaders()

    # Sanity check: the normalization stats recomputed fresh from dataset.py
    # should match what's stored in the checkpoint. If they don't, the split
    # seed or dataset version has changed since training, which would make
    # any comparison invalid.
    if abs(dataset_mean - mean) > 1e-4 or abs(dataset_std - std) > 1e-4:
        print("WARNING: normalization stats differ between checkpoint and "
              "current dataset split. Results may not be meaningful.")

    print("-" * 60)
    print("Running inference on test set...")

    all_preds_ev = []
    all_true_ev = []
    all_smiles = []

    total_abs_error = 0.0
    total_sq_error = 0.0
    num_graphs = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(config.DEVICE)
            predictions = model(batch)

            pred_ev = denormalize(predictions, mean, std).squeeze(-1)
            true_ev = denormalize(batch.y, mean, std).squeeze(-1)

            errors = pred_ev - true_ev
            total_abs_error += errors.abs().sum().item()
            total_sq_error += (errors ** 2).sum().item()
            num_graphs += pred_ev.size(0)

            all_preds_ev.extend(pred_ev.cpu().tolist())
            all_true_ev.extend(true_ev.cpu().tolist())
            if hasattr(batch, "smiles"):
                all_smiles.extend(batch.smiles)

    mae_ev = total_abs_error / num_graphs
    rmse_ev = (total_sq_error / num_graphs) ** 0.5
    mae_hartree = mae_ev / config.HARTREE_TO_EV
    mae_mev = mae_ev * 1000

    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Number of test molecules:  {num_graphs}")
    print(f"MAE:                       {mae_ev:.4f} eV  ({mae_mev:.1f} meV, {mae_hartree:.6f} Hartree)")
    print(f"RMSE:                      {rmse_ev:.4f} eV")
    print("-" * 60)
    print("Literature comparison (HOMO-LUMO gap MAE, same QM9 target):")
    print(f"  Gilmer et al. 2017 (MPNN):        ~117 meV")
    print(f"  Wu et al. 2018 (MoleculeNet):     ~60-100 meV (method-dependent)")
    print(f"  Klicpera et al. 2020 (DimeNet):   ~33 meV (specialized, directional)")
    print(f"  This model:                       {mae_mev:.1f} meV")
    print("=" * 60)

    results = {
        "num_test_molecules": num_graphs,
        "mae_ev": mae_ev,
        "rmse_ev": rmse_ev,
        "mae_hartree": mae_hartree,
        "mae_mev": mae_mev,
        "checkpoint_epoch": checkpoint["epoch"],
        "predictions_ev": all_preds_ev,
        "ground_truth_ev": all_true_ev,
        "smiles": all_smiles,
    }

    results_path = os.path.join(config.RESULTS_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {results_path}")

    return results


if __name__ == "__main__":
    evaluate()