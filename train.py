"""
train.py — Training loop for the QM9 HOMO-LUMO gap MPNN.

Responsibilities:
  1. Build dataloaders (via dataset.get_dataloaders) and the model (via
     model.MPNN).
  2. Train with Adam + ReduceLROnPlateau + gradient clipping.
  3. Track validation MAE in eV (de-normalized), the unit used across the
     benchmark literature, so the number is directly interpretable and
     comparable (Gilmer et al. 2017 report ~0.0043 Hartree, ~0.12 eV MAE
     on this exact target with a similar MPNN).
  4. Early-stop on validation MAE plateau.
  5. Save the best checkpoint (model weights + normalization stats) and a
     JSON training history for later plotting in visualize.py.

Run this file directly to train from scratch.
"""

import json
import os
import time

import torch
import torch.nn.functional as F

import config
from dataset import get_dataloaders
from model import MPNN


def denormalize(y_norm, mean, std):
    """Convert standardized targets/predictions back to physical eV units."""
    return y_norm * std + mean


def run_epoch(model, loader, mean, std, optimizer=None):
    """
    Run one full pass over `loader`. If `optimizer` is provided, this is a
    training epoch (gradients computed and applied); otherwise it's an
    evaluation epoch (no_grad, used for validation).

    Returns:
        avg_loss (float): mean MSE loss in normalized (standardized) units
        mae_ev (float): mean absolute error in eV (de-normalized), the
                         physically meaningful metric reported in the README
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_abs_error_ev = 0.0
    num_graphs = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(config.DEVICE)

            if is_training:
                optimizer.zero_grad()

            predictions = model(batch)
            loss = F.mse_loss(predictions, batch.y)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()

            batch_size = batch.y.size(0)
            total_loss += loss.item() * batch_size
            num_graphs += batch_size

            # De-normalize both predictions and targets to compute MAE in eV,
            # the physically interpretable unit used in the literature.
            pred_ev = denormalize(predictions.detach(), mean, std)
            true_ev = denormalize(batch.y, mean, std)
            total_abs_error_ev += (pred_ev - true_ev).abs().sum().item()

    avg_loss = total_loss / num_graphs
    mae_ev = total_abs_error_ev / num_graphs
    return avg_loss, mae_ev


def save_latest_checkpoint(model, optimizer, scheduler, epoch, best_val_mae,
                            epochs_without_improvement, history, mean, std):
    """
    Save full training state after every epoch (not just improvements) so
    an interrupted run (Ctrl+C, crash, sleep) can be resumed exactly where
    it left off, rather than restarting from epoch 1.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_mae": best_val_mae,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "mean": mean,
            "std": std,
        },
        config.LATEST_CHECKPOINT_PATH,
    )


def train():
    torch.manual_seed(config.TORCH_SEED)

    print("=" * 60)
    print("Loading QM9 dataloaders...")
    print("=" * 60)
    train_loader, val_loader, test_loader, mean, std = get_dataloaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | "
          f"Test batches: {len(test_loader)}")
    print(f"Normalization — mean: {mean:.4f} eV, std: {std:.4f} eV")

    model = MPNN().to(config.DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    history = {"train_loss": [], "val_loss": [], "val_mae_ev": [], "lr": []}
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1

    # --- Resume from a previous run if a checkpoint exists ---
    if os.path.exists(config.LATEST_CHECKPOINT_PATH):
        print("-" * 60)
        print(f"Found existing checkpoint at {config.LATEST_CHECKPOINT_PATH}")
        checkpoint = torch.load(config.LATEST_CHECKPOINT_PATH, map_location=config.DEVICE, weights_only=False)

        # Sanity check: resuming only makes sense if normalization stats match
        # the current dataset split (same seed, same data).
        if abs(checkpoint["mean"] - mean) > 1e-4 or abs(checkpoint["std"] - std) > 1e-4:
            print("WARNING: checkpoint normalization stats don't match current "
                  "dataset split. Starting fresh instead of resuming.")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_val_mae = checkpoint["best_val_mae"]
            epochs_without_improvement = checkpoint["epochs_without_improvement"]
            history = checkpoint["history"]
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch} "
                  f"(best val MAE so far: {best_val_mae:.4f} eV)")
        print("-" * 60)

    print("=" * 60)
    print("Starting training" if start_epoch == 1 else "Resuming training")
    print("=" * 60)

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_mae_ev = run_epoch(model, train_loader, mean, std, optimizer)
        val_loss, val_mae_ev = run_epoch(model, val_loader, mean, std, optimizer=None)

        scheduler.step(val_mae_ev)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_ev"].append(val_mae_ev)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        improved = val_mae_ev < best_val_mae
        if improved:
            best_val_mae = val_mae_ev
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "epoch": epoch,
                    "val_mae_ev": val_mae_ev,
                },
                config.BEST_MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1

        # Always save the latest full training state, every epoch, so a
        # Ctrl+C / crash / sleep never costs more than one epoch of progress.
        save_latest_checkpoint(
            model, optimizer, scheduler, epoch, best_val_mae,
            epochs_without_improvement, history, mean, std
        )

        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_MAE={val_mae_ev:.4f} eV | lr={current_lr:.2e} | "
            f"{epoch_time:.1f}s{marker}"
        )

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping: no improvement for {config.EARLY_STOPPING_PATIENCE} epochs.")
            break

    with open(config.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    print(f"Training complete. Best val MAE: {best_val_mae:.4f} eV")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")
    print(f"Training history saved to: {config.TRAINING_HISTORY_PATH}")
    print("=" * 60)

    return model, history, mean, std


if __name__ == "__main__":
    train()