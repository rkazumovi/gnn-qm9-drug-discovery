# Full QM9 target index reference — verified directly against PyG's
# qm9.py source (the docstring table in the QM9 class), which is the only
# authoritative source since PyG applies its own unit conversion during
# processing (raw Hartree/kcal-mol columns are converted to eV internally
# before the dataset is ever cached to disk):
#   0: mu (dipole moment, Debye)
#   1: alpha (isotropic polarizability, Bohr^3)
#   2: homo (eV — already converted from Hartree by PyG)
#   3: lumo (eV — already converted from Hartree by PyG)
#   4: gap = lumo - homo (eV — already converted from Hartree by PyG)  <-- target
#   5: r2 (electronic spatial extent, Bohr^2)
#   6: zpve (eV — already converted from Hartree by PyG)
#   7: u0 (internal energy at 0K, eV — already converted)
#   8: u298 (internal energy at 298.15K, eV — already converted)
#   9: h298 (enthalpy at 298.15K, eV — already converted)
#   10: g298 (free energy at 298.15K, eV — already converted)
#   11: cv (heat capacity at 298.15K, cal/mol*K)
#   12-15: atomization-energy variants of u0/u298/h298/g298 (eV — already converted)
#   16: A, 17: B, 18: C (rotational constants, GHz)
#
# IMPORTANT: PyG's QM9 loader already converts Hartree/kcal-mol columns to
# eV internally (see `conversion` tensor in qm9.py). Do NOT re-multiply
# data.y by HARTREE_TO_EV — that would double-convert an already-eV value.
TARGET_INDEX = 4
TARGET_NAME = "HOMO-LUMO gap"

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

import config


def load_qm9():
    """Load the raw QM9 dataset via PyG (downloads/caches automatically)."""
    dataset = QM9(root=config.DATA_ROOT)
    return dataset


def extract_target_in_ev(data):
    """
    Given a PyG Data object from QM9, return the HOMO-LUMO gap in eV as a
    scalar tensor, and overwrite data.y with just this single value.

    IMPORTANT: PyG's QM9 loader already converts this column from Hartree
    to eV internally during dataset processing (see the `conversion` tensor
    in torch_geometric/datasets/qm9.py). data.y[0, TARGET_INDEX] is already
    in eV — do NOT multiply by HARTREE_TO_EV here, or you'll double-convert.
    """
    gap_ev = data.y[0, config.TARGET_INDEX].clone()
    data.y = gap_ev.view(1, 1)
    return data


def make_splits(dataset):
    """
    Deterministically shuffle and split the dataset indices into
    train/val/test according to config ratios and config.SPLIT_SEED.

    Returns three PyG Subset-like index lists (via dataset.index_select),
    which behave like independent datasets for DataLoader purposes.
    """
    num_samples = len(dataset)

    generator = torch.Generator().manual_seed(config.SPLIT_SEED)
    perm = torch.randperm(num_samples, generator=generator)

    train_end = int(config.TRAIN_RATIO * num_samples)
    val_end = train_end + int(config.VAL_RATIO * num_samples)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    train_set = dataset[train_idx]
    val_set = dataset[val_idx]
    test_set = dataset[test_idx]

    return train_set, val_set, test_set


def compute_normalization_stats(train_set):
    """
    Compute mean and std of the HOMO-LUMO gap (in eV) across the TRAINING
    SET ONLY. These stats are later used to standardize targets for
    training stability (MPNNs, like most neural nets, train more reliably
    on roughly zero-mean, unit-variance targets) and to invert predictions
    back to physical units (eV) at evaluation time.
    """
    gaps_ev = torch.tensor(
    [d.y[0, config.TARGET_INDEX].item() for d in train_set]
)
    mean = gaps_ev.mean().item()
    std = gaps_ev.std().item()
    return mean, std


class NormalizedQM9(torch.utils.data.Dataset):
    """
    Thin wrapper around a QM9 subset that, on each __getitem__ call:
      - extracts the HOMO-LUMO gap target in eV
      - standardizes it using precomputed (mean, std) from the training set

    The graph structure (x, edge_index, edge_attr, pos, z) is left untouched;
    only data.y is replaced with the standardized scalar target.
    """

    def __init__(self, subset, mean, std):
        self.subset = subset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data = self.subset[idx].clone()
        data = extract_target_in_ev(data)
        data.y = (data.y - self.mean) / self.std
        return data


def get_dataloaders(batch_size=None):
    """
    Single entry point for the rest of the project.

    Returns:
        train_loader, val_loader, test_loader, mean, std

    mean/std are the training-set HOMO-LUMO gap statistics (in eV), needed
    by train.py (for reference) and evaluate.py (to de-normalize predictions
    back into eV for reporting MAE in physically meaningful units).
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    dataset = load_qm9()
    train_set, val_set, test_set = make_splits(dataset)

    mean, std = compute_normalization_stats(train_set)

    train_norm = NormalizedQM9(train_set, mean, std)
    val_norm = NormalizedQM9(val_set, mean, std)
    test_norm = NormalizedQM9(test_set, mean, std)

    train_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_norm, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_norm, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, mean, std


if __name__ == "__main__":
    print("=" * 60)
    print("QM9 Dataset Pipeline — Self Test")
    print("=" * 60)

    torch.manual_seed(config.TORCH_SEED)

    dataset = load_qm9()
    print(f"Total molecules in QM9 (post-sanitization): {len(dataset)}")

    train_set, val_set, test_set = make_splits(dataset)
    print(f"Train / Val / Test sizes: {len(train_set)} / {len(val_set)} / {len(test_set)}")

    mean, std = compute_normalization_stats(train_set)
    print(f"HOMO-LUMO gap (train set) — mean: {mean:.4f} eV, std: {std:.4f} eV")

    train_loader, val_loader, test_loader, mean, std = get_dataloaders()

    batch = next(iter(train_loader))
    print("-" * 60)
    print("Sample batch from train_loader:")
    print(batch)
    print(f"Batch target (normalized) — mean: {batch.y.mean().item():.4f}, "
          f"std: {batch.y.std().item():.4f}")

    # Sanity check: de-normalize the batch targets and confirm they fall in
    # a physically reasonable HOMO-LUMO gap range (QM9 molecules typically
    # span roughly 0-15 eV, with most mass between 4-10 eV).
    denorm = batch.y * std + mean
    print(f"Batch target (de-normalized, eV) — min: {denorm.min().item():.4f}, "
          f"max: {denorm.max().item():.4f}")
    print("=" * 60)