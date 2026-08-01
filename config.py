"""
config.py — Central configuration for the QM9 HOMO-LUMO Gap MPNN project.
"""

import os
import torch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "QM9")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

for _dir in (DATA_ROOT, CHECKPOINT_DIR, RESULTS_DIR):
    os.makedirs(_dir, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_INDEX = 4
TARGET_NAME = "HOMO-LUMO gap"
HARTREE_TO_EV = 27.211386245988

TRAIN_RATIO = 0.85
VAL_RATIO = 0.075
TEST_RATIO = 0.075
SPLIT_SEED = 42

NODE_FEATURE_DIM = 11
EDGE_FEATURE_DIM = 4
HIDDEN_DIM = 64
NUM_MESSAGE_PASSING_STEPS = 3
SET2SET_PROCESSING_STEPS = 3
DROPOUT = 0.0

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 100
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 15
GRAD_CLIP_NORM = 5.0

TORCH_SEED = 42


if __name__ == "__main__":
    print("=" * 60)
    print("QM9 MPNN — Configuration Summary")
    print("=" * 60)
    print(f"Device:                 {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU:                    {torch.cuda.get_device_name(0)}")
    print(f"Project root:           {PROJECT_ROOT}")
    print(f"Data root:              {DATA_ROOT}")
    print(f"Checkpoint dir:         {CHECKPOINT_DIR}")
    print(f"Results dir:            {RESULTS_DIR}")
    print("-" * 60)
    print(f"Target property:        {TARGET_NAME} (QM9 column index {TARGET_INDEX})")
    print(f"Hartree -> eV factor:   {HARTREE_TO_EV}")
    print("-" * 60)
    print(f"Split ratios:           train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print(f"Split seed:             {SPLIT_SEED}")
    print("-" * 60)
    print(f"Node feature dim:       {NODE_FEATURE_DIM}")
    print(f"Edge feature dim:       {EDGE_FEATURE_DIM}")
    print(f"Hidden dim:             {HIDDEN_DIM}")
    print(f"Message passing steps:  {NUM_MESSAGE_PASSING_STEPS}")
    print(f"Set2Set steps:          {SET2SET_PROCESSING_STEPS}")
    print("-" * 60)
    print(f"Batch size:             {BATCH_SIZE}")
    print(f"Learning rate:          {LEARNING_RATE}")
    print(f"Num epochs:             {NUM_EPOCHS}")
    print(f"Early stopping patience:{EARLY_STOPPING_PATIENCE}")
    print("=" * 60)