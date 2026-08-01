"""
model.py — Message Passing Neural Network (MPNN) for QM9 property prediction.

Architecture follows Gilmer et al. (2017), "Neural Message Passing for
Quantum Chemistry" (https://arxiv.org/abs/1704.01212) — the paper that
established QM9 as a standard GNN benchmark and against which we compare
our final results.

Components:
  1. Node embedding:  Linear(NODE_FEATURE_DIM -> HIDDEN_DIM)
  2. Edge network:     MLP(EDGE_FEATURE_DIM -> HIDDEN_DIM * HIDDEN_DIM),
                       used by NNConv to produce a per-edge weight matrix
                       conditioned on bond type (single/double/triple/aromatic).
  3. Message passing:  NUM_MESSAGE_PASSING_STEPS rounds of NNConv followed
                       by a GRU cell update (recurrent node-state update,
                       as in the original paper — more stable than a plain
                       MLP update across many message-passing steps).
  4. Readout:          Set2Set (LSTM-based attention pooling over atoms),
                       chosen over mean/sum pooling because the original
                       MPNN paper showed it measurably improves QM9 accuracy.
  5. Output head:      MLP(2 * HIDDEN_DIM -> HIDDEN_DIM -> 1)

Run this file directly to build the model, print its parameter count, and
verify a forward + backward pass on a synthetic batch matching QM9's shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set

import config


class MPNN(nn.Module):
    def __init__(
        self,
        node_feature_dim=config.NODE_FEATURE_DIM,
        edge_feature_dim=config.EDGE_FEATURE_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_message_passing_steps=config.NUM_MESSAGE_PASSING_STEPS,
        set2set_steps=config.SET2SET_PROCESSING_STEPS,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = num_message_passing_steps

        # 1. Node embedding: raw atom features -> hidden representation
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
        )

        # 2. Edge network: bond-type features -> a full (hidden x hidden)
        #    weight matrix used by NNConv to condition the message on bond type.
        edge_network = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim * hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim // 2, hidden_dim * hidden_dim),
        )

        # 3. Message passing layer (shared weights across all T steps, as
        #    in the original MPNN paper) + GRU for the recurrent node update.
        self.conv = NNConv(hidden_dim, hidden_dim, edge_network, aggr="mean")
        self.gru = nn.GRU(hidden_dim, hidden_dim)

        # 4. Set2Set readout: produces a graph-level embedding of size
        #    2 * hidden_dim (concatenation of query and attended context).
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps)

        # 5. Output head: graph embedding -> scalar property prediction.
        self.output_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Node embedding. GRU expects a (seq_len=1, batch, hidden) hidden
        # state, so we keep an explicit `h` tensor across message-passing steps.
        h = self.node_embedding(x)          # [num_nodes, hidden_dim]
        h_gru = h.unsqueeze(0)               # [1, num_nodes, hidden_dim]

        for _ in range(self.num_message_passing_steps):
            m = F.relu(self.conv(h, edge_index, edge_attr))  # messages
            m = m.unsqueeze(0)                                # [1, num_nodes, hidden_dim]
            h_out, h_gru = self.gru(m, h_gru)
            h = h_out.squeeze(0)

        # Graph-level readout across all atoms belonging to each molecule.
        graph_embedding = self.set2set(h, batch)  # [num_graphs, 2 * hidden_dim]

        out = self.output_head(graph_embedding)   # [num_graphs, 1]
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("MPNN Model — Self Test")
    print("=" * 60)

    torch.manual_seed(config.TORCH_SEED)

    model = MPNN().to(config.DEVICE)
    print(model)
    print("-" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("-" * 60)

    # Build a synthetic batch matching QM9's exact tensor shapes so we can
    # verify the forward/backward pass without needing to load the full
    # dataset here (dataset.py already verified the real data separately).
    from torch_geometric.data import Data, Batch

    def make_fake_molecule(num_atoms, num_bonds):
        x = torch.randn(num_atoms, config.NODE_FEATURE_DIM)
        edge_index = torch.randint(0, num_atoms, (2, num_bonds))
        edge_attr = F.one_hot(
            torch.randint(0, config.EDGE_FEATURE_DIM, (num_bonds,)),
            num_classes=config.EDGE_FEATURE_DIM,
        ).float()
        y = torch.randn(1, 1)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    fake_molecules = [make_fake_molecule(9, 16), make_fake_molecule(12, 22), make_fake_molecule(7, 10)]
    batch = Batch.from_data_list(fake_molecules).to(config.DEVICE)

    print(f"Synthetic batch: {batch}")

    # Forward pass
    model.train()
    predictions = model(batch)
    print(f"Output shape: {predictions.shape} (expected: [{len(fake_molecules)}, 1])")
    assert predictions.shape == (len(fake_molecules), 1), "Output shape mismatch!"

    # Backward pass — confirm gradients flow through the entire network.
    loss = F.mse_loss(predictions, batch.y)
    loss.backward()

    total_grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"Loss: {loss.item():.4f}")
    print(f"Total gradient norm across all parameters: {total_grad_norm:.4f}")
    assert total_grad_norm > 0, "No gradients flowed — check the computation graph!"

    print("-" * 60)
    print("Forward + backward pass verified successfully.")
    print("=" * 60)