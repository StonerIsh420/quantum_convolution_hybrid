import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

# =========================================================
# 0. High-level configuration
# =========================================================

N_MOLECULES = 500         # how many QM9 molecules to use
BATCH_SIZE = 1            # graphs per batch (1 keeps the quantum call simple)
EPOCHS = 3
NUM_SEEDS = 3

NUM_HOPS = 1              # k-hop neighbourhood for each atom
MAX_PATCH_SIZE = 4        # number of neighbours per patch

N_QUBITS = 4              # quantum kernel width (also patch_dim after compression)
CLASS_HIDDEN = 4          # hidden width in classical kernel (gives it *more* params)

LR = 5e-3                 # shared learning rate

DEVICE = torch.device("cpu")

# where to save figures for the README
FIG_DIR = os.path.join("results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# =========================================================
# 1. Data: QM9 subset and patch extraction
# =========================================================

print("Loading QM9 Dataset (this may take a moment)...")
dataset = QM9(root="/tmp/QM9")
dataset = dataset[:N_MOLECULES]
num_features = dataset.num_features
print(f"QM9 loaded. Molecules: {len(dataset)}")
print(f"Features per atom: {num_features}")
print("Target: Predicting dipole moment (y[:, 0])")

# simple 80 / 10 / 10 split by index
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

train_set = dataset[:n_train]
val_set = dataset[n_train:n_train + n_val]
test_set = dataset[n_train + n_val:]


def make_loader(ds, shuffle):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = make_loader(train_set, shuffle=True)
val_loader = make_loader(val_set, shuffle=False)
test_loader = make_loader(test_set, shuffle=False)


def extract_patches(data, num_hops=NUM_HOPS, max_patch_size=MAX_PATCH_SIZE):
    """Turn a molecular graph into per-atom patches.

    Output shape: [num_nodes, max_patch_size, num_features]
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    x = data.x

    patches = torch.zeros((num_nodes, max_patch_size, x.size(1)), dtype=x.dtype)

    for node_idx in range(num_nodes):
        subset, _, _, _ = k_hop_subgraph(
            node_idx, num_hops, edge_index, relabel_nodes=False
        )
        patch_features = x[subset]
        cur = patch_features.size(0)
        if cur > max_patch_size:
            patches[node_idx] = patch_features[:max_patch_size]
        else:
            patches[node_idx, :cur] = patch_features
    return patches


# =========================================================
# 2. Quantum autograd wrapper (parameter-shift on weights only)
# =========================================================

class QuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data, weights, estimator, qc, observable):
        # input_data: [B_flat, patch_dim]
        ctx.estimator = estimator
        ctx.qc = qc
        ctx.observable = observable
        ctx.save_for_backward(input_data, weights)

        x_np = input_data.detach().cpu().numpy()
        w_np = weights.detach().cpu().numpy()

        batch_size = x_np.shape[0]
        # concatenate data + weights into a single parameter vector for each job
        all_params = np.hstack([x_np, np.tile(w_np, (batch_size, 1))])

        job = estimator.run([(qc, observable, all_params)])
        result = job.result()[0].data.evs
        return torch.tensor(result, dtype=torch.float32, device=input_data.device)

    @staticmethod
    def backward(ctx, grad_output):
        input_data, weights = ctx.saved_tensors
        estimator = ctx.estimator
        qc = ctx.qc
        observable = ctx.observable

        x_np = input_data.detach().cpu().numpy()
        w_np = weights.detach().cpu().numpy()
        batch_size = x_np.shape[0]

        shift = np.pi / 2
        grads = []

        # parameter-shift on each circuit weight
        for i in range(len(w_np)):
            w_pos = w_np.copy()
            w_neg = w_np.copy()
            w_pos[i] += shift
            w_neg[i] -= shift

            p_pos = np.hstack([x_np, np.tile(w_pos, (batch_size, 1))])
            p_neg = np.hstack([x_np, np.tile(w_neg, (batch_size, 1))])

            res_p = estimator.run([(qc, observable, p_pos)]).result()[0].data.evs
            res_m = estimator.run([(qc, observable, p_neg)]).result()[0].data.evs

            grads.append(
                torch.tensor((res_p - res_m) / 2,
                             dtype=torch.float32,
                             device=input_data.device)
            )

        grad_w = torch.stack(grads, dim=1)       # [B_flat, n_weights]

        # weights are shared, so we sum gradient contributions across the batch
        grad_w = (grad_output.unsqueeze(1) * grad_w).sum(dim=0)

        # no gradient wrt input_data (we treat compressor as fixed encoder)
        return None, grad_w, None, None, None


# =========================================================
# 3. Models: Hybrid quantum kernel vs classical kernel
# =========================================================

class HybridRegressor(nn.Module):
    """Graph-level regressor with a shared quantum convolution kernel.

    Compressor (atomic features -> 1 scalar) is treated as a fixed encoder:
    gradients do not flow back through it, which makes the comparison cleaner.
    """
    def __init__(self, n_feats: int):
        super().__init__()
        self.compressor = nn.Linear(n_feats, 1)
        # freeze compressor so it is not counted as trainable degrees of freedom
        for p in self.compressor.parameters():
            p.requires_grad = False

        # 4-qubit variational kernel with TWO entangling layers
        self.qc = QuantumCircuit(N_QUBITS)
        self.input_params = ParameterVector("x", N_QUBITS)
        self.weight_params_1 = ParameterVector("w1", N_QUBITS)
        self.weight_params_2 = ParameterVector("w2", N_QUBITS)

        # encode patch scalars onto qubits
        for i in range(N_QUBITS):
            self.qc.ry(self.input_params[i], i)

        # first entangling layer
        for i in range(N_QUBITS):
            self.qc.cz(i, (i + 1) % N_QUBITS)
            self.qc.rx(self.weight_params_1[i], i)

        # second entangling layer (extra expressivity)
        for i in range(N_QUBITS):
            self.qc.cz(i, (i + 1) % N_QUBITS)
            self.qc.rx(self.weight_params_2[i], i)

        self.estimator = StatevectorEstimator()
        self.observable = SparsePauliOp.from_list([("Z" * N_QUBITS, 1.0)])

        n_weights = len(self.weight_params_1) + len(self.weight_params_2)
        self.q_weights = nn.Parameter(torch.rand(n_weights) * np.pi)

        # graph-level regressor (1 scalar per graph)
        self.regressor = nn.Linear(1, 1)

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        # x_patches: [B, num_nodes, max_patch_size, num_feats]
        bsz, num_nodes, patch_size, _ = x_patches.shape

        # [B, N, P, F] -> [B, N, P, 1]
        compressed = torch.tanh(self.compressor(x_patches))

        # flatten nodes for the quantum call: [B*N, P]
        flat = compressed.view(bsz * num_nodes, patch_size)

        q_out = QuantumFunction.apply(
            flat, self.q_weights, self.estimator, self.qc, self.observable
        )
        # [B*N] -> [B, N]
        node_feats = q_out.view(bsz, num_nodes)

        # global mean pooling over atoms -> [B, 1]
        graph_feat = node_feats.mean(dim=1, keepdim=True)
        return self.regressor(graph_feat)      # [B, 1]


class ClassicalRegressor(nn.Module):
    """Purely classical control model.

    We give this model a *larger* kernel+head than the hybrid, so any hybrid
    advantage is conservative.
    """
    def __init__(self, n_feats: int):
        super().__init__()
        self.compressor = nn.Linear(n_feats, 1)

        self.kernel = nn.Linear(1, CLASS_HIDDEN)
        self.regressor = nn.Linear(CLASS_HIDDEN, 1)

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        bsz, num_nodes, patch_size, _ = x_patches.shape

        x = torch.tanh(self.compressor(x_patches))      # [B, N, P, 1]
        k = torch.tanh(self.kernel(x))                  # [B, N, P, H]
        patch_feat = k.mean(dim=2)                      # [B, N, H]
        graph_feat = patch_feat.mean(dim=1)             # [B, H]
        return self.regressor(graph_feat)               # [B, 1]


# =========================================================
# 4. Training / evaluation utilities
# =========================================================

def count_parameters(model: nn.Module, exclude_prefix: str | None = None) -> int:
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if exclude_prefix is not None and name.startswith(exclude_prefix):
            continue
        total += p.numel()
    return total


def train_one_epoch(q_model, c_model, loader, q_opt, c_opt, criterion):
    q_model.train()
    c_model.train()

    q_losses = []
    c_losses = []

    for batch in loader:
        patches = extract_patches(batch).unsqueeze(0).to(DEVICE)
        target = batch.y[:, 0].unsqueeze(1).to(DEVICE)  # [B, 1]

        # quantum
        q_opt.zero_grad()
        out_q = q_model(patches)                        # [B, 1]
        loss_q = criterion(out_q, target)
        loss_q.backward()
        q_opt.step()
        q_losses.append(loss_q.item())

        # classical
        c_opt.zero_grad()
        out_c = c_model(patches)                        # [B, 1]
        loss_c = criterion(out_c, target)
        loss_c.backward()
        c_opt.step()
        c_losses.append(loss_c.item())

    return q_losses, c_losses


@torch.no_grad()
def eval_model(q_model, c_model, loader, criterion):
    q_model.eval()
    c_model.eval()

    q_losses = []
    c_losses = []

    for batch in loader:
        patches = extract_patches(batch).unsqueeze(0).to(DEVICE)
        target = batch.y[:, 0].unsqueeze(1).to(DEVICE)

        out_q = q_model(patches)
        out_c = c_model(patches)

        q_losses.append(criterion(out_q, target).item())
        c_losses.append(criterion(out_c, target).item())

    return float(np.mean(q_losses)), float(np.mean(c_losses))


def smooth(y, window):
    if len(y) < window:
        return np.array(y)
    box = np.ones(window) / window
    return np.convolve(y, box, mode="same")


# =========================================================
# 5. Multi-seed experiment
# =========================================================

criterion = nn.MSELoss()

q_all_steps = []
c_all_steps = []

val_q_all = []
val_c_all = []
test_q_all = []
test_c_all = []

for seed in range(NUM_SEEDS):
    print(f"\n=== Seed {seed} ===")
    torch.manual_seed(seed)
    np.random.seed(seed)

    q_model = HybridRegressor(num_features).to(DEVICE)
    c_model = ClassicalRegressor(num_features).to(DEVICE)

    q_opt = optim.Adam(q_model.parameters(), lr=LR)
    c_opt = optim.Adam(c_model.parameters(), lr=LR)

    # trainable kernel+head params
    kernel_head_q = count_parameters(q_model)  # compressor frozen
    kernel_head_c = count_parameters(c_model, exclude_prefix="compressor")

    print(f"HybridRegressor trainable params (kernel+head):   {kernel_head_q}")
    print(f"ClassicalRegressor trainable params (kernel+head): {kernel_head_c}")

    for epoch in range(1, EPOCHS + 1):
        print(f"--- Training epoch {epoch} ---")

        q_steps, c_steps = train_one_epoch(
            q_model, c_model, train_loader, q_opt, c_opt, criterion
        )

        val_q, val_c = eval_model(q_model, c_model, val_loader, criterion)
        test_q, test_c = eval_model(q_model, c_model, test_loader, criterion)

        print(
            f"[Seed {seed} | Epoch {epoch}] "
            f"val_Q = {val_q:.4f}, val_C = {val_c:.4f} | "
            f"test_Q = {test_q:.4f}, test_C = {test_c:.4f}"
        )

        q_all_steps.append(q_steps)
        c_all_steps.append(c_steps)

        val_q_all.append(val_q)
        val_c_all.append(val_c)
        test_q_all.append(test_q)
        test_c_all.append(test_c)

# =========================================================
# 6. Aggregate and plot (and save figures)
# =========================================================

q_all_steps = np.array(q_all_steps)  # [NUM_SEEDS * EPOCHS, n_train_steps]
c_all_steps = np.array(c_all_steps)

mean_q = q_all_steps.mean(axis=0)
mean_c = c_all_steps.mean(axis=0)

plt.figure(figsize=(10, 5))
plt.plot(
    smooth(mean_q, 20),
    color="blue",
    linewidth=2,
    label="Quantum Hybrid (train MSE, mean over seeds)",
)
plt.plot(
    smooth(mean_c, 20),
    color="red",
    linestyle="--",
    linewidth=2,
    label="Classical Baseline (train MSE, mean over seeds)",
)
plt.xlabel("Training molecules (steps)")
plt.ylabel("Mean squared error (dipole moment)")
plt.title("QM9 Regression: Quantum vs Classical (train MSE, averaged over seeds)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
train_fig_path = os.path.join(FIG_DIR, "qm9_train_curve.png")
plt.savefig(train_fig_path, dpi=200)
plt.show()

# reshape per-epoch metrics: [NUM_SEEDS, EPOCHS]
val_q_arr = np.array(val_q_all).reshape(NUM_SEEDS, EPOCHS)
val_c_arr = np.array(val_c_all).reshape(NUM_SEEDS, EPOCHS)
test_q_arr = np.array(test_q_all).reshape(NUM_SEEDS, EPOCHS)
test_c_arr = np.array(test_c_all).reshape(NUM_SEEDS, EPOCHS)

val_q_mean = val_q_arr.mean(axis=0)
val_q_std = val_q_arr.std(axis=0)
val_c_mean = val_c_arr.mean(axis=0)
val_c_std = val_c_arr.std(axis=0)

test_q_mean = test_q_arr.mean(axis=0)
test_q_std = test_q_arr.std(axis=0)
test_c_mean = test_c_arr.mean(axis=0)
test_c_std = test_c_arr.std(axis=0)

epochs = np.arange(1, EPOCHS + 1)

plt.figure(figsize=(10, 5))
plt.errorbar(
    epochs,
    val_q_mean,
    yerr=val_q_std,
    fmt="-o",
    capsize=4,
    label="Quantum Hybrid (val MSE)",
)
plt.errorbar(
    epochs,
    val_c_mean,
    yerr=val_c_std,
    fmt="--o",
    capsize=4,
    label="Classical Baseline (val MSE)",
)
plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("Validation MSE over epochs (mean ± std over seeds)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
val_fig_path = os.path.join(FIG_DIR, "qm9_val_epoch.png")
plt.savefig(val_fig_path, dpi=200)
plt.show()

plt.figure(figsize=(10, 5))
plt.errorbar(
    epochs,
    test_q_mean,
    yerr=test_q_std,
    fmt="-o",
    capsize=4,
    label="Quantum Hybrid (test MSE)",
)
plt.errorbar(
    epochs,
    test_c_mean,
    yerr=test_c_std,
    fmt="--o",
    capsize=4,
    label="Classical Baseline (test MSE)",
)
plt.xlabel("Epoch")
plt.ylabel("Test MSE")
plt.title("Test MSE over epochs (mean ± std over seeds)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
test_fig_path = os.path.join(FIG_DIR, "qm9_test_epoch.png")
plt.savefig(test_fig_path, dpi=200)
plt.show()
