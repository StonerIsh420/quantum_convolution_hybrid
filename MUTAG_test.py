import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader
# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

# --- 1. Setup Data ---
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG').shuffle()
# Use 50 graphs for the proof plot
train_loader = DataLoader(dataset[:50], batch_size=1, shuffle=True)

def extract_patches(data, num_hops=1, max_patch_size=4):
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    x = data.x
    patch_tensor = torch.zeros((num_nodes, max_patch_size, x.size(1)))
    for node_idx in range(num_nodes):
        subset, _, _, _ = k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False)
        patch_features = x[subset]
        current_size = patch_features.size(0)
        if current_size > max_patch_size:
            patch_tensor[node_idx] = patch_features[:max_patch_size]
        else:
            patch_tensor[node_idx, :current_size] = patch_features
    return patch_tensor

# --- 2. Define Models ---

class QuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data, weights, estimator, qc, observable):
        ctx.estimator = estimator; ctx.qc = qc; ctx.observable = observable
        ctx.save_for_backward(input_data, weights)
        input_np = input_data.detach().numpy()
        weights_np = weights.detach().numpy()
        batch_size = input_np.shape[0]
        all_params = np.hstack([input_np, np.tile(weights_np, (batch_size, 1))])
        job = estimator.run([(qc, observable, all_params)])
        result = job.result()[0].data.evs
        return torch.tensor(result, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        input_data, weights = ctx.saved_tensors
        shift = np.pi / 2
        weights_np = weights.detach().numpy()
        input_np = input_data.detach().numpy()
        grads = []
        # Simple finite difference gradient approximation
        for i in range(len(weights_np)):
            w_pos = weights_np.copy(); w_pos[i] += shift
            w_neg = weights_np.copy(); w_neg[i] -= shift
            p_pos = np.hstack([input_np, np.tile(w_pos, (input_np.shape[0], 1))])
            p_neg = np.hstack([input_np, np.tile(w_neg, (input_np.shape[0], 1))])
            res_p = ctx.estimator.run([(ctx.qc, ctx.observable, p_pos)]).result()[0].data.evs
            res_m = ctx.estimator.run([(ctx.qc, ctx.observable, p_neg)]).result()[0].data.evs
            grads.append(torch.tensor((res_p - res_m) / 2, dtype=torch.float32))
        grad_weights = torch.stack(grads, dim=1)
        final_grad = torch.matmul(grad_output.unsqueeze(0), grad_weights).squeeze(0)
        return None, final_grad, None, None, None

class HybridGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.compressor = nn.Linear(7, 1)
        self.qc = QuantumCircuit(4)
        self.params = ParameterVector('x', 4); self.weights = ParameterVector('w', 4)
        for i in range(4): self.qc.ry(self.params[i], i)
        for i in range(4): self.qc.cz(i, (i+1)%4); self.qc.rx(self.weights[i], i)
        self.est = StatevectorEstimator(); self.obs = SparsePauliOp.from_list([("ZZZZ", 1)])
        self.q_weights = nn.Parameter(torch.rand(4) * np.pi)
        self.classifier = nn.Linear(1, 2)
    
    def forward(self, x):
        batch, nodes, patch, _ = x.shape
        comp = torch.tanh(self.compressor(x)).view(batch*nodes, patch)
        q_out = QuantumFunction.apply(comp, self.q_weights, self.est, self.qc, self.obs)
        # Reshape and Pool: [Batch, Nodes] -> [Batch, 1]
        return self.classifier(q_out.view(batch, nodes).mean(dim=1, keepdim=True))

class ClassicalGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.compressor = nn.Linear(7, 1)
        self.linear_kernel = nn.Linear(1, 1) 
        self.classifier = nn.Linear(1, 2)
    def forward(self, x):
        # x shape: [Batch, Nodes, PatchSize, Features]
        batch, nodes, _, _ = x.shape
        comp = torch.tanh(self.compressor(x))
        
        # Classical Kernel & Mean Pooling
        # [B, N, P, 1] -> [B, N, 1] -> [B, 1, 1]
        c_out = torch.tanh(self.linear_kernel(comp)).mean(dim=2).mean(dim=1, keepdim=True)
        
        # FIX: Squeeze the extra dimension [B, 1, 1] -> [B, 1]
        c_out = c_out.squeeze(-1)
        
        return self.classifier(c_out)

# --- 3. The Experiment Loop ---
q_model = HybridGNN(); c_model = ClassicalGNN()
q_opt = optim.Adam(q_model.parameters(), lr=0.02)
c_opt = optim.Adam(c_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

q_losses = []; c_losses = []

print("--- Running Comparison (2 Epochs) ---")
for epoch in range(2):
    for i, batch in enumerate(train_loader):
        patches = extract_patches(batch, num_hops=1, max_patch_size=4).unsqueeze(0)
        y = batch.y
        
        # Train Quantum
        q_opt.zero_grad()
        out_q = q_model(patches)
        loss_q = criterion(out_q, y.long())
        loss_q.backward()
        q_opt.step()
        q_losses.append(loss_q.item())
        
        # Train Classical
        c_opt.zero_grad()
        out_c = c_model(patches)
        loss_c = criterion(out_c, y.long())
        loss_c.backward()
        c_opt.step()
        c_losses.append(loss_c.item())
        
        if i % 10 == 0:
            print(f"Step {i} | Q-Loss: {loss_q.item():.3f} | C-Loss: {loss_c.item():.3f}")
            
    print(f"Epoch {epoch+1} Done")

# --- 4. Plotting ---
plt.figure(figsize=(10, 5))
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plt.plot(smooth(q_losses, 5), color='blue', linewidth=2, label='Quantum Hybrid')
plt.plot(smooth(c_losses, 5), color='red', linestyle='--', linewidth=2, label='Classical Baseline')
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("PhD Proof: Quantum vs Classical Convergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()