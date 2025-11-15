# Quantum Convolution Hybrid  
A Hybrid Quantum–Classical GNN Framework for Molecular Learning

This repository implements and evaluates a hybrid quantum–classical graph neural network (GNN) to test whether quantum convolution layers provide measurable benefits on chemically structured datasets such as MUTAG and QM9.

The core question:

> **Do quantum convolution layers extract chemically meaningful local structure as efficiently as, or more efficiently than, equivalently sized classical kernels?**

Two self-contained scripts make this entire project reproducible and easy to inspect:

- `MUTAG_test.py`
- `QM9_w_Graphing.py`

Everything runs with PyTorch, PyTorch Geometric, and Qiskit.

---

## 1. Repository Layout (Current State)

This is the real structure you have right now:

```text
quantum_convolution_hybrid/
│
├── MUTAG_test.py
├── QM9_w_Graphing.py
├── README.md
└── requirements.txt   # optional
```

Future structure (optional) is discussed later, but the README matches your current files exactly.

---

## 2. Research Motivation

We make a concrete, falsifiable hypothesis:

> **A small variational quantum convolution layer can match or slightly outperform a similarly sized classical convolution kernel on molecular graph tasks.**

Unlike typical QML demos that use MNIST or random data, this project uses chemically structured graphs, where correlations and local geometry matter.

Success criteria:

- Quantum kernel is competitive **without having more parameters** than the classical baseline.
- Quantum kernel shows **no gradient collapse** (barren plateau behaviour), i.e., it is trainable.
- Multiple seeds show **stable validation/test MSE** with meaningful error bars.

This is designed as credible scientific evidence, not hype.

---

## 3. Experiments

### 3.1 MUTAG — Classification Sanity Check

**Script:** `MUTAG_test.py`  
**Task:** Binary classification (mutagenic vs non-mutagenic).

Pipeline (high level):

1. Load MUTAG via PyTorch Geometric.
2. Extract fixed-size patches using `k_hop_subgraph`.
3. Compress node features → scalar using a small linear layer.
4. Compare:
   - **HybridGNN**: 4‑qubit variational quantum circuit as the convolution kernel.
   - **ClassicalGNN**: simple classical learnable kernel.

Purpose: ensure the quantum kernel is stable, differentiable, and does not collapse.  
This is a sanity check, not the main result.

---

### 3.2 QM9 — Main Regression Experiment (Dipole Moments)

**Script:** `QM9_w_Graphing.py`  
**Task:** Predict molecular dipole moment (QM9 target `y[:, 0]`).  
**Dataset subset:** first 500 molecules (fast simulation, still chemically meaningful).

Pipeline:

1. Load QM9 (PyTorch Geometric dataset).
2. Use the first 500 molecules.
3. For each molecule:
   - Extract k‑hop patches around each atom using `k_hop_subgraph`.
   - Cap patch size to a fixed `max_patch_size` for batching.
4. Compress high‑dimensional atom features to scalars.
5. Feed patches into two competing models:

#### HybridRegressor (Quantum)

- **Compressor**: linear layer from `num_features → 1` (optionally frozen to isolate the quantum head).
- **4‑qubit variational circuit**:
  - `Ry` input encoding from compressed patch scalars.
  - `CZ` entanglement ring.
  - `Rx` gates with trainable parameters.
- **Expectation value** computed via `StatevectorEstimator` using a Pauli observable (e.g., `ZZZZ`).
- Per‑node quantum outputs → mean pooled over nodes → scalar graph feature → linear regressor → predicted dipole moment.

The quantum circuit is connected to PyTorch via a custom `torch.autograd.Function` that uses finite‑difference parameter shifts for the trainable quantum weights.

#### ClassicalRegressor (Baseline)

- Same patch extraction and compressor as the quantum model.
- Replace the quantum circuit with a **classical kernel** (small linear/MLP layer).
- Same pooling strategy and scalar regression head.

#### Parameter Matching

The code prints **trainable parameter counts** for the kernel+head portion of each model.

In your final parameter‑matched setup, you ran with:

| Model      | Kernel + head trainable params |
|-----------|---------------------------------|
| Quantum   | 6                               |
| Classical | 13                              |

This makes the comparison conservative in favour of the classical baseline.

#### Multi‑Seed Experimental Protocol

`QM9_w_Graphing.py`:

- Trains for multiple epochs (e.g., 3).
- Uses multiple random seeds (e.g., 3).
- Records, for each model:
  - Per‑step training MSE.
  - Per‑epoch validation MSE (mean ± std over seeds).
  - Per‑epoch test MSE (mean ± std over seeds).
- Plots all of the above curves.

This gives a proper experimental picture instead of one cherry‑picked run.

---

## 4. Results (Representative Run)

Across 3 epochs on 500 molecules and 3 random seeds, you obtained:

### Validation MSE (Epoch 3, mean ± std)

- **Quantum:** ≈ 1.80 ± 0.05  
- **Classical:** ≈ 1.88 ± 0.16  

### Test MSE (Epoch 3, mean ± std)

- **Quantum:** ≈ 1.68 ± 0.06  
- **Classical:** ≈ 1.70 ± 0.02  

### Interpretation

- The quantum model is **competitive** with a larger classical kernel.
- No signs of barren plateaus or divergence.
- Differences are within error bars, but the quantum model does **not** underperform despite having fewer trainable parameters in the kernel+head.

This is exactly the kind of “credible, modest positive signal” you want to support the thesis that quantum convolution can be a viable building block on chemically structured data.

---

## 5. Figures (Where to Put Plots)

When your script generates plots, you can store them under:

```text
results/
└── figures/
    ├── qm9_train_curve.png
    ├── qm9_val_epoch.png
    ├── qm9_test_epoch.png
    └── qm9_param_matched.png
```

Then reference them in this README:

```markdown
## [QM9 Train MSE Curve]
![image](https://github.com/StonerIsh420/quantum_convolution_hybrid/blob/main/results/figures/qm9_train_curve.png?raw=true)

## [QM9 Validation MSE per Epoch]
![image](https://github.com/StonerIsh420/quantum_convolution_hybrid/blob/main/results/figures/qm9_val_epoch.png?raw=true)

## [QM9 Test MSE per Epoch]
![image](https://github.com/StonerIsh420/quantum_convolution_hybrid/blob/main/results/figures/qm9_test_epoch.png?raw=true)

## [QM9 Parameter‑Matched Comparison]
![image](./results/figures/qm9_param_matched.png)
```

If you have not saved the plots yet, you can add, in `QM9_w_Graphing.py`, right before each `plt.show()`:

```python
plt.savefig("results/figures/qm9_train_curve.png", dpi=300)
# or the appropriate filename per plot
```

Make sure the `results/figures` folders exist.

---

## 6. Installation

If you maintain a `requirements.txt`, a minimal version could look like:

```text
torch
torch-geometric
qiskit
numpy
matplotlib
```

### Install steps

```bash
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# Linux / macOS:
#   source .venv/bin/activate

pip install -r requirements.txt
```

> Note: PyTorch Geometric may require a specific install command depending on your OS/CUDA version. Check the official docs if a direct `pip install torch-geometric` fails.

---

## 7. Running the Experiments

### MUTAG Classification

```bash
python MUTAG_test.py
```

This will:

- Download MUTAG (if not already cached).
- Run the hybrid vs classical classification experiment.
- Print loss / accuracy logs.

### QM9 Regression

```bash
python QM9_w_Graphing.py
```

This will:

- Download QM9 on first run.
- Restrict to the first 500 molecules.
- Run multi‑seed, multi‑epoch quantum vs classical regression.
- Print per‑epoch validation and test MSE.
- Display (and optionally save) training/validation/test plots.

---

## 8. Design Choices & Limitations

- **4‑qubit kernel**: small enough for CPU statevector simulation; still capable of nontrivial entanglement.
- **Frozen compressor (optional)**: keeps the comparison focused on the quantum vs classical kernel rather than the encoder.
- **Conservative classical baseline**: the classical kernel+head has equal or more parameters than the quantum head.
- **Finite‑difference gradients**: slower than analytic parameter‑shift but extremely explicit and easy to reason about.
- **Subset of QM9 (500 molecules)**: chosen to keep Qiskit simulation tractable while operating on real chemical structure.

This is a research prototype, not a production ML stack.

---

## 9. Future Refactor (Optional)

If you later decide to modularise the code, a natural layout would be:

```text
quantum_convolution_hybrid/
│
├── models/
│   ├── hybrid_quantum_regressor.py
│   ├── classical_regressor.py
│   └── quantum_function.py
│
├── experiments/
│   ├── mutag_experiment.py
│   └── qm9_experiment.py
│
├── results/
│   └── figures/
│       ├── qm9_train_curve.png
│       ├── qm9_val_epoch.png
│       ├── qm9_test_epoch.png
│       └── qm9_param_matched.png
│
├── README.md
└── requirements.txt
```

For now, everything living in `MUTAG_test.py` and `QM9_w_Graphing.py` is perfectly fine for a prototype and for a paper appendix.

---

## 10. Citation

If you use this work in academic writing, a simple software citation is:

```text
@software{quantum_convolution_hybrid,
  title   = {Hybrid Quantum Convolution on Molecular Graphs},
  author  = {Stoner},
  year    = {2025}
}
```

---

## 11. Summary

This project:

- Provides a **real experimental benchmark** of quantum vs classical convolution on molecular graphs.
- Uses **matched or conservative** parameter counting for fair comparison.
- Includes **multi‑seed evaluation with error bars**, not just a single lucky run.
- Produces **reproducible plots** from small, readable Python scripts.
- Lives entirely in two files that can drop straight into a thesis or paper appendix.

It is a compact but serious scaffold for exploring quantum convolution on chemically structured data.
