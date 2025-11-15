# Quantum Convolution Hybrid: A Hybrid Quantum–Classical GNN Framework

This repository implements and evaluates a **hybrid quantum–classical graph neural network** designed to test whether **quantum convolution layers** provide measurable benefits on **chemically structured data**.

The project benchmarks a small, Qiskit-simulated quantum kernel against a matched classical baseline on two molecular datasets:

- **MUTAG** (binary classification, graph-level)
- **QM9** (regression: dipole moment prediction)

The goal is not to claim "quantum advantage", but to produce **controlled, reproducible experiments** that answer a precise question:

> Does a small variational quantum convolution layer capture chemically meaningful structure as well as — or better than — a similarly-sized classical kernel?

---

## Key Features

### ✔ Hybrid Quantum–Classical Architecture
- Patch extraction over graph neighborhoods using `k-hop` subgraphs.
- Linear compressor + 4-qubit quantum circuit using Qiskit’s `StatevectorEstimator`.
- Differentiable circuit evaluation via a custom PyTorch autograd function.

### ✔ Classical Baseline Matched by Parameter Count
- Same patch extraction and compressor.
- Purely classical learnable kernel with comparable or greater trainable parameters.
- Ensures a fair comparison between classical and hybrid models.

### ✔ Rigorous Experimental Protocol
- Multiple seeds (`seed = 0, 1, 2`).
- Per-step training MSE curves.
- Per-epoch validation and test MSE with error bars (mean ± std).
- Identical train/val/test splits for both models.
- Reproducibility utilities and deterministic seeds.

### ✔ Datasets
- **MUTAG:** tiny graph dataset for fast validation of the pipeline.
- **QM9 (500-molecule subset):** more chemically challenging regression target.

---

## Repository Structure

