# Quantum Convolution as a Feature Extractor for Non-Euclidean Data: A Hybrid Quantum–Classical Learning Architecture

## Abstract

Hybrid quantum–classical architectures are a promising route to extracting value from noisy, intermediate-scale quantum (NISQ) devices before fully fault-tolerant quantum computers become available. This work develops and analyzes a concrete, falsifiable claim:

> **Claim.** Quantum convolution layers provide a discernible advantage in feature extraction for non-Euclidean datasets compared to classical convolution.

We instantiate this claim in a realistic setting, where a classical convolutional neural network (CNN) operating on structured scientific data—such as molecular graphs or particle-collision event topologies—is augmented by replacing a central convolutional layer with a parametrized quantum variational circuit. The resulting hybrid model is implemented using Qiskit for circuit simulation and PyTorch for automatic differentiation and optimization.

On the theoretical side, we formalize **quantum convolution** as a family of shared-parameter unitary channels with local receptive fields. We show that, under mild assumptions on the data encoding and entangling pattern, such layers can represent families of nonlinear, permutation-equivariant feature maps whose classical approximation with standard convolutions requires either substantial depth or a super-constant blow-up in channel dimension.

On the empirical side (to be instantiated in follow-up work using the included implementation design), we propose a set of controlled experiments on benchmark non-Euclidean datasets. The key evaluation criteria are: (i) convergence speed, (ii) test accuracy at a fixed parameter budget, and (iii) robustness to parameter pruning. Our central hypothesis is that the hybrid quantum–classical network converges faster and attains higher accuracy than a purely classical baseline with a comparable number of trainable parameters, thereby providing an operational notion of **discernible advantage** well before asymptotic quantum supremacy is relevant.

This paper provides: (i) a precise formulation of quantum convolution for non-Euclidean data, (ii) a hybrid architecture design that can be implemented today using Qiskit and PyTorch, and (iii) a roadmap for demonstrating practically meaningful advantages from quantum feature extraction in a regime accessible to near-term hardware and high-performance classical simulation.

---

## 1. Introduction

Quantum machine learning (QML) explores how quantum information processing can enhance or accelerate learning tasks. Asymptotic complexity results show that quantum models can, in principle, represent functions that are intractable for efficient classical computation. However, these results are far from the scale and noise regime of present-day devices.

Current progress is therefore driven by **hybrid quantum–classical architectures**, in which small parametrized quantum circuits are integrated into otherwise classical learning pipelines. In parallel, many scientific and industrial problems involve **non-Euclidean** data: graphs of interacting particles, molecular structures, irregular meshes, and point clouds on manifolds. Generalized convolutional architectures—graph neural networks (GNNs), geometric deep learning models—have achieved strong performance on such data, but they face representational and optimization challenges when long-range correlations and entangled local patterns are central.

This paper targets the intersection of these fronts. We focus on the following working claim:

> **Claim.** Quantum convolution layers provide a discernible advantage in feature extraction for non-Euclidean datasets compared to classical convolution.

The goal is not to prove a full-blown complexity separation, but to make this claim technically meaningful, implementable, and empirically testable. Concretely, we:

1. Define a **quantum convolution layer** as a shared-parameter variational quantum circuit acting on local patches or neighborhoods of a structured input, equipped with a topology-aware encoding map and pooling/readout scheme.
2. Embed this quantum convolution layer into a PyTorch-based CNN as a drop-in replacement for a classical convolutional block, using Qiskit to simulate the quantum circuit and the parameter-shift rule to compute gradients.
3. Argue, using expressivity and inductive-bias considerations, that such layers can more compactly represent certain entangled local feature maps than classical convolutions with comparable parameter budgets.
4. Propose an experimental protocol to compare hybrid and classical models on realistic non-Euclidean datasets, with a staged evaluation from standard graph benchmarks (MUTAG, PROTEINS) to molecular property prediction and particle-physics event classification.

The remainder of this paper is organized as follows. Section 2 reviews key background on quantum computing, variational circuits, and non-Euclidean deep learning. Section 3 formulates the learning problem and baselines. Section 4 defines the quantum convolution layer. Section 5 describes the hybrid architecture and training loop. Section 6 develops a theoretical perspective on potential advantages. Section 7 outlines the experimental design. Section 8 discusses implementation details in Qiskit and PyTorch. Section 9 discusses limitations and extensions, and Section 10 concludes.

---

## 2. Background

### 2.1 Quantum Computation and Variational Circuits

A quantum state on \(n\) qubits is a complex vector \(|\psi\rangle\) in a \(2^n\)-dimensional Hilbert space with unit norm. Computation proceeds by applying unitary transformations and measuring in a chosen basis. In the NISQ regime, a prominent paradigm is the **variational quantum circuit** (VQC): a parametrized circuit

\[ U(\boldsymbol{\theta}) = U_L(\theta_L) \cdots U_2(\theta_2) U_1(\theta_1), \]

where each \(U_\ell(\theta_\ell)\) is either a parametrized single-qubit rotation or a fixed entangling gate.

Given an input encoding map \(E(\mathbf{x})\) that prepares a state \(|\psi(\mathbf{x})\rangle\), the model output is often an expectation value

\[ f(\mathbf{x}; \boldsymbol{\theta}) = \langle \psi(\mathbf{x}) | U(\boldsymbol{\theta})^\dagger O U(\boldsymbol{\theta}) | \psi(\mathbf{x}) \rangle, \]

for some observable \(O\). Training a VQC typically involves classical gradient-based optimization of a loss function \(C(\boldsymbol{\theta})\). For many standard gate families, gradients can be estimated with the **parameter-shift rule**:

\[ \frac{\partial C}{\partial \theta_k} = \frac{1}{2} \left[C(\boldsymbol{\theta} + \tfrac{\pi}{2} \mathbf{e}_k) - C(\boldsymbol{\theta} - \tfrac{\pi}{2} \mathbf{e}_k)\right], \]

where \(\mathbf{e}_k\) is the unit vector in direction \(k\). In practice, expectation values are estimated via repeated measurements (“shots”), introducing additional stochasticity and noise.

### 2.2 Quantum Machine Learning and Hybrid Architectures

Quantum machine learning explores models where some or all of the computation is carried out on quantum devices. In the NISQ setting, **hybrid** models dominate: small quantum circuits are interleaved with classical neural network components. Typical patterns include:

- Classical preprocessing, a quantum block, then classical postprocessing.
- Replacing a dense or convolutional layer with a quantum circuit that outputs learned features.
- Using quantum circuits to generate kernels or embeddings, which are then consumed by classical models.

The hybrid approach leverages the strengths of both substrates: the potentially rich representational capacity of quantum circuits and the robust, scalable optimization capabilities of classical deep learning.

### 2.3 Non-Euclidean Data and Convolution

Non-Euclidean data refers to inputs whose intrinsic structure is not aligned with a regular grid in \(\mathbb{R}^d\). Key examples include:

- **Graphs**: atoms and bonds in molecules, tracks and deposits in particle detectors, nodes and edges in social networks.
- **Manifolds**: surfaces or curved spaces discretized as meshes or point clouds.
- **Higher-order complexes**: hypergraphs, simplicial complexes, and other combinatorial structures.

Graph neural networks (GNNs) generalize convolution by defining message-passing or spectral operators that aggregate information from neighbors. For a graph \(G=(V, E)\) with node features \(\mathbf{x}_v\), a typical layer updates node \(v\) via

\[ \mathbf{h}_v^{(k+1)} = \sigma\left( W_1 \mathbf{h}_v^{(k)} + W_2 \sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{h}_u^{(k)} \right), \]

where \(\sigma\) is a nonlinearity and \(\alpha_{vu}\) encodes attention or normalization. Classical convolutions on regular grids arise as a special case.

### 2.4 Why Quantum Convolution?

Quantum circuits offer two properties that make them appealing as potential feature extractors for non-Euclidean data:

1. **Entanglement as structured correlation**: Entangling gates can encode higher-order correlations among local features without explicitly expanding tensor dimensions.
2. **Expressive local unitaries**: Even shallow circuits on a small number of qubits can represent highly nonlinear transformations of their inputs, playing a role analogous to deep stacks of classical layers but with fewer trainable parameters.

The central question is whether these properties can be turned into a concrete, trainable layer that plugs into modern deep learning architectures and yields measurable benefits on real structured data.

---

## 3. Problem Formulation

We consider supervised learning on a non-Euclidean dataset

\[ \mathcal{D} = \{(X_i, y_i)\}_{i=1}^N, \]

where each input \(X_i\) is a structured object, such as:

- A **molecular graph** \(X_i = (V_i, E_i, \mathbf{f}_i)\), with atoms as nodes, bonds as edges, and node/edge features.
- A **particle-collision event**, where \(X_i\) encodes detected tracks or calorimeter deposits arranged on an irregular lattice or graph.

The label \(y_i\) can be a scalar property (for regression) or a discrete class.

### 3.1 Classical Baseline Architecture

As a baseline, we consider a classical architecture with the following components:

1. An input encoder that maps raw inputs \(X\) to an initial feature representation \(\mathbf{h}^{(0)}\) on graph nodes or grid cells.
2. A stack of classical convolutional or graph-convolutional layers:

   \[ \mathbf{h}^{(0)} \mapsto \mathbf{h}^{(1)} \mapsto \cdots \mapsto \mathbf{h}^{(L_c)}. \]

3. A global pooling operation (e.g., sum, mean, or attention pooling) to obtain a fixed-size representation \(\mathbf{z}\).
4. Fully connected layers mapping \(\mathbf{z}\) to the output \(f_{\text{classical}}(X)\).

Parameter count, depth, and width are chosen subject to a fixed budget, to enable fair comparison with the hybrid model.

### 3.2 Hybrid Quantum–Classical Architecture

In the hybrid architecture, we replace an intermediate convolutional block with a **quantum convolution layer** \(Q_{\text{conv}}\). Schematically,

\[ X \mapsto \mathbf{h}^{(0)} \mapsto \mathbf{h}^{(1)} \mapsto Q_{\text{conv}}(\mathbf{h}^{(1)}) = \mathbf{h}^{(q)} \mapsto \mathbf{h}^{(L_c)} \mapsto \mathbf{z} \mapsto f_{\text{hybrid}}(X). \]

The quantum convolution operates on local neighborhoods (patches) extracted from \(\mathbf{h}^{(1)}\), encoding each patch into a small number of qubits, applying a shared parametrized quantum circuit \(U(\boldsymbol{\theta})\), and measuring expectation values as patch-level features.

The central comparative question is: **under a comparable parameter budget**, and using equivalent training procedures, does the hybrid model \(f_{\text{hybrid}}\) converge faster or achieve higher accuracy than \(f_{\text{classical}}\)?

---

## 4. Quantum Convolution Layer

We now define the quantum convolution layer in more detail.

### 4.1 Patch Extraction and Ordering

Let \(\mathbf{h}^{(1)}\) be an intermediate feature representation defined on nodes or grid cells. For each center node or cell \(i\), we define a local neighborhood \(\mathcal{N}_R(i)\) of radius \(R\) (graph distance or geometric distance). The patch features are

\[ P_i = \{ \mathbf{h}^{(1)}_j : j \in \mathcal{N}_R(i) \}. \]

We then fix an ordering scheme for the nodes in the patch (e.g., breadth-first traversal with deterministic tie-breaking rules) to obtain a sequence of feature vectors

\[ \mathbf{x}_{i,1}, \ldots, \mathbf{x}_{i,m}, \]

where \(m = |\mathcal{N}_R(i)|\) is fixed via padding or truncation. This sequence will be encoded into a small quantum register.

### 4.2 Data Encoding into Qubits

We choose an encoding map \(\mathcal{E}\) that maps patch features to a quantum state on \(n\) qubits. A simple and hardware-friendly approach is **angle encoding**:

1. Select \(d'\) features per node and flatten them across the patch.
2. Associate each scalar feature value \(x_{i,k,\ell}\) with a single qubit and apply a rotation

   \[ R_y(\phi_{i,k,\ell}) = \exp\left(-i \tfrac{\phi_{i,k,\ell}}{2} Y \right), \]

   where \(\phi_{i,k,\ell}\) is a scaled version of \(x_{i,k,\ell}\).

Starting from the all-zero state \(|0\rangle^{\otimes n}\), the encoded state is

\[ |\psi(P_i)\rangle = \prod_{k,\ell} R_y(\phi_{i,k,\ell}) |0\rangle^{\otimes n}. \]

Alternative encodings (e.g., amplitude encoding, more complex feature maps) are possible but may impose stronger requirements on circuit depth and numerical stability.

#### Topology-aware graph-to-qubit mapping

A critical design issue is that naïvely flattening a graph patch into a list of angles can discard the underlying topology. If node \(u\) is connected to node \(v\) in the graph but their associated qubits are not correspondingly entangled, the circuit may fail to reflect the actual interaction structure.

To address this, the quantum convolution explicitly treats **graph-to-qubit mapping** as part of the design space:

- The ordering of nodes within a patch is chosen so that **adjacent nodes in the graph are mapped to nearby qubits**, making it natural to entangle them.
- The entangling pattern (Section 4.3) is aligned with the local adjacency: for example, controlled-Z gates are placed along edges in \(\mathcal{N}_R(i)\), or along a traversal that respects the local graph structure.
- For irregular patches, we consider fixed “canonicalization” procedures (e.g., BFS ordering with degree-aware tie-breaking) so that structurally similar patches map to similar qubit layouts.

We explicitly acknowledge that encoding is a potential bottleneck: a poor graph-to-qubit mapping can destroy geometric inductive bias and make the quantum layer look like a generic MLP on angles. Part of the empirical work is therefore to compare topology-aware encodings against naïve flattening and quantify their impact on performance and parameter efficiency.

### 4.3 Shared-Parameter Variational Circuit

We define a parametrized circuit \(U(\boldsymbol{\theta})\) acting on \(n\) qubits, composed of \(L\) layers of single-qubit rotations and entangling gates:

\[ U(\boldsymbol{\theta}) = \prod_{\ell=1}^{L} \Big( U_{\text{ent}}^{(\ell)} \cdot U_{\text{rot}}^{(\ell)}(\boldsymbol{\theta}^{(\ell)}) \Big), \]

where:

- \(U_{\text{rot}}^{(\ell)}(\boldsymbol{\theta}^{(\ell)})\) is a product of single-qubit rotations, e.g.
  \[ U_{\text{rot}}^{(\ell)} = \bigotimes_{q=1}^{n} R_y(\theta^{(\ell)}_q), \]
- \(U_{\text{ent}}^{(\ell)}\) is a fixed pattern of entangling gates (e.g., controlled-Z gates) arranged along a chosen topology (line, ring, or local graph) over the \(n\) qubits.

Crucially, **the parameters \(\boldsymbol{\theta}\) are shared across all patches \(P_i\)**, mirroring how classical convolutional filters are shared across spatial locations. This is the quantum analogue of a convolution kernel: a small, expressive, reusable feature extractor applied to many local neighborhoods.

### 4.4 Measurement and Feature Aggregation

After encoding and applying the variational circuit, we measure a set of observables \(\{O_r\}_{r=1}^R\) to obtain a feature vector for the patch:

\[ f_r(P_i; \boldsymbol{\theta}) = \langle \psi(P_i) | U(\boldsymbol{\theta})^\dagger O_r U(\boldsymbol{\theta}) | \psi(P_i) \rangle, \quad r = 1, \dots, R. \]

In practice, each \(f_r\) is approximated by repeated measurements (“shots”) on the corresponding circuit. Aggregating across \(r\) yields a patch feature vector \(\mathbf{z}_i \in \mathbb{R}^R\). Applying this procedure to all centers \(i\) yields a transformed feature set \(\{\mathbf{z}_i\}\), which can be reshaped into a tensor compatible with subsequent classical layers.

### 4.5 Equivariance and Invariance Considerations

Non-Euclidean data often carry symmetries, such as permutation invariance within neighborhoods or approximate rotational invariance in physical systems. To respect these symmetries, we can design the encoding and observable sets such that:

- The chosen observables \(O_r\) are symmetric with respect to qubit permutations induced by isomorphisms of the patch.
- The entangling pattern follows the structure of the patch, e.g., mirroring graph edges or geometric adjacency.

In practice, perfect equivariance may be costly. Approximate strategies include averaging features over multiple orderings, or learning a small ensemble of circuits with tied parameters. Topology-aware graph-to-qubit mappings (above) are a first-order attempt to encode structure without requiring exact group-equivariant design.

---

## 5. Hybrid Architecture and Training

### 5.1 Network Structure

The hybrid network is structured as follows:

1. **Input encoder**: maps each raw input \(X\) to an initial feature representation \(\mathbf{h}^{(0)}\) defined on a graph or grid.
2. **Classical convolution block**: one or more layers mapping \(\mathbf{h}^{(0)} \to \mathbf{h}^{(1)}\), implemented with standard CNN or GNN layers.
3. **Quantum convolution block**: the quantum convolution layer maps \(\mathbf{h}^{(1)} \to \mathbf{h}^{(q)}\).
4. **Classical readout block**: additional classical layers map \(\mathbf{h}^{(q)} \to \mathbf{z} \to \(f_{\text{hybrid}}(X)\).

The classical blocks can be tailored to the domain (grid-based CNN for calorimeter images, message-passing for molecular graphs), while the quantum block provides a reusable, local, topology-aware feature extractor.

### 5.2 Integration with PyTorch and Qiskit

Practically, the quantum convolution layer is implemented as a PyTorch module with:

- A `forward` method that:
  - Receives a tensor of patch features.
  - For each patch (or batch of patches), constructs or reuses a Qiskit circuit encoding the patch and applying \(U(\boldsymbol{\theta})\).
  - Executes circuits on a Qiskit backend (statevector simulator for development, shot-based simulator or hardware for realistic conditions).
  - Computes expectation values of specified observables and returns a PyTorch tensor of patch features.

- A custom autograd mechanism—either via PyTorch’s `autograd.Function` or a dedicated interface—that uses the parameter-shift rule to estimate gradients of expectations with respect to \(\boldsymbol{\theta}\).

To manage computational cost, we batch patches, reuse circuit templates, and restrict ourselves to modest numbers of qubits and circuit depth.

### 5.3 Training Objective and Optimization

Let \(\mathcal{L}(f(X), y)\) be a suitable loss (e.g., cross-entropy for classification, mean squared error for regression). Over the dataset \(\mathcal{D}\), the training objective is:

\[ \mathcal{J}(\boldsymbol{\theta}, \boldsymbol{\phi}) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}\big(f_{\text{hybrid}}(X_i; \boldsymbol{\theta}, \boldsymbol{\phi}), y_i\big) + \lambda \, \Omega(\boldsymbol{\theta}, \boldsymbol{\phi}), \]

where:

- \(\boldsymbol{\theta}\) are the quantum layer parameters.
- \(\boldsymbol{\phi}\) are the classical layer parameters.
- \(\Omega\) is a regularizer (e.g., weight decay).
- \(\lambda\) balances regularization strength.

Optimization proceeds with stochastic gradient descent or its variants (Adam, RMSprop). Classical parameters \(\boldsymbol{\phi}\) are updated using standard backpropagation. Quantum parameters \(\boldsymbol{\theta}\) are updated using gradients estimated by the parameter-shift rule at the quantum layer.

---

## 6. Theoretical Perspective on Advantage

We now articulate what we mean by “discernible advantage” and how quantum convolution can, in principle, provide it.

### 6.1 Notions of Advantage

We adopt three operational criteria for advantage:

1. **Convergence advantage**: For a fixed training pipeline, the hybrid network reaches a specified validation performance threshold in fewer epochs or gradient steps than the classical baseline.
2. **Parameter-efficiency advantage**: For a fixed parameter budget, the hybrid network attains lower generalization error than any classical network in a defined reference family (e.g., CNNs or GNNs with bounded depth and width).
3. **Robustness advantage**: For a fixed performance level, the hybrid network is more robust to parameter pruning or injected noise.

Any of these, if established empirically under controlled conditions, constitutes a practical advantage.

### 6.2 Expressivity of Quantum Convolution

Informally, a quantum convolution layer with \(n\) qubits and \(L\) layers of parametrized gates can represent a family of functions whose expressive power grows rapidly with \(n\) and \(L\), while the number of trainable parameters may grow only linearly. Classical convolution layers, by contrast, are constructed from affine transformations and pointwise nonlinearities, with expressivity governed by channel count and depth.

**Informal statement (expressivity gap).** Consider local feature maps \(f: \mathbb{R}^m \to \mathbb{R}^R\) realized by constant-depth quantum circuits on \(n = O(m)\) qubits with \(O(n)\) parameters and a fixed entangling topology. Under mild assumptions on the encoding map, there exist families of such maps that cannot be approximated by any constant-depth classical convolution with \(O(n)\) channels to within small error, unless the classical network uses super-constant width or depth.

This type of statement is aligned with known expressivity separations between quantum and classical models. Translating it rigorously into the language of convolutions on non-Euclidean data is nontrivial, but the underlying intuition is that entanglement enables compact representation of higher-order local correlations.

### 6.3 Inductive Bias on Non-Euclidean Data

Non-Euclidean scientific data often exhibit complex local correlation patterns: aromatic rings in molecules, multi-particle interference structures in collider events, or higher-order motifs in interaction networks. Classical convolutions can represent these patterns in principle but may require deep stacks and wide channels.

Quantum convolution naturally computes expectation values of multi-qubit observables:

\[ f_r(P_i; \boldsymbol{\theta}) = \langle \psi(P_i) | O_r(\boldsymbol{\theta}) | \psi(P_i) \rangle, \]

where \(O_r(\boldsymbol{\theta}) = U(\boldsymbol{\theta})^\dagger O_r U(\boldsymbol{\theta})\) can act as a high-order, nonlinear feature detector over structured subsets of the patch.

We hypothesize that, for tasks where such high-order local structures are predictive, the quantum convolution layer provides a better inductive bias per parameter than standard convolutions, leading to faster convergence and higher accuracy under the same capacity constraints.

### 6.4 Limitations of the Theoretical Argument

Several limitations must be acknowledged:

- **Simulation regime**: Experiments run on classical hardware simulate quantum circuits. Any observed advantage reflects representational and optimization benefits, not asymptotic complexity gains.
- **Task dependence**: If the underlying task depends only on low-order statistics, quantum convolution may offer little or no benefit and could introduce unnecessary noise.
- **Barren plateaus**: Variational quantum circuits can suffer from vanishing gradients in high-dimensional parameter spaces.

The proposed architecture explicitly addresses the barren plateau issue in two ways:

1. **Small, shallow kernels**: The quantum layer is designed as a shallow circuit on a *small* number of qubits (e.g., 4–8) per patch, not as a single deep monolithic circuit over the whole system. This keeps the Hilbert space dimension modest for each optimization problem, which empirical and theoretical work suggests reduces the severity of barren plateaus.
2. **Shared parameters across patches**: The same small kernel \(U(\boldsymbol{\theta})\) is reused across many patches, exactly as in classical convolution. This “quantum kernel sharing” effectively concentrates optimization signal: each parameter update is informed by many local contexts, increasing gradient signal-to-noise and reducing the risk that gradients vanish globally.

In other words, the convolutional pattern (many applications of a small circuit) is not only a parameter-efficiency trick, but also a deliberate **anti–barren plateau design choice**. Nevertheless, the barren plateau risk cannot be eliminated outright and remains a key part of the empirical investigation.

Thus, the claim of advantage is inherently domain-dependent and must be evaluated empirically on tasks with rich local structure, using architectures that are specifically designed to avoid known optimization pathologies.

---

## 7. Experimental Design

To test the central claim, we propose a controlled empirical study comparing classical and hybrid models on realistic non-Euclidean datasets, starting from small, standardized graph benchmarks and scaling up to more complex scientific data.

### 7.1 Datasets

We adopt a **staged dataset strategy**:

1. **Hello-world graph benchmarks (sanity check and first results)**  
   - **MUTAG**: A small dataset of nitroaromatic compounds labeled by mutagenicity. Graph structure matters, but the dataset is small enough that simulation is tractable and baselines are well-understood.  
   - **PROTEINS**: A dataset of protein graphs classified by structural or functional properties. This provides a slightly more complex benchmark while remaining widely studied in the GNN literature.

   These datasets serve as the “hello-world” setting for quantum graph convolution. If the hybrid model can match or outperform classical baselines on MUTAG/PROTEINS under a strict parameter budget, that already constitutes a concrete, interpretable result.

2. **Molecular property prediction (scientific relevance)**  
   - Small-molecule datasets where target properties (e.g., energy levels, reactivity indicators) depend on local substructures and electronic interactions. Here, the focus shifts toward scientifically meaningful targets and the relationship between quantum features and quantum-chemical properties.

3. **Particle-physics event classification (high-complexity regime)**  
   - Datasets of collision events labeled by process type or the presence of particular signatures, with inputs represented as graphs or irregular calorimeter images. These probe the architecture in a regime with complex local and global correlations and offer a direct link to high-energy physics applications.

Datasets are chosen to be large enough to support meaningful generalization and contain nontrivial local structure, but not so large that quantum simulation becomes intractable. In each regime, the same protocol—classical vs hybrid vs capacity-matched classical—is applied.

### 7.2 Models and Baselines

We consider the following models:

1. **Classical CNN/GNN baseline**: A purely classical model with one or more convolutional or graph-convolutional layers.
2. **Hybrid (single quantum layer)**: The same as the baseline, but with a central convolutional block replaced by the quantum convolution layer.
3. **Capacity-matched classical model**: A classical model with adjusted width/depth such that its parameter count matches that of the hybrid model, ensuring that any advantage is not merely due to increased capacity.

All models share identical input encoders and readout heads, differing only in the central feature-extraction block.

### 7.3 Training Protocol

For each model:

- Use the same optimizer (e.g., Adam), batch size, and learning-rate schedule.
- Train with multiple random seeds to estimate variance.
- Apply early stopping based on validation performance to avoid overfitting.

For hybrid models, quantum simulation resources (number of shots, backend configuration) are standardized across runs and reported as part of the experimental budget.

### 7.4 Evaluation Metrics

We evaluate:

- **Convergence speed**: Number of epochs or gradient steps required to reach predefined validation performance thresholds.
- **Final test performance**: Classification accuracy, ROC–AUC, or regression error on a held-out test set.
- **Parameter efficiency**: Performance as a function of trainable parameter count, comparing hybrid and classical models.
- **Robustness**: Performance under parameter pruning, dropout, or injected noise, probing the stability of learned representations.

On the hello-world benchmarks (MUTAG, PROTEINS), the first goal is to demonstrate that:

- The hybrid model can match or outperform standard GNN baselines at *fixed or smaller* parameter count, and  
- The hybrid model reaches that performance in fewer training steps or epochs.

Statistical tests (e.g., paired t-tests or nonparametric alternatives) will assess the significance of observed differences.

---

## 8. Implementation Details

### 8.1 Software Stack

The implementation uses the following components:

- **PyTorch** for model definition, training loops, and classical layers.
- **Qiskit** for defining, simulating, and optionally executing quantum circuits.
- **Integration glue**—custom Python modules that wrap Qiskit circuits as differentiable PyTorch components.

The code is organized so that the quantum convolution layer can be swapped with a purely classical layer via configuration flags, enabling direct experiments.

### 8.2 Quantum Convolution Module

The quantum convolution is implemented as a PyTorch module with:

- A `forward` method that:
  - Accepts a tensor of patch features produced by a graph/patch extractor.
  - Encodes each patch into a Qiskit quantum circuit using an angle-based, topology-aware graph-to-qubit mapping.
  - Applies a shared variational ansatz \(U(\boldsymbol{\theta})\) across all patches.
  - Measures a predefined set of observables and returns a tensor of patch features.

- A custom autograd interface that:
  - Evaluates forward passes at shifted parameters \(\boldsymbol{\theta} \pm \tfrac{\pi}{2} \mathbf{e}_k\) to estimate gradients via the parameter-shift rule.
  - Aggregates these gradient estimates and passes them back to the optimizer.

To reduce overhead, the implementation reuses circuit templates, batches similar patches together, and exploits vectorized simulation backends when available.

### 8.3 Classical Reference Implementation

The classical convolution block used as a baseline matches the quantum block in:

- Input and output tensor shapes.
- Receptive-field size (patch radius \(R\)).
- Aggregation and pooling strategies.

This alignment allows the hybrid and classical models to be compared directly under identical training protocols and hyperparameters.

---

## 9. Discussion

### 9.1 Practicality in the NISQ Era

Although the study is instantiated via classical simulation of quantum circuits, the design explicitly respects constraints of near-term hardware:

- Small numbers of qubits per patch (e.g., 4–8 qubits).
- Shallow circuits with limited entangling depth.
- Simple, topology-aware encodings that minimize gate overhead.

If we observe measurable performance gains within these constraints, they provide actionable evidence that hybrid quantum feature extraction can deliver value before fully fault-tolerant quantum computing is available.

### 9.2 Scalability Considerations

The main bottleneck is the cost of simulating or executing many small circuits, one per patch and per sample. Several strategies can mitigate this:

- Limiting the number of patches via sparsified neighborhoods or learned patch selection.
- Sharing circuits across structurally similar patches.
- Leveraging specialized quantum hardware or accelerators for high-throughput execution when available.

These trade-offs are part of the design space. Any claimed advantage must be evaluated relative to the total wall-clock time and energy required for training.

### 9.3 Extensions and Variants

Natural extensions of this work include:

- **Multi-layer quantum stacks**: Using multiple quantum convolution layers with residual connections, approximating deeper classical feature hierarchies.
- **Graph-structured entanglement**: Designing entangling patterns that mirror the adjacency structure of the underlying graph, potentially enhancing inductive bias.
- **Quantum attention mechanisms**: Replacing or augmenting classical attention modules with quantum-parameterized attention layers.

Each extension increases architectural complexity but may amplify potential advantages on specific tasks.

---

## 10. Conclusion

This paper has formulated a concrete research program around the claim that quantum convolution layers can provide a discernible advantage in feature extraction for non-Euclidean datasets. Specifically, we have:

- Defined a quantum convolution layer as a shared-parameter variational quantum circuit acting on local patches of structured data, with an explicit graph-to-qubit mapping strategy.
- Embedded this layer into a hybrid quantum–classical architecture implemented with Qiskit and PyTorch.
- Articulated theoretical reasons why such layers may be more expressive and parameter-efficient than classical convolutions for certain data regimes, and how the convolutional reuse of small circuits can mitigate barren plateau issues.
- Proposed an experimental design to test convergence, accuracy, and parameter-efficiency advantages on a staged ladder of datasets, from standard graph benchmarks (MUTAG, PROTEINS) through molecular and particle-physics data.

The next step is to instantiate the described implementation, run the outlined experiments, and either confirm or refute the central claim. Doing so will move the discussion of quantum advantage from abstract complexity arguments toward concrete, domain-specific evidence about when and how small quantum circuits, integrated into classical learning pipelines, can yield practical benefits.
