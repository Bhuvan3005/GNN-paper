# CLAUDE.md

# Project Overview

This repository implements my research project on **Graph Neural Networks (GNNs) for Cardiovascular Disease Prediction using Feature-Node Graphs and Explainable AI**.

This is an IEEE journal research project, **not a coursework assignment**. Any modifications should preserve the research methodology unless explicitly requested.

---

# Research Objective

The goal of this work is to demonstrate that representing **clinical features as graph nodes** enables Graph Convolutional Networks (GCNs) to model relationships among cardiovascular risk factors more effectively than treating features independently.

Unlike conventional patient-node GNNs, each graph represents **one patient**, while the **graph topology represents relationships between clinical features**.

The project also evaluates the interpretability of graph-based predictions using multiple Explainable AI (XAI) metrics.

---

# Dataset

Dataset:
- UCI Cleveland Heart Disease Dataset

Samples:
- 303 patients

Target:
- Binary classification
    0 = No cardiovascular disease
    1 = Cardiovascular disease

Clinical Features (13):

1. age
2. sex
3. cp
4. trestbps
5. chol
6. fbs
7. restecg
8. thalach
9. exang
10. oldpeak
11. slope
12. ca
13. thal

---

# Data Preprocessing

Current preprocessing:

- Missing values:
    None

- Continuous features:
    MinMax Scaling

- Binary features:
    kept unchanged

- Ordinal features:
    MinMax scaled

Scaling is always fit **only on the training data**.

No preprocessing should introduce data leakage.

---

# Graph Construction

The graph is a **feature graph**, NOT a patient similarity graph.

Each feature is a node.

Number of nodes:

13

Graph edges are built using Pearson correlation computed ONLY on the training data.

Pipeline:

Training Fold
        ↓
Pearson Correlation Matrix
        ↓
Threshold Graph
        ↓
Minimum Spanning Tree (MST)
        ↓
Connected Feature Graph
        ↓
edge_index

The graph is reconstructed independently for every fold.

---

# Correlation Threshold

Graph construction uses

τ = 0.15

This threshold is intentionally FIXED.

Do NOT optimize it during cross-validation.

Reason:

- τ was selected using preliminary sensitivity analysis.
- Keeping τ fixed makes experiments reproducible.
- Only the classification threshold is optimized during validation.

If modifying the code, preserve

threshold = 0.15

unless explicitly instructed otherwise.

---

# Minimum Spanning Tree

The MST is mandatory.

Purpose:

- guarantees graph connectivity
- ensures every feature participates in message passing
- connects weakly correlated features

Distance:

distance = 1 - |Pearson correlation|

MST edges are added only if they are missing from the threshold graph.

---

# Spectral Analysis

Graph validation includes

- Laplacian
- Fiedler Eigenvalue
- Degree Centrality

The Laplacian is computed using absolute edge weights.

Do not change the spectral analysis implementation unless requested.

---

# Graph Representation

Each patient becomes one graph.

For every patient:

Nodes:
13 clinical features

Node Features:

Patient's normalized feature values

Shape:

x = [13,1]

Graph topology:

edge_index

This topology is identical for every patient.

Only node values change.

Graph label:

0 or 1

---

# Model

Architecture:

GCNConv

↓

ReLU

↓

Dropout (0.30)

↓

GCNConv

↓

ReLU

↓

Global Mean Pool

↓

Linear

↓

Sigmoid

Loss:

Binary Cross Entropy

Optimizer:

Adam

Learning Rate:

0.001

Weight Decay:

1e-4

Hidden Dimension:

32

Epochs:

100

---

# Cross Validation

Evaluation uses

5-fold Stratified Cross Validation.

For each fold:

Training Fold

↓

Validation Split (from training only)

↓

Scaling

↓

Graph Construction

↓

GCN Training

↓

Threshold Selection

↓

Test Evaluation

The graph is rebuilt independently for every fold.

No information from the test fold is used during graph construction.

---

# Classification Threshold

This is DIFFERENT from the graph correlation threshold.

For each fold:

Validation probabilities are used to find the threshold that maximizes F1-score.

This threshold is then frozen.

Only this threshold varies between folds.

The graph correlation threshold remains fixed at

τ = 0.15.

---

# Baseline Models

Comparison models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Multi-Layer Perceptron (MLP)

All models use the SAME StratifiedKFold splits.

Evaluation protocol must remain identical across all models.

---

# Evaluation Metrics

Primary metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- MCC

Additional metrics:

- Specificity
- False Positive Rate
- False Negative Rate

Final results are reported as

Mean ± Standard Deviation

across the five folds.

---

# Explainable AI

The project evaluates:

- Baseline Feature Attribution
- GNNExplainer

Metrics include:

- Fidelity
- Sparsity
- Stability
- Feature Importance Accuracy
- Clinical Relevance Score
- Explanation Coverage
- Trustworthiness Index

Do not modify the XAI pipeline unless requested.

---

# Research Contributions

This work proposes:

1. Feature-node graph representation for clinical tabular data.

2. Correlation-based graph construction with MST augmentation.

3. Spectral validation before GNN training.

4. Graph Convolutional Network for cardiovascular prediction.

5. Comprehensive multi-metric XAI evaluation.

---

# Important Constraints

Never introduce:

- data leakage
- graph construction using test data
- scaler fitted on validation or test data

Never compare models using different evaluation protocols.

All models must use identical cross-validation folds.

Do not change the research methodology without explicit instructions.

---

# Coding Guidelines

When modifying code:

- preserve reproducibility
- preserve random_state = 42
- preserve fixed graph threshold τ = 0.15
- preserve 5-fold stratified cross-validation
- preserve graph construction pipeline
- preserve spectral validation
- preserve XAI pipeline

Only modify components explicitly requested.
