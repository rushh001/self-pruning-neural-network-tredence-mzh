# Self-Pruning Neural Network — Report

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity

Each weight's gate is computed as `g = sigmoid(s)` where `s` is a learnable
scalar (gate score).  The total loss is:

```
Loss = CrossEntropy(ŷ, y)  +  λ · Σ g_i
```

The gradient of the L1 sparsity term w.r.t. each gate score `s` is
`λ · sigmoid(s) · (1 − sigmoid(s))`, which is always positive, so gradient
descent continuously pushes every gate score downward.  Weights that are
**important for classification** resist this pull because the cross-entropy
gradient pushes their score back up; weights that are **unimportant** cannot
resist and their scores keep falling, driving gates close to 0.
In practice, sigmoid gates are typically interpreted with threshold-based pruning
(for example, gate < 1e-2 counts as pruned).

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|---:|---:|---:|
| 1.00e-06 | 52.99 | 0.00 |
| 1.00e-05 | 55.77 | 8.51 |
| 1.00e-04 | 57.49 | 44.02 |

## Gate Distribution (Best Model)

![Gate distribution](best_model_gate_distribution.png)

A successful run shows a large spike at gate ≈ 0 (pruned connections) and a
separate cluster of active connections with gate values > 0.5.
