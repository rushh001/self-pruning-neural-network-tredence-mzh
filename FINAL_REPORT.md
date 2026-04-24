# Final Report: The Self-Pruning Neural Network

## Problem Summary
This project implements a self-pruning feed-forward neural network for CIFAR-10 where each weight has a learnable gate. During training, the model jointly learns classification and sparsity by optimizing:

`Total Loss = CrossEntropyLoss + lambda * SparsityLoss`

where `SparsityLoss` is the sum of all gate values across all prunable layers.

## Implementation Summary
The final implementation is in [self_pruning_cifar10.py](self_pruning_cifar10.py) and includes:

- Custom `PrunableLinear(in_features, out_features)`
- Gate computation with sigmoid: `gates = sigmoid(gate_scores)`
- Pruned weights: `pruned_weights = weight * gates`
- Full train/eval loop on CIFAR-10
- Sparsity metric: percentage of gates below threshold (`gate < 1e-2`)
- Multi-lambda comparison (3 values)
- Auto-generated artifacts (CSV + Markdown + gate histogram)

## Experiment Setup
- Dataset: CIFAR-10 (`torchvision.datasets`)
- Model: 3-layer MLP with prunable linear layers
- Optimizer: Adam
- Default lambdas tested: `1e-6, 1e-5, 1e-4`
- Sparsity threshold: `1e-2`

## Results
From [outputs/results.csv](outputs/results.csv):

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|---:|---:|---:|
| 1.00e-06 | 52.99 | 0.00 |
| 1.00e-05 | 55.77 | 8.51 |
| 1.00e-04 | 57.49 | 44.02 |

## Analysis
- Increasing `lambda` increased sparsity as expected.
- In this run, higher sparsity also improved test accuracy.
- This indicates the gate penalty acted as useful regularization while removing redundant connections.
- The best tested setting was `lambda = 1e-4` with 57.49% accuracy and 44.02% sparsity.

## Conclusion
The implementation satisfies the assignment requirements and demonstrates successful self-pruning behavior with a clear sparsity-vs-performance trend.

## Artifacts
- Results table: [outputs/results.csv](outputs/results.csv)
- Generated report: [outputs/report.md](outputs/report.md)
- Gate distribution plot: [outputs/best_model_gate_distribution.png](outputs/best_model_gate_distribution.png)
