# How To Run

## 1) Install dependencies

```bash
pip install torch torchvision matplotlib numpy
```

## 2) Run experiments (3 lambda values minimum)

```bash
python self_pruning_cifar10.py --epochs 8 --lambdas 1e-7,5e-7,1e-6
```

You can try a stronger pruning sweep:

```bash
python self_pruning_cifar10.py --epochs 8 --lambdas 5e-7,1e-6,5e-6
```

## 3) Output artifacts

After completion, these files are generated in `outputs/`:
- `results.csv`
- `best_model_gate_distribution.png`
- `report.md`

`outputs/report.md` is the short report required by the task:
- explanation of L1 sparsity on sigmoid gates
- lambda vs accuracy vs sparsity table
- gate distribution plot for the best model
