"""Self-Pruning Neural Network on CIFAR-10.

This script follows the task specification directly:
1) custom PrunableLinear layer with learnable gate scores,
2) total loss = classification loss + lambda * sparsity loss,
3) training/evaluation across multiple lambda values,
4) reporting accuracy, sparsity, and gate-value distribution.
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PrunableLinear
# ──────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that multiplies each weight by a
    learnable gate in (0, 1).

    gate = sigmoid(gate_score)
    effective_weight = weight * gate

    The gate_scores are regular nn.Parameters, so gradients flow through both
    the weight and the gate via standard autograd.  Initialising gate_scores
    at +2.0 means gates start near sigmoid(2)≈0.88, giving the optimiser room
    to push them toward zero when penalised.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight + bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))

        # One scalar gate per weight element (same shape)
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming uniform for weights – matches nn.Linear default
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5.0))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

        # Gates start near 0.88 so the network begins close to a dense model
        # and gradually learns to prune.
        nn.init.constant_(self.gate_scores, 2.0)

    def gates(self) -> torch.Tensor:
        """Returns gate values in (0, 1) via sigmoid."""
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pruned_weights = self.weight * self.gates()   # element-wise gate
        return F.linear(x, pruned_weights, self.bias) # standard matmul


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Network  (MLP with Batch-Norm for better accuracy)
# ──────────────────────────────────────────────────────────────────────────────

class SelfPruningMLP(nn.Module):
    """
    Standard feed-forward MLP using PrunableLinear layers only.
    """

    def __init__(self):
        super().__init__()
        # 3×32×32 = 3072 inputs
        self.fc1 = PrunableLinear(3 * 32 * 32, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    lambda_value:    float
    test_accuracy:   float
    sparsity_percent: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_prunable_layers(model: nn.Module) -> Iterable[PrunableLinear]:
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            yield module


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    """L1 norm of ALL gate values across all PrunableLinear layers."""
    return sum(layer.gates().sum() for layer in get_prunable_layers(model))


def collect_all_gates(model: nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat(
            [layer.gates().reshape(-1) for layer in get_prunable_layers(model)]
        )


def compute_sparsity_percent(model: nn.Module, threshold: float) -> float:
    all_gates = collect_all_gates(model)
    return 100.0 * (all_gates < threshold).sum().item() / all_gates.numel()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    lambda_value: float,
    device:       torch.device,
) -> tuple[float, float, float]:

    model.train()
    total_ce = total_sparse = total_loss = batches = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        ce     = criterion(logits, labels)
        sparse = sparsity_loss(model)           # L1 of all gates
        loss   = ce + lambda_value * sparse

        loss.backward()
        optimizer.step()

        total_ce     += ce.item()
        total_sparse += sparse.item()
        total_loss   += loss.item()
        batches      += 1

    n = max(batches, 1)
    return total_ce / n, total_sparse / n, total_loss / n


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Data
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10("./data", train=True, transform=transform, download=True)
    test_ds = datasets.CIFAR10("./data", train=False, transform=transform, download=True)

    use_pin_memory = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=use_pin_memory)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(gates: torch.Tensor, lambda_value: float, output_path: str) -> None:
    arr = gates.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full histogram
    axes[0].hist(arr, bins=120, range=(0.0, 1.0), color="#1f77b4", alpha=0.85)
    axes[0].set_title(f"Gate Distribution  (λ={lambda_value:.1e})", fontsize=13)
    axes[0].set_xlabel("Gate value  sigmoid(gate_score)")
    axes[0].set_ylabel("Count")

    # Right: zoom into [0, 0.05] to show the spike at zero clearly
    axes[1].hist(arr, bins=60, range=(0.0, 0.05), color="#ff7f0e", alpha=0.85)
    axes[1].set_title("Zoom: gates near zero", fontsize=13)
    axes[1].set_xlabel("Gate value")
    axes[1].set_ylabel("Count")

    near_zero = (arr < 0.01).mean() * 100
    fig.suptitle(f"{near_zero:.1f}% of gates < 0.01  (pruned)", fontsize=12, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved gate histogram → {output_path}")


def write_results_csv(results: List[ExperimentResult], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "test_accuracy", "sparsity_percent"])
        for r in results:
            w.writerow([r.lambda_value, f"{r.test_accuracy:.4f}", f"{r.sparsity_percent:.4f}"])


def write_markdown_report(
    results: List[ExperimentResult],
    best_plot_path: str,
    output_path: str,
) -> None:
    lines = [
        "# Self-Pruning Neural Network — Report",
        "",
        "## Why L1 Penalty on Sigmoid Gates Encourages Sparsity",
        "",
        "Each weight's gate is computed as `g = sigmoid(s)` where `s` is a learnable",
        "scalar (gate score).  The total loss is:",
        "",
        "```",
        "Loss = CrossEntropy(ŷ, y)  +  λ · Σ g_i",
        "```",
        "",
        "The gradient of the L1 sparsity term w.r.t. each gate score `s` is",
        "`λ · sigmoid(s) · (1 − sigmoid(s))`, which is always positive, so gradient",
        "descent continuously pushes every gate score downward.  Weights that are",
        "**important for classification** resist this pull because the cross-entropy",
        "gradient pushes their score back up; weights that are **unimportant** cannot",
        "resist and their scores keep falling, driving gates close to 0.",
        "In practice, sigmoid gates are typically interpreted with threshold-based pruning",
        "(for example, gate < 1e-2 counts as pruned).",
        "",
        "## Results",
        "",
        "| Lambda | Test Accuracy (%) | Sparsity Level (%) |",
        "|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.lambda_value:.2e} | {r.test_accuracy:.2f} | {r.sparsity_percent:.2f} |"
        )

    lines += [
        "",
        "## Gate Distribution (Best Model)",
        "",
        f"![Gate distribution]({best_plot_path})",
        "",
        "A successful run shows a large spike at gate ≈ 0 (pruned connections) and a",
        "separate cluster of active connections with gate values > 0.5.",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved markdown report → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def parse_lambdas(text: str) -> List[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(values) < 3:
        raise ValueError("Provide at least 3 lambda values, e.g. 1e-4,1e-3,5e-3")
    return values


def run_experiments(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    train_loader, test_loader = build_dataloaders(args.batch_size, args.num_workers)
    lambdas = parse_lambdas(args.lambdas)

    results:      List[ExperimentResult] = []
    best_model    = None
    best_lambda   = None
    best_accuracy = -1.0

    for lambda_value in lambdas:
        print("\n" + "=" * 70)
        print(f"  λ = {lambda_value:.2e}  |  epochs = {args.epochs}")
        print("=" * 70)

        model = SelfPruningMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            avg_ce, avg_sparse, avg_total = train_one_epoch(
                model, train_loader, optimizer, criterion, lambda_value, device
            )
            print(
                f"  Ep {epoch:02d}/{args.epochs}  "
                f"CE={avg_ce:.4f}  Sparse={avg_sparse:.0f}  "
                f"Total={avg_total:.4f}"
            )

        test_acc = evaluate(model, test_loader, device)
        sparsity = compute_sparsity_percent(model, args.prune_threshold)

        print(f"  → Test accuracy : {test_acc:.2f}%")
        print(f"  → Sparsity      : {sparsity:.2f}%  (gate < {args.prune_threshold})")

        results.append(ExperimentResult(lambda_value, test_acc, sparsity))

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model    = model
            best_lambda   = lambda_value

    # ── Save outputs ──────────────────────────────────────────────────────────
    csv_path  = os.path.join(args.output_dir, "results.csv")
    plot_name = "best_model_gate_distribution.png"
    plot_path = os.path.join(args.output_dir, plot_name)
    md_path   = os.path.join(args.output_dir, "report.md")

    write_results_csv(results, csv_path)
    best_gates = collect_all_gates(best_model)
    plot_gate_distribution(best_gates, best_lambda, plot_path)
    write_markdown_report(results, plot_name, md_path)

    print("\n" + "=" * 70)
    print("Summary")
    for r in results:
        print(f"  λ={r.lambda_value:.2e}  acc={r.test_accuracy:.2f}%  sparsity={r.sparsity_percent:.2f}%")
    print(f"\nArtifacts saved to: {args.output_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-Pruning Neural Network on CIFAR-10")
    p.add_argument("--epochs", type=int, default=8,
                   help="Training epochs per lambda")
    p.add_argument("--batch-size",     type=int,   default=128)
    p.add_argument("--learning-rate",  type=float, default=1e-3)
    p.add_argument(
        "--lambdas", type=str, default="1e-6,1e-5,1e-4",
        help="Comma-separated lambda values (at least three values)."
    )
    p.add_argument("--prune-threshold", type=float, default=1e-2,
                   help="Gate threshold below which a weight is counted as pruned")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--num-workers",    type=int,   default=2)
    p.add_argument("--output-dir",     type=str,   default="outputs")
    p.add_argument("--device",         type=str,   default="auto",
                   choices=["auto", "cpu", "cuda"])
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_experiments(args)