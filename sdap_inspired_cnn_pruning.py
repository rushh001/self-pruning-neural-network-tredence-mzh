"""Self-Pruning Neural Network on CIFAR-10.

This script is aligned with the requirement file:
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


class PrunableLinear(nn.Module):
    """Linear layer with one learnable gate score per weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5.0))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.gate_scores, 2.0)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pruned_weight = self.weight * self.gates()
        return F.linear(x, pruned_weight, self.bias)


class SelfPruningMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@dataclass
class ExperimentResult:
    lambda_value: float
    test_accuracy: float
    sparsity_percent: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_ds = datasets.CIFAR10("./data", train=True, transform=transform, download=True)
    test_ds = datasets.CIFAR10("./data", train=False, transform=transform, download=True)

    use_pin_memory = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=use_pin_memory)
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(test_ds, shuffle=False, **kw)


def prunable_layers(model: nn.Module) -> Iterable[PrunableLinear]:
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            yield m


def gate_l1_loss(model: nn.Module) -> torch.Tensor:
    return sum(layer.gates().sum() for layer in prunable_layers(model))


def collect_all_gates(model: nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([layer.gates().reshape(-1) for layer in prunable_layers(model)])


def sparsity_percent(model: nn.Module, threshold: float) -> float:
    gates = collect_all_gates(model)
    return 100.0 * (gates < threshold).sum().item() / gates.numel()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_criterion: nn.Module,
    lambda_value: float,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    ce_sum = sparse_sum = total_sum = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        ce = ce_criterion(logits, y)
        sparse = gate_l1_loss(model)

        loss = ce + lambda_value * sparse
        loss.backward()
        optimizer.step()

        ce_sum += ce.item()
        sparse_sum += sparse.item()
        total_sum += loss.item()
        n += 1

    return ce_sum / n, sparse_sum / n, total_sum / n


def parse_lambdas(text: str) -> List[float]:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) < 3:
        raise ValueError("Provide at least 3 lambda values, e.g. 1e-5,5e-5,1e-4")
    return vals


def plot_gates(gates: torch.Tensor, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(gates.cpu().numpy(), bins=80, range=(0.0, 1.0), color="#2a6fbb", alpha=0.9)
    plt.title("Distribution of Final Gate Values")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_results(results: List[ExperimentResult], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "results_sdap_file.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "test_accuracy", "sparsity_percent"])
        for r in results:
            w.writerow([r.lambda_value, f"{r.test_accuracy:.4f}", f"{r.sparsity_percent:.4f}"])


def save_report(
    results: List[ExperimentResult],
    best_plot_name: str,
    out_path: str,
) -> None:
    lines = [
        "# Self-Pruning Neural Network Report",
        "",
        "## Why L1 Penalty on Sigmoid Gates Encourages Sparsity",
        "",
        "Each weight has a gate defined as sigmoid(gate_score), so each gate is in [0, 1].",
        "The sparsity term is the sum of all gate values. Adding this L1-like term to",
        "classification loss pushes many gates toward very small values, while useful",
        "connections keep larger gates to preserve accuracy.",
        "",
        "## Results",
        "",
        "| Lambda | Test Accuracy (%) | Sparsity Level (%) |",
        "|---:|---:|---:|",
    ]

    for r in results:
        lines.append(f"| {r.lambda_value:.2e} | {r.test_accuracy:.2f} | {r.sparsity_percent:.2f} |")

    lines += [
        "",
        "## Gate Distribution (Best Model)",
        "",
        f"![Gate distribution]({best_plot_name})",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    train_loader, test_loader = build_dataloaders(args.batch_size, args.num_workers)
    lambdas = parse_lambdas(args.lambdas)

    # Train prunable models for multiple lambda values.
    results: List[ExperimentResult] = []
    best_acc = -1.0
    best_model = None

    for lambda_value in lambdas:
        print("=" * 80)
        print(f"Training model for lambda={lambda_value:.2e}")

        model = SelfPruningMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        ce_criterion = nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            ce, sparse, total = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                ce_criterion=ce_criterion,
                lambda_value=lambda_value,
                device=device,
            )
            print(
                f"Epoch {epoch:02d}/{args.epochs} | CE={ce:.4f} "
                f"| Sparse={sparse:.1f} | Total={total:.4f}"
            )

        acc = evaluate(model, test_loader, device)
        sp = sparsity_percent(model, args.prune_threshold)
        print(f"Final test accuracy: {acc:.2f}%")
        print(f"Sparsity level (<{args.prune_threshold}): {sp:.2f}%")

        results.append(ExperimentResult(lambda_value=lambda_value, test_accuracy=acc, sparsity_percent=sp))

        if acc > best_acc:
            best_acc = acc
            best_model = model

    assert best_model is not None

    save_results(results, args.output_dir)

    plot_name = "best_model_gate_distribution_sdap_file.png"
    plot_path = os.path.join(args.output_dir, plot_name)
    gates = collect_all_gates(best_model)
    plot_gates(gates, plot_path)

    report_path = os.path.join(args.output_dir, "report_sdap_file.md")
    save_report(
        results=results,
        best_plot_name=plot_name,
        out_path=report_path,
    )

    print("=" * 80)
    print("Summary")
    for r in results:
        print(f"lambda={r.lambda_value:.2e} | acc={r.test_accuracy:.2f}% | sparsity={r.sparsity_percent:.2f}%")
    print(f"Saved: {os.path.join(args.output_dir, 'results_sdap_file.csv')}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-pruning neural network on CIFAR-10")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--lambdas", type=str, default="1e-6,1e-5,1e-4")
    p.add_argument("--prune-threshold", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
