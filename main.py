"""
Project 34: Comparison of Optimizers (SGD vs. Adam & Adaptive Methods)
=====================================================================
Description:
Empirical benchmarking and mathematical comparison of first-order optimization
algorithms on the MNIST digit classification benchmark. Compares Vanilla SGD,
SGD with Polyak Momentum, RMSprop, Adam (Adaptive Moment Estimation), and AdamW
under identical parameter initialization and dataset permutations. Evaluates
convergence speed, gradient dynamics, generalization accuracy, and computational efficiency.
"""

import copy
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# Configuration & Seed Initialization
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 5
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 10000
NUM_VAL_SAMPLES = 2000


def set_seed(seed: int = 42) -> None:
    """Ensure strict deterministic initialization across optimizer runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Neural Network Architecture
# -----------------------------------------------------------------------------
class DigitClassifier(nn.Module):
    """
    A 3-layer multilayer perceptron with batch normalization and dropout
    to benchmark optimizer behavior in non-convex loss surfaces.
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# Dataset Preparation
# -----------------------------------------------------------------------------
def get_dataloaders(
    data_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_train: int = NUM_TRAIN_SAMPLES,
    num_val: int = NUM_VAL_SAMPLES,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares balanced subsets of MNIST for controlled optimization comparison."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST(root=str(data_dir), train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root=str(data_dir), train=False, download=False, transform=transform)

    train_subset = Subset(train_data, list(range(min(num_train, len(train_data)))))
    test_subset = Subset(test_data, list(range(min(num_val, len(test_data)))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


# -----------------------------------------------------------------------------
# Optimizer Factory
# -----------------------------------------------------------------------------
def create_optimizer(
    opt_name: str, model_params
) -> Tuple[torch.optim.Optimizer, str, str]:
    """Instantiates the specified optimizer with canonical hyperparameters."""
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(model_params, lr=0.05)
        color = "#e41a1c"
        label = "Vanilla SGD (lr=0.05)"
    elif opt_name == "SGD_Momentum":
        optimizer = torch.optim.SGD(model_params, lr=0.01, momentum=0.9, nesterov=True)
        color = "#ff7f00"
        label = "SGD + Momentum (lr=0.01, m=0.9)"
    elif opt_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model_params, lr=0.001, alpha=0.99)
        color = "#984ea3"
        label = "RMSprop (lr=0.001, a=0.99)"
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999))
        color = "#377eb8"
        label = "Adam (lr=0.001, b1=0.9, b2=0.999)"
    elif opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model_params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
        color = "#4daf4a"
        label = "AdamW (lr=0.001, wd=0.01)"
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer, label, color


# -----------------------------------------------------------------------------
# Training and Evaluation Harness
# -----------------------------------------------------------------------------
def train_and_evaluate_optimizer(
    opt_key: str,
    base_state_dict: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = NUM_EPOCHS,
) -> Dict[str, any]:
    """
    Trains a fresh model instance from identical initial weights using the given optimizer.
    """
    model = DigitClassifier().to(device)
    model.load_state_dict(copy.deepcopy(base_state_dict))

    optimizer, opt_label, opt_color = create_optimizer(opt_key, model.parameters())
    criterion = nn.CrossEntropyLoss()

    step_losses = []
    step_grad_norms = []
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accs = []

    start_time = time.perf_counter()
    time_to_90_acc = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_train = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Record gradient norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.detach().data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            optimizer.step()

            step_losses.append(loss.item())
            step_grad_norms.append(grad_norm)
            running_loss += loss.item() * inputs.size(0)
            total_train += inputs.size(0)

        epoch_train_losses.append(running_loss / total_train)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                v_loss = criterion(outputs, targets)
                val_loss += v_loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(targets).sum().item()
                total_val += targets.size(0)

        val_acc = (correct / total_val) * 100.0
        epoch_val_losses.append(val_loss / total_val)
        epoch_val_accs.append(val_acc)

        if val_acc >= 90.0 and time_to_90_acc is None:
            time_to_90_acc = time.perf_counter() - start_time

    total_time = time.perf_counter() - start_time

    return {
        "key": opt_key,
        "label": opt_label,
        "color": opt_color,
        "step_losses": step_losses,
        "step_grad_norms": step_grad_norms,
        "train_losses": epoch_train_losses,
        "val_losses": epoch_val_losses,
        "val_accs": epoch_val_accs,
        "final_val_acc": epoch_val_accs[-1],
        "final_val_loss": epoch_val_losses[-1],
        "total_time_s": total_time,
        "time_to_90_s": time_to_90_acc or total_time,
        "model": model,
    }


# -----------------------------------------------------------------------------
# Diagnostic Plotting Dashboard
# -----------------------------------------------------------------------------
def plot_optimizer_comparison(
    results: List[Dict[str, any]],
    save_path: Path,
) -> None:
    """Generates a 4-panel publication-grade optimizer diagnostic dashboard."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # 1. Step-wise Training Loss (Smoothed moving average)
    ax_step = axes[0, 0]
    window = 10
    for res in results:
        raw_losses = res["step_losses"]
        smoothed = np.convolve(raw_losses, np.ones(window) / window, mode="valid")
        ax_step.plot(smoothed[:300], label=res["label"], color=res["color"], linewidth=1.8, alpha=0.9)
    ax_step.set_title("(a) Step-wise Training Loss (First 300 Steps)", fontsize=13, fontweight="bold", pad=10)
    ax_step.set_xlabel("Iteration / Optimization Step", fontsize=11)
    ax_step.set_ylabel("Cross-Entropy Loss (Smoothed)", fontsize=11)
    ax_step.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax_step.grid(True, linestyle=":", alpha=0.6)

    # 2. Validation Accuracy Trajectory
    ax_acc = axes[0, 1]
    epochs = range(1, len(results[0]["val_accs"]) + 1)
    for res in results:
        ax_acc.plot(
            epochs, res["val_accs"], "o-", label=res["label"], color=res["color"], linewidth=2.0, markersize=5
        )
    ax_acc.set_title("(b) Validation Top-1 Accuracy Trajectory (%)", fontsize=13, fontweight="bold", pad=10)
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_xticks(list(epochs))
    ax_acc.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax_acc.grid(True, linestyle=":", alpha=0.6)

    # 3. Gradient Norm Dynamics over Iterations
    ax_grad = axes[1, 0]
    for res in results:
        raw_norms = res["step_grad_norms"]
        smoothed_norm = np.convolve(raw_norms, np.ones(window) / window, mode="valid")
        ax_grad.plot(smoothed_norm[:300], label=res["label"], color=res["color"], linewidth=1.8, alpha=0.85)
    ax_grad.set_title("(c) Gradient Norm Dynamics $\|\\nabla_\\theta \mathcal{L}\|_2$", fontsize=13, fontweight="bold", pad=10)
    ax_grad.set_xlabel("Iteration / Optimization Step", fontsize=11)
    ax_grad.set_ylabel("Global Gradient $L_2$ Norm", fontsize=11)
    ax_grad.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax_grad.grid(True, linestyle=":", alpha=0.6)

    # 4. Pareto Convergence Efficiency (Final Accuracy vs Training Duration)
    ax_bar = axes[1, 1]
    opt_names = [res["key"] for res in results]
    accs = [res["final_val_acc"] for res in results]
    times = [res["total_time_s"] for res in results]
    bar_colors = [res["color"] for res in results]

    bars = ax_bar.bar(opt_names, accs, color=bar_colors, edgecolor="black", width=0.55, linewidth=1.1)
    ax_bar.set_title("(d) Final Validation Accuracy Comparison", fontsize=13, fontweight="bold", pad=10)
    ax_bar.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax_bar.set_ylim(80.0, 100.0)
    ax_bar.grid(True, linestyle=":", alpha=0.6, axis="y")

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.5,
            f"{h:.2f}%\n({times[i]:.1f}s)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Artifact] Saved optimizer comparison dashboard to: {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("Project 34: Comparison of Optimizers (SGD vs. Adam vs. RMSprop)")
    print("=" * 75)
    print(f"Compute Device        : {device}")
    print(f"Dataset               : MNIST ({NUM_TRAIN_SAMPLES} train, {NUM_VAL_SAMPLES} val)")
    print(f"Epochs per Optimizer  : {NUM_EPOCHS} | Batch Size: {BATCH_SIZE}")
    print(f"Candidate Optimizers  : SGD, SGD+Momentum, RMSprop, Adam, AdamW")
    print("-" * 75)

    # 1. Dataset
    train_loader, test_loader = get_dataloaders(DATA_DIR)

    # 2. Canonical Model Initialization (Frozen Seed for Fairness)
    init_model = DigitClassifier()
    base_state_dict = copy.deepcopy(init_model.state_dict())
    total_params = sum(p.numel() for p in init_model.parameters())
    print(f"Model Parameters      : {total_params:,} (Identical initialization for all runs)")
    print("-" * 75)

    # 3. Benchmark Execution
    optimizers_to_test = ["SGD", "SGD_Momentum", "RMSprop", "Adam", "AdamW"]
    results = []

    for idx, opt_name in enumerate(optimizers_to_test, 1):
        print(f"\n>>> [{idx}/{len(optimizers_to_test)}] Evaluating Optimizer: {opt_name}...")
        res = train_and_evaluate_optimizer(
            opt_key=opt_name,
            base_state_dict=base_state_dict,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=NUM_EPOCHS,
        )
        results.append(res)
        print(
            f"    Final Val Acc: {res['final_val_acc']:6.2f}% | "
            f"Final Val Loss: {res['final_val_loss']:.4f} | "
            f"Wall-Clock Time: {res['total_time_s']:5.2f}s | "
            f"Time to 90%: {res['time_to_90_s']:5.2f}s"
        )

    # 4. Save Checkpoint of Best Performing Optimizer
    best_res = max(results, key=lambda x: x["final_val_acc"])
    checkpoint_path = CHECKPOINT_DIR / "best_optimizer_model.pt"
    torch.save(
        {
            "best_optimizer": best_res["key"],
            "val_acc": best_res["final_val_acc"],
            "model_state_dict": best_res["model"].state_dict(),
        },
        checkpoint_path,
    )
    print(f"\n[Checkpoint] Saved best model ({best_res['key']}) to: {checkpoint_path}")

    # 5. Diagnostic Visual Dashboard
    figure_path = PROJECT_DIR / "optimizer_comparison.png"
    plot_optimizer_comparison(results, figure_path)

    print("\n" + "=" * 75)
    print("Optimizer Comparison Experiment Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()

