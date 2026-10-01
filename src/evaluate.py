"""Turn the finished runs into figures and a metrics.json.

Run after run_experiments.sh:  python -m src.evaluate
"""
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data import CLASSES, FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, denormalize, get_loaders, load_arrays
from src.gradcam import GradCAM
from src.models import load_trained

FINAL_CANDIDATES = ["full_30ep", "no-dropout_30ep"]  # picked between on validation accuracy
FINAL_WEIGHTS = MODELS_DIR / "cifarnet_final.pt"
ABLATION = ["baseline", "no-batchnorm", "no-augment", "no-dropout", "full"]
NICE = {"baseline": "Baseline CNN", "no-batchnorm": "CifarNet − batch norm", "no-augment": "CifarNet − augmentation",
        "no-dropout": "CifarNet − dropout", "full": "CifarNet (everything)"}


def load_run(name):
    return json.loads((OUTPUTS_DIR / "runs" / f"{name}.json").read_text())


@torch.no_grad()
def predict_all(model, loader):
    probs, ys = [], []
    for x, y in loader:
        probs.append(model(x).softmax(1))
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    runs = {n: load_run(f"{n}_8ep") for n in ABLATION}
    long_runs = {n: load_run(n) for n in FINAL_CANDIDATES if (OUTPUTS_DIR / "runs" / f"{n}.json").exists()}
    final_run = max(long_runs, key=lambda n: long_runs[n]["best_val_acc"])
    final = long_runs[final_run]
    shutil.copy(MODELS_DIR / f"{final_run}.pt", FINAL_WEIGHTS)
    print("30-epoch runs (val / test):", {n: (r["best_val_acc"], r["test_acc"]) for n, r in long_runs.items()},
          "-> final:", final_run)

    # ---- 1. Ablation: what each idea is worth -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    names = ABLATION
    test = [100 * runs[n]["test_acc"] for n in names]
    train = [100 * runs[n]["final_train_acc"] for n in names]
    y = np.arange(len(names))
    axes[0].barh(y + 0.2, train, 0.4, color="#BBBBBB", label="train accuracy (last epoch)")
    axes[0].barh(y - 0.2, test, 0.4, color="#4C72B0", label="test accuracy")
    for i, v in enumerate(test):
        axes[0].text(v + 0.5, i - 0.2, f"{v:.1f}%", va="center", fontsize=9)
    axes[0].set_yticks(y, [NICE[n] for n in names])
    axes[0].set(xlim=(40, 102), xlabel="%", title="Ablation (8 epochs each)")
    axes[0].legend(loc="lower right", fontsize=8)
    for n, color in zip(names, ["#999999", "#C44E52", "#DD8452", "#8172B2", "#4C72B0"]):
        axes[1].plot([h["epoch"] for h in runs[n]["history"]], [100 * h["val_acc"] for h in runs[n]["history"]],
                     marker="o", ms=3, color=color, label=NICE[n])
    axes[1].set(xlabel="epoch", ylabel="validation accuracy (%)", title="Validation accuracy during training")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ablation.png", dpi=120)
    plt.close(fig)

    # ---- 2. The two 30-epoch runs: does dropout pay off with a longer budget? ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for n, color in zip(long_runs, ["#4C72B0", "#8172B2"]):
        h = long_runs[n]["history"]
        ep = [r["epoch"] for r in h]
        label = NICE[n.replace("_30ep", "")]
        axes[0].plot(ep, [100 * r["train_acc"] for r in h], color=color, ls="--", label=f"{label}, train")
        axes[0].plot(ep, [100 * r["val_acc"] for r in h], color=color, label=f"{label}, validation")
        axes[1].plot(ep, [r["train_loss"] for r in h], color=color, ls="--", label=f"{label}, train")
        axes[1].plot(ep, [r["val_loss"] for r in h], color=color, label=f"{label}, validation")
    axes[0].set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy (train is on augmented images)")
    axes[1].set(xlabel="epoch", title="Loss")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("The two 30-epoch runs")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=120)
    plt.close(fig)

    # ---- 3. Test-set analysis of the final model ----------------------------
    model = load_trained(FINAL_WEIGHTS)
    _, _, test_loader = get_loaders(batch_size=500, augment=False, num_workers=2)
    probs, y_true = predict_all(model, test_loader)
    y_pred = probs.argmax(1)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set(xlabel="Predicted", ylabel="True", title=f"Confusion matrix (test accuracy {final['test_acc']:.1%})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    per_class = {c: round(report[c]["recall"], 4) for c in CLASSES}
    order = sorted(CLASSES, key=per_class.get)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(order, [100 * per_class[c] for c in order], color=["#C44E52" if per_class[c] < 0.85 else "#4C72B0" for c in order])
    for i, c in enumerate(order):
        ax.text(100 * per_class[c] - 1, i, f"{100 * per_class[c]:.1f}%", ha="right", va="center", color="white", fontsize=9)
    ax.set(xlim=(50, 100), xlabel="Test accuracy (%)", title="Per-class accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_accuracy.png", dpi=120)
    plt.close(fig)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    top_pairs = [(CLASSES[i], CLASSES[j], int(off[i, j]))
                 for i, j in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1][:6], off.shape))]

    # ---- 4. Confident mistakes ---------------------------------------------
    test_x, _ = load_arrays("test")
    wrong = np.where(y_pred != y_true)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:18]
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.6))
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(test_x[i], interpolation="nearest")
        ax.set_title(f"{CLASSES[y_true[i]]} → {CLASSES[y_pred[i]]}\n{probs[i].max():.0%} sure", fontsize=8)
        ax.axis("off")
    fig.suptitle("The model's most confident mistakes (true → predicted). They're 32×32 pixels: squint and some are fair.")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confident_mistakes.png", dpi=120)
    plt.close(fig)

    # ---- 5. Grad-CAM -------------------------------------------------------
    cam = GradCAM(model, model.last_conv)
    rng = np.random.default_rng(1)
    picks = []
    for c in range(10):  # a correctly classified example of each class (any example if there is none)
        pool = np.where((y_true == c) & (y_pred == c))[0]
        pool = pool if len(pool) else np.where(y_true == c)[0]
        picks.append(int(rng.choice(pool)))
    fig, axes = plt.subplots(2, 10, figsize=(16, 3.8))
    for k, i in enumerate(picks):
        x, _ = test_loader.dataset[i]
        heat, cls, p = cam(x.unsqueeze(0))
        img = denormalize(x).permute(1, 2, 0).numpy()
        axes[0, k].imshow(img, interpolation="nearest")
        axes[0, k].set_title(CLASSES[y_true[i]], fontsize=9)
        axes[1, k].imshow(img, interpolation="nearest")
        axes[1, k].imshow(heat, cmap="jet", alpha=0.45, interpolation="bilinear")
        axes[1, k].set_title(f"{p[cls]:.0%}", fontsize=9)
        for ax in axes[:, k]:
            ax.axis("off")
    fig.suptitle("Grad-CAM: where the network looks when it gets it right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gradcam.png", dpi=120)
    plt.close(fig)

    # Example grid of the dataset itself
    train_x, train_y = load_arrays("train")
    fig, axes = plt.subplots(4, 10, figsize=(14, 6))
    for c in range(10):
        idx = np.where(train_y == c)[0][:4]
        for r, i in enumerate(idx):
            axes[r, c].imshow(train_x[i], interpolation="nearest")
            axes[r, c].axis("off")
        axes[0, c].set_title(CLASSES[c], fontsize=10)
    fig.suptitle("CIFAR-10: 32×32 colour images, 10 classes")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_samples.png", dpi=110)
    plt.close(fig)

    summary = {
        "final_run": final_run,
        "final": {k: v for k, v in final.items() if k != "history"},
        "long_runs": {n: {k: v for k, v in r.items() if k != "history"} for n, r in long_runs.items()},
        "ablation": {n: {"test_acc": runs[n]["test_acc"], "final_train_acc": runs[n]["final_train_acc"],
                         "best_val_acc": runs[n]["best_val_acc"], "params": runs[n]["params"],
                         "avg_epoch_seconds": runs[n]["avg_epoch_seconds"]} for n in ABLATION},
        "per_class_accuracy": per_class,
        "most_confused_pairs": [{"true": a, "pred": b, "count": n} for a, b, n in top_pairs],
        "n_test_errors": int((y_pred != y_true).sum()),
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["per_class_accuracy", "most_confused_pairs", "n_test_errors"]}, indent=1))
    print(f"Final test accuracy: {final['test_acc']:.4f}")


if __name__ == "__main__":
    main()
