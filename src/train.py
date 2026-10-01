"""Train one of the CIFAR-10 experiments.

Examples:
    python -m src.train --experiment full --epochs 30          # the final model
    python -m src.train --experiment no-augment --epochs 8     # one ablation run

Each run writes outputs/runs/<experiment>_<epochs>ep.json (history + test
accuracy) and models/<experiment>_<epochs>ep.pt (best checkpoint on validation).
"""
import argparse
import json
import time

import torch
from torch import nn

from src.data import MODELS_DIR, OUTPUTS_DIR, get_loaders
from src.models import EXPERIMENTS, build_model, count_params


def run_epoch(model, loader, loss_fn, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.contiguous(memory_format=torch.channels_last)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), default="full")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = EXPERIMENTS[args.experiment]
    run_name = f"{args.experiment}_{args.epochs}ep"
    (OUTPUTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(args.batch_size, augment=config["augment"], seed=args.seed)
    model = build_model(**config).to(memory_format=torch.channels_last)
    print(f"[{run_name}] {count_params(model):,} params | config={config}", flush=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader), pct_start=0.25)

    ckpt_path = MODELS_DIR / f"{run_name}.pt"
    history, best_val, start = [], 0.0, time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, scheduler)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "seconds": round(time.time() - t0, 1)})
        saved = ""
        if va_acc > best_val:
            best_val, saved = va_acc, "  <- best"
            torch.save({"config": config, "state_dict": model.state_dict()}, ckpt_path)
        print(f"[{run_name}] epoch {epoch:2d}/{args.epochs} | train {tr_loss:.3f} / {tr_acc:.3f} | "
              f"val {va_loss:.3f} / {va_acc:.3f} | {time.time() - t0:.0f}s{saved}", flush=True)

    best = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(best["state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, loss_fn)
    print(f"[{run_name}] TEST accuracy {test_acc:.4f}", flush=True)

    summary = {
        "experiment": args.experiment, "epochs": args.epochs, "config": config,
        "params": count_params(model), "best_val_acc": round(best_val, 4),
        "final_train_acc": round(history[-1]["train_acc"], 4),
        "test_acc": round(test_acc, 4), "test_loss": round(test_loss, 4),
        "avg_epoch_seconds": round(sum(h["seconds"] for h in history) / len(history), 1),
        "total_minutes": round((time.time() - start) / 60, 1), "history": history,
    }
    (OUTPUTS_DIR / "runs" / f"{run_name}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
