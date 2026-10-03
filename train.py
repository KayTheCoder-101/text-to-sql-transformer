import os
import sys
import time
import json

import torch
import torch.nn as nn
import sentencepiece as spm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STARTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "starter")
sys.path.insert(0, STARTER_DIR)

from dataset import make_loader
from tokenizer import PAD_ID

# ---------------- Settings ----------------
EPOCHS = 1
BATCH_SIZE = 64
D_MODEL = 256
WARMUP = 4000
LABEL_SMOOTHING = 0.1
SEED = 0
CKPT_DIR = "checkpoints"
USE_STANDIN = False   # set to False to train the real Transformer
MAX_BATCHES = 20       # set to None for real training (None = every batch)
OVERFIT = True       # NEW: True = memorize one batch (bug check), then exit
OVERFIT_STEPS = 300    # NEW

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Learning-rate schedule (Task 3.4) ----------------  # CHANGED
def get_lr(step, d_model=D_MODEL, warmup=WARMUP):
    step = max(step, 1)    # step 0 would divide by zero (LambdaLR starts at 0)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def plot_lr_schedule(path="results/lr_schedule.png"):                 # CHANGED
    steps = list(range(1, 20001))
    plt.figure(figsize=(10, 6))
    plt.plot(steps, [get_lr(s) for s in steps], color="blue", label="Learning Rate")
    plt.axvline(WARMUP, color="red", linestyle="--", label="warmup ends")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Noam Learning Rate Schedule")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------- Stand-in model (temporary, for testing the loop) ----------------
class StandInModel(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt_in):
        return self.out(self.emb(tgt_in))  # (B, T, vocab); ignores src


def build_model(vocab_size):
    if USE_STANDIN:
        return StandInModel(vocab_size).to(DEVICE)
    from model.transformer import Transformer  # imported only when needed
    return Transformer(vocab_size, d_model=D_MODEL, num_heads=4, num_layers=3,
                       d_ff=1024, dropout=0.1, pad_id=PAD_ID).to(DEVICE)


# ---------------- Train / evaluate ----------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total, batches = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]       # teacher forcing (3.1)
        logits = model(src, tgt_in)                      # (B, T, V)
        V = logits.size(-1)
        loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()                                 # schedule is in steps

        total += loss.item()
        batches += 1
        if MAX_BATCHES is not None and batches == MAX_BATCHES:
            break
    return total / max(batches, 1)


def evaluate(model, loader, criterion):
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in)
            V = logits.size(-1)
            loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))
            total += loss.item()
            batches += 1
            if MAX_BATCHES is not None and batches == MAX_BATCHES:
                break
    return total / max(batches, 1)


# ---------------- Overfit test (bug check) ----------------  # NEW
def overfit_one_batch(model, loader, criterion):
    """Train on ONE fixed batch. A correct model memorizes it, so the loss must fall a lot.
    Uses a constant learning rate: the warmup schedule would be far too small in 300 steps."""
    src, tgt = next(iter(loader))
    src, tgt = src.to(DEVICE), tgt.to(DEVICE)
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9)
    model.train()
    for step in range(1, OVERFIT_STEPS + 1):
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 50 == 0:
            print(f"overfit step {step:3d}  loss {loss.item():.4f}")


# ---------------- Checkpoint helpers ----------------
def save_resume(path, model, optimizer, scheduler, epoch, best_dev, best_epoch, history):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_dev": best_dev,
        "best_epoch": best_epoch,
        "history": history,
    }, path)


def load_resume(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_dev"], ckpt["best_epoch"], ckpt["history"]


# ---------------- Loss curve (Figure 2) ----------------
def plot_losses(history, path="results/loss_curve.png"):
    epochs = [h["epoch"] for h in history]
    plt.figure()
    plt.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    plt.plot(epochs, [h["dev_loss"] for h in history], label="dev loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training and dev loss")
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# ---------------- Main ----------------
if __name__ == "__main__":
    torch.manual_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    plot_lr_schedule()

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(STARTER_DIR, "sql_sp.model"))
    train_dl = make_loader(os.path.join(STARTER_DIR, "train_pairs.jsonl"), sp,
                           train=True, batch_size=BATCH_SIZE)
    dev_dl = make_loader(os.path.join(STARTER_DIR, "dev_pairs.jsonl"), sp,
                         train=False, batch_size=BATCH_SIZE)

    model = build_model(sp.get_piece_size())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)

    if OVERFIT:                                                        # NEW
        overfit_one_batch(model, train_dl, criterion)
        sys.exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    start_epoch, best_dev, best_epoch, history = 1, float("inf"), 0, []
    resume_path = os.path.join(CKPT_DIR, "last.pt")
    if os.path.exists(resume_path):
        epoch, best_dev, best_epoch, history = load_resume(resume_path, model, optimizer, scheduler)
        start_epoch = epoch + 1
        print("resumed from epoch", epoch)

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()                                      # NEW
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, scheduler)
        dev_loss = evaluate(model, dev_dl, criterion)
        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start                         # NEW
        print(f"epoch {epoch:2d}  train {train_loss:.4f}  dev {dev_loss:.4f}  "
              f"lr {lr:.3e}  time {epoch_time:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "dev_loss": dev_loss, "lr": lr, "time": epoch_time})  # CHANGED

        if dev_loss < best_dev:
            best_dev, best_epoch = dev_loss, epoch
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best.pt"))
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"epoch_{epoch}.pt"))
        save_resume(resume_path, model, optimizer, scheduler,
                    epoch, best_dev, best_epoch, history)
        with open("results/train_log.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    total_time = sum(h.get("time", 0.0) for h in history)              # CHANGED: survives resume
    print(f"total training time: {total_time / 60:.1f} min")
    print(f"best epoch: {best_epoch}  best dev loss: {best_dev:.4f}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    plot_losses(history)