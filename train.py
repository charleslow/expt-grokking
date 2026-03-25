"""
Grokking replication: Modular division (mod 97), decoder-only transformer.
Following Power et al. 2022, Appendix A.1.2.
"""

import math
import random
import json
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters (from paper Appendix A.1.2) ──────────────────────────────
P = 97                    # prime modulus
FRAC_TRAIN = 0.5          # fraction of data for training
D_MODEL = 128             # transformer width
N_LAYERS = 2              # transformer depth
N_HEADS = 4               # attention heads
D_FF = 4 * D_MODEL        # feed-forward hidden dim (512)
SEQ_LEN = 5               # x, op, y, =, result
VOCAB_SIZE = P + 2         # 0..96 + op_token(97) + eq_token(98)
OP_TOKEN = P               # 97
EQ_TOKEN = P + 1           # 98

LR = 1e-3
WEIGHT_DECAY = 0.0            # Figure 1 uses Adam with NO weight decay
BETAS = (0.9, 0.98)
WARMUP_STEPS = 10
BATCH_SIZE = 512
NUM_STEPS = 100_000        # 10^5 probe; will increase to 10^6 if grokking not seen
LOG_EVERY = 5000
SEED = 42

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Flush denormalized floats to zero on CPU — prevents 10x slowdown when weights
# grow large without weight decay (denormal float ops are 10-100x slower on x86)
if device.type == "cpu":
    torch.set_flush_denormal(True)
    print("Flush denormal: enabled (CPU mode)")

# ── Data: modular division x/y mod 97 ────────────────────────────────────────
def build_dataset(seed=SEED):
    """Build all equations x/y mod 97 and split 50/50."""
    equations = []
    for x in range(P):
        for y in range(1, P):  # y > 0 for division
            result = (x * pow(y, P - 2, P)) % P  # Fermat's little theorem
            # Sequence: [x, op, y, eq, result]
            equations.append((x, OP_TOKEN, y, EQ_TOKEN, result))

    rng = random.Random(seed)
    rng.shuffle(equations)

    n_train = int(len(equations) * FRAC_TRAIN)
    train_eqs = equations[:n_train]
    val_eqs = equations[n_train:]

    train_data = torch.tensor(train_eqs, dtype=torch.long)
    val_data = torch.tensor(val_eqs, dtype=torch.long)

    print(f"Total equations: {len(equations)}")
    print(f"Train: {len(train_eqs)}, Val: {len(val_eqs)}")

    return train_data, val_data

# ── Model: Decoder-only Transformer ──────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, causal_mask):
        # Pre-norm style
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask, is_causal=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, d_ff=D_FF, max_len=SEQ_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Register causal mask
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)  # True = masked

        # Count non-embedding parameters
        n_emb = sum(p.numel() for p in [self.tok_emb.weight, self.pos_emb.weight])
        n_total = sum(p.numel() for p in self.parameters())
        print(f"Total params: {n_total:,}, Non-embedding params: {n_total - n_emb:,}")

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(positions)

        # Expand causal mask for MultiheadAttention: (T, T) float mask
        mask = self.causal_mask[:T, :T]
        causal = torch.zeros_like(mask, dtype=h.dtype)
        causal.masked_fill_(mask, float('-inf'))

        for block in self.blocks:
            h = block(h, causal)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits


# ── Training ─────────────────────────────────────────────────────────────────
def get_batches(data, batch_size, device):
    """Yield random minibatches forever."""
    n = len(data)
    while True:
        idx = torch.randint(0, n, (min(batch_size, n),))
        yield data[idx].to(device)


def lr_schedule(step):
    """Linear warmup then constant."""
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    return 1.0


def evaluate(model, data, device):
    """Compute loss and accuracy on full dataset (answer token only)."""
    model.eval()
    with torch.no_grad():
        # Process in chunks to avoid OOM
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        chunk_size = 2048

        for i in range(0, len(data), chunk_size):
            batch = data[i:i+chunk_size].to(device)
            # Feed first 4 tokens [x, op, y, =], predict result from = position
            logits = model(batch[:, :-1])
            answer_logits = logits[:, -1, :]  # logits at = position
            answer_targets = batch[:, -1]     # result token
            loss = F.cross_entropy(answer_logits, answer_targets)
            preds = answer_logits.argmax(dim=-1)

            total_loss += loss.item() * len(batch)
            total_correct += (preds == answer_targets).sum().item()
            total_count += len(batch)

    model.train()
    return total_loss / total_count, total_correct / total_count


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    train_data, val_data = build_dataset()
    model = GrokkingTransformer().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    train_iter = get_batches(train_data, BATCH_SIZE, device)

    # Effective batch size: min(512, len(train_data)//2) per paper
    eff_batch = min(BATCH_SIZE, len(train_data) // 2)
    print(f"Effective batch size: {eff_batch}")
    print(f"Training for {NUM_STEPS} steps, logging every {LOG_EVERY} steps")
    print(f"LR={LR}, WD={WEIGHT_DECAY}, Betas={BETAS}, Warmup={WARMUP_STEPS}")
    print()

    # Header for structured logging
    print(f"{'step':>8} | {'train_loss':>10} | {'train_acc':>9} | {'val_loss':>10} | {'val_acc':>9} | {'lr':>10} | {'time':>8} | {'w_norm':>8} | {'step_ms':>8}")
    print("-" * 110)

    t0 = time.time()
    step_t0 = time.time()

    for step in range(NUM_STEPS):
        batch = next(train_iter)

        # Feed first 4 tokens [x, op, y, =], predict result from = position
        logits = model(batch[:, :-1])
        answer_logits = logits[:, -1, :]  # logits at = position
        answer_targets = batch[:, -1]     # result token
        loss = F.cross_entropy(answer_logits, answer_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
            train_loss, train_acc = evaluate(model, train_data, device)
            val_loss, val_acc = evaluate(model, val_data, device)
            elapsed = time.time() - t0
            current_lr = scheduler.get_last_lr()[0]

            # Compute weight norm for diagnostics
            w_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5

            # Average ms per training step since last log
            interval = max(step, 1)  # avoid div by zero at step 0
            if step > 0:
                avg_step_ms = (time.time() - step_t0) / LOG_EVERY * 1000
            else:
                avg_step_ms = 0.0
            step_t0 = time.time()

            print(f"{step:>8} | {train_loss:>10.4f} | {train_acc:>8.4f}% | {val_loss:>10.4f} | {val_acc:>8.4f}% | {current_lr:>10.6f} | {elapsed:>7.1f}s | {w_norm:>8.1f} | {avg_step_ms:>7.1f}")
            sys.stdout.flush()

    # Final evaluation
    train_loss, train_acc = evaluate(model, train_data, device)
    val_loss, val_acc = evaluate(model, val_data, device)

    print()
    print("=" * 80)
    print(f"FINAL RESULTS (step {NUM_STEPS}):")
    print(f"  Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}%")
    print(f"  Val loss:   {val_loss:.4f},   Val acc:   {val_acc:.4f}%")
    print("=" * 80)

    # Save final metrics as JSON for easy parsing
    metrics = {
        "step": NUM_STEPS,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    with open("final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to final_metrics.json")


if __name__ == "__main__":
    main()
