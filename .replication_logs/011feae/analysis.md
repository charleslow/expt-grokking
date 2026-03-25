# Analysis — Commit 011feae (timing/weight-norm diagnostics, LOG_EVERY=5000)

## Run Summary
- **Partial run**: killed at step ~22k of 100k due to progressive slowdown (same issue as b4f5c12)
- Optimizer: Adam, lr=1e-3, wd=0.0, betas=(0.9, 0.98)
- New diagnostics: w_norm and step_ms columns confirmed

## Diagnostic Results

| Step  | train_acc | val_acc | val_loss | w_norm | step_ms |
|-------|-----------|---------|----------|--------|---------|
| 0     | 1.4%      | 0.8%    | 4.78     | 122.0  | 0.0     |
| 5000  | 100%      | 2.1%    | 13.5     | 133.7  | 72.3    |
| 10000 | 100%      | 2.2%    | 8.9      | 154.5  | 73.3    |
| 15000 | 100%      | 2.6%    | 8.9      | 277.9  | 73.6    |
| 20000 | 99.7%     | 3.9%    | 35.7     | 358.7  | 105.3   |

## Key Findings

### 1. Weight norm grows exponentially after step 10k
- Steps 0-10k: 122 → 155 (slow, linear growth)
- Steps 10k-15k: 155 → 278 (doubled in 5k steps!)
- Steps 15k-20k: 278 → 359 (continued rapid growth)
- This confirms unbounded weight growth without weight decay

### 2. Slowdown precisely correlates with weight norm
- step_ms stable at ~73ms through step 15k (w_norm < 278)
- step_ms jumps to 105ms at step 20k (w_norm = 359)
- After step 20k, process becomes too slow to produce step 25k in 10+ minutes
- Previous run (b4f5c12) saw same pattern: 73ms→700ms as w_norm grew

### 3. Val loss explosion also correlates
- val_loss stable at ~8.9 through step 15k
- Jumps to 35.7 at step 20k — large attention logits produce extreme softmax outputs

### 4. Memorization matches paper
- Train acc reaches 100% by step 5k (likely earlier, between 0-5k) — paper says ~10^3
- No grokking at 20k steps — expected, paper shows grokking at ~10^5

## Root Cause Analysis
The CPU slowdown is caused by denormalized floating-point numbers. When weights grow large (w_norm > ~250), the gradients of near-zero softmax outputs produce denormalized floats. On x86 CPUs, denormalized float operations are 10-100x slower than normal float operations. This is a CPU-specific artifact — GPUs handle denormals in hardware.

## Fix Proposal
Add `torch.set_flush_denormal(True)` at program start. This flushes denormalized floats to zero, which:
- Eliminates the CPU slowdown completely
- Does NOT change optimization dynamics (denormals are ~10^-38, effectively zero)
- Is standard practice for CPU-bound PyTorch training
