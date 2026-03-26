# Analysis — Commit b4f5c12 (Adam wd=0, 100k steps target)

## Run Summary
- **Partial run**: killed at step 24,000 of 100,000 due to progressive slowdown
- Optimizer: Adam, lr=1e-3, wd=0.0, betas=(0.9, 0.98)
- Warmup: 10 steps, batch size 512

## Key Observations

### Memorization (matches paper)
- Train accuracy reaches 100% by step 1000 (paper: ~10^3) ✓
- Train loss drops to ~0.0 and stays there

### No grokking at 24k steps (expected)
- Val accuracy stays at 1.4–7.5% through 24k steps (chance is ~1%)
- Val loss initially climbed to 13.5, dropped to 8.8, then exploded to 38.9 after step 18k
- This matches Figure 1 where val stays near chance until ~10^5 steps

### Performance degradation (PROBLEM)
Per-1k-step wall time:
- Steps 0–17k: ~73s/1k (constant)
- Step 18k: 78s/1k
- Step 19k: 117s/1k
- Step 20k: 185s/1k
- Step 21k: 276s/1k
- Step 22k: 413s/1k
- Step 23k: 570s/1k
- Step 24k: 704s/1k

Slowdown correlates with val_loss spike at step 18k (8.97 → 28.48). Diagnostic at 5k steps showed constant forward/eval times and moderate weight growth (norm 122→134, max_param stable at 4.2). The explosive slowdown must emerge later, likely when weights grow large enough to cause numerical issues in attention/softmax on CPU.

## Comparison to Paper Target
- **Qualitatively correct**: memorization at ~10^3, no generalization at 24k — matches Figure 1's pattern
- **Quantitative replication blocked**: need 10^5–10^6 steps, but progressive slowdown makes this infeasible at current rate (would take 50+ hours)

## Root Cause Hypothesis for Slowdown
Without weight decay, model weights grow unbounded. Around step 18k, weights cross a threshold where attention logits become very large, causing numerical instability (val_loss spikes) and slower CPU computation (possibly denormalized floats, NaN handling, or softmax special cases). This is a CPU-specific issue — on GPU, the same computation would be constant time.

## Next Steps
1. Need to prevent the performance degradation without changing the optimization dynamics
2. Options: (a) add float16 or gradient monitoring, (b) reduce eval frequency, (c) investigate if the slowdown is in forward, backward, or eval
