# Ideas Queue

1. **[NEXT] Flush denormals to fix CPU slowdown**: Add `torch.set_flush_denormal(True)` at program start. This eliminates the 10x CPU slowdown caused by denormalized floats when weights grow large without weight decay. Does not change optimization dynamics. Then re-run with NUM_STEPS=100000.
2. **Scale to 10⁶ steps if needed**: If val still near chance at 100k, increase NUM_STEPS to 1000000 with LOG_EVERY=10000. Paper says grokking at ~10⁵ steps.
3. **Weight decay ablation**: Run with different wd values (0.01, 0.1, 1.0) to see speed-of-grokking as a function of wd — matches paper's Figure 2.
4. **Try modular addition**: Simpler operation, might grok faster — useful for debugging or alternative target.
