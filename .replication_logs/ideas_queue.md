# Ideas Queue

1. **[NEXT] Scale to 10⁶ steps**: Increase NUM_STEPS to 1,000,000 with LOG_EVERY=10,000 (or 50,000). Paper shows grokking completes by 10⁶. We're at 97.8% at 100k — need to confirm we reach 100% and compare the full learning curve shape. At 73ms/step, this will take ~20 hours.
2. **Increase logging resolution around grokking onset**: LOG_EVERY=5000 is too coarse to see the exact grokking transition. Consider LOG_EVERY=1000 for the 20k-60k range, or add a separate finer-grained log.
3. **Weight decay ablation**: Run with different wd values (0.01, 0.1, 1.0) to see speed-of-grokking as a function of wd — matches paper's Figure 2.
4. **Try modular addition**: Simpler operation, might grok faster — useful for debugging or alternative target.
