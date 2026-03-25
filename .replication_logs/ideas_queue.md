# Ideas Queue

1. **[NEXT] Switch to Adam wd=0 for Figure 1 replication**: Use Adam optimizer (no weight decay) and 10⁶ steps to replicate the dramatic grokking delay. This is what Figure 1 actually uses. On CPU, 10⁶ steps will take ~7 hours — consider reducing LOG_EVERY to 5000 and accepting the long runtime.
2. **Try 10⁵ steps with wd=0 first**: Test whether any grokking signal appears within 10⁵ steps without weight decay. If val acc is still near chance, confirms we need the full 10⁶.
3. **Weight decay ablation**: Run with different wd values (0.01, 0.1, 1.0) to see speed-of-grokking as a function of wd — matches paper's Figure 2.
4. **Try modular addition**: Simpler operation, might grok faster — useful for debugging or alternative target.
