# Ideas Queue

1. **[NEXT] Fix CPU slowdown with wd=0**: Add timing instrumentation to training loop to isolate whether slowdown is in forward pass, backward pass, or optimizer step. If caused by weight magnitude, consider: (a) periodic weight norm logging, (b) running with `torch.set_num_threads()` to avoid thread contention, (c) reducing eval frequency to LOG_EVERY=5000. Do NOT add gradient clipping (paper doesn't use it).
2. **Re-run Adam wd=0 with fix**: Once slowdown is fixed, run full 100k steps. If val still near chance at 100k, confirms need for 10⁶ steps.
3. **Scale to 10⁶ steps**: If grokking hasn't started by 100k, increase NUM_STEPS to 10⁶ with LOG_EVERY=10000. Paper says grokking at ~10⁵ steps, so it should start within this window.
4. **Weight decay ablation**: Run with different wd values (0.01, 0.1, 1.0) to see speed-of-grokking as a function of wd — matches paper's Figure 2.
5. **Try modular addition**: Simpler operation, might grok faster — useful for debugging or alternative target.
