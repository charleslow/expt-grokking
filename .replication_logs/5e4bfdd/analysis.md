# Analysis — Commit 5e4bfdd (Baseline with AdamW wd=1)

## Observation
- Train accuracy reaches 100% at step ~500 (paper: ~10^3 steps)
- Val accuracy rises from ~1.8% at step 500 to 66.6% at step 1500 to 100% at step 2000
- Both train and val converge to 100% with low loss (~0.003) by step 4000

## Comparison to Paper Target (Figure 1)
Figure 1 shows a dramatic grokking delay:
- Train: 100% at ~10^3 steps
- Val: near chance until ~10^5 steps, then jumps to 100% at ~10^6

Our results show grokking happening ~50x faster (val 100% at step 2000 vs 10^6).

## Root Cause of Discrepancy
Figure 1 explicitly uses **Adam with NO weight decay** and 10^6 steps (stated in paper Appendix A.1.2: "For the experiments reported in Section 3.1 we increased the optimization budget to 10^6, and used Adam optimizer with no weight decay").

Our run used AdamW with wd=1, which the paper shows dramatically accelerates generalization (Figure 2 left panel).

## Conclusion
The model implementation is correct — we see the memorization-then-generalization pattern, just compressed in time due to weight decay. To replicate Figure 1's dramatic grokking delay, we need to switch to Adam with wd=0 and run for 10^6 steps.

## Architecture Verification
- Non-embedding params: 409,472 (~4x10^5) — matches paper
- Model correctly computes loss only on the answer token
- Data split: 4656 train / 4656 val from 9312 total equations — matches paper
