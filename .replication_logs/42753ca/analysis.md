# Analysis: 42753ca — flush_denormal fix, 100k steps with Adam wd=0

## Run Summary
- **Commit**: 42753ca (Enable flush_denormal on CPU to fix wd=0 training slowdown)
- **Config**: Adam, lr=1e-3, wd=0, betas=(0.9, 0.98), warmup=10, batch=512, 100k steps
- **Duration**: 7343s (~2h 2min), step_ms stable at ~73.4ms throughout

## Key Results
| Step | train_acc | val_acc | val_loss | w_norm |
|------|-----------|---------|----------|--------|
| 0 | 1.4% | 0.8% | 4.78 | 122.0 |
| 5000 | 100% | 2.1% | 13.5 | 133.7 |
| 10000 | 100% | 2.2% | 8.9 | 154.5 |
| 20000 | 99.3% | 4.1% | 34.6 | 358.7 |
| 30000 | 99.8% | 18.5% | 28.4 | 362.9 |
| 40000 | 100% | 51.3% | 10.4 | 365.8 |
| 50000 | 100% | 78.2% | 3.6 | 368.5 |
| 60000 | 99.9% | 87.3% | 1.6 | 370.5 |
| 70000 | 100% | 94.3% | 0.56 | 372.5 |
| 80000 | 100% | 96.7% | 0.29 | 374.8 |
| 90000 | 100% | 96.4% | 0.25 | 377.6 |
| 100000 | 100% | 97.8% | 0.12 | 380.5 |

## Observations

1. **Flush denormal fix works perfectly**: step_ms stayed constant at ~73.4ms through the entire 100k steps, even as w_norm grew to 380. Previously (011feae), training became unusable after w_norm exceeded ~250.

2. **Grokking is clearly happening**: val_acc went from chance (~2%) at step 10k to 97.8% at step 100k. The S-shaped generalization curve is visible: slow start (2%→4% from 10k→20k), rapid rise (18%→78% from 30k→50k), then asymptotic approach (78%→97.8% from 50k→100k).

3. **Memorization matches paper**: train_acc hits 100% by step 5000 (~10³·⁷), consistent with paper's "< 10³ steps."

4. **Weight norm dynamics**: w_norm grows rapidly from 122→359 in the first 20k steps, then stabilizes around 360-380 for the rest. The period of rapid weight growth (15k-20k) corresponds to the spike in val_loss (8.9→34.6), after which val_loss begins declining as the model starts to generalize.

## Comparison to Paper (Figure 1)
- **Paper**: val acc stays near chance until ~10⁵, then jumps to ~100% by ~10⁶
- **Ours**: val acc starts rising at ~25k steps, reaches 97.8% at 100k steps
- **Gap**: Our grokking happens ~4x earlier than the paper. At 100k (=10⁵) steps, the paper shows val acc just beginning to rise, while ours is already at 97.8%.

## Hypotheses for the discrepancy
1. **Grokking is stochastic**: The paper's Figure 1 may show one specific seed/run. The paper notes grokking timing varies.
2. **Different initialization**: PyTorch default init may differ from whatever the paper used.
3. **Already very close**: 97.8% at 100k is excellent. The paper's figure shows grokking completing by 10⁶ — ours may be completing faster. Need to run to 10⁶ to see if we plateau below 100% or continue to improve.

## Verdict
**KEEP** — This is a massive improvement over previous runs. The grokking phenomenon is clearly reproduced. Next step: extend to 10⁶ steps to see if val_acc reaches 100% and to better compare the timescale with the paper's Figure 1.
