# Replication Insights — Grokking (Power et al. 2022)

## Paper Target
**Figure 1 (left)**: Modular division (mod 97), 50% training data
- Training accuracy reaches ~100% at ~10³ steps
- Validation accuracy stays near chance until ~10⁵ steps
- Validation accuracy jumps to ~100% by ~10⁶ steps
- Key: the "grokking" phenomenon — generalization long after memorization

**IMPORTANT**: Figure 1 uses **Adam (no weight decay)** with 10⁶ steps.
The default AdamW wd=1 setting causes much faster grokking.

### Architecture (from Appendix A.1.2)
- Decoder-only transformer, 2 layers, width 128, 4 attention heads
- ~4×10⁵ non-embedding parameters (verified: 409,472)
- Causal attention masking
- Loss and accuracy computed only on the answer token

### Optimization (for Figure 1 specifically)
- **Adam** (NOT AdamW), lr=10⁻³, **weight_decay=0**, β₁=0.9, β₂=0.98
- Linear learning rate warmup over first 10 updates
- Minibatch size 512 (or half training set if smaller)
- Optimization budget: **10⁶** gradient updates

### Data
- Binary operation: x/y (mod 97) for 0 ≤ x < 97, 0 < y < 97
- Total equations: 97 × 96 = 9312
- Training: 50% → 4656 equations
- Validation: remaining 4656
- Format: ⟨x⟩⟨/⟩⟨y⟩⟨=⟩⟨result⟩ (each a separate token)

## Validated Insights
1. **Architecture is correct**: 409,472 non-embedding params matches paper's ~4×10⁵
2. **wd=1 causes fast grokking**: With AdamW wd=1, val acc hits 100% by step 2000. This matches the paper's observation that weight decay dramatically accelerates generalization.
3. **Memorization works**: Train acc reaches 100% at ~500 steps, roughly consistent with paper's ~10³.

## Current Best
commit: 5e4bfdd
metric: train_acc=100%, val_acc=100% (but with wd=1, not matching Figure 1 dynamics)
