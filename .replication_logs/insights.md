# Replication Insights — Grokking (Power et al. 2022)

## Paper Target
**Figure 1 (left)**: Modular division (mod 97), 50% training data
- Training accuracy reaches ~100% at ~10³ steps
- Validation accuracy stays near chance until ~10⁵ steps
- Validation accuracy jumps to ~100% by ~10⁶ steps
- Key: the "grokking" phenomenon — generalization long after memorization

### Architecture (from Appendix A.1.2)
- Decoder-only transformer, 2 layers, width 128, 4 attention heads
- ~4×10⁵ non-embedding parameters
- Causal attention masking
- Loss and accuracy computed only on the answer token

### Optimization
- AdamW, lr=10⁻³, weight_decay=1, β₁=0.9, β₂=0.98
- Linear learning rate warmup over first 10 updates
- Minibatch size 512 (or half training set if smaller)
- Optimization budget: 10⁶ gradient updates

### Data
- Binary operation: x/y (mod 97) for 0 ≤ x < 97, 0 < y < 97
- Total equations: 97 × 96 = 9312
- Training: 50% → 4656 equations
- Validation: remaining 4656
- Format: ⟨x⟩⟨/⟩⟨y⟩⟨=⟩⟨result⟩ (each a separate token)

## Current Best
commit: (none yet)
metric: (none)
