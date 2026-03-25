# Ideas Queue

1. **Baseline**: Implement the exact architecture and training setup from the paper. Use modular division mod 97, 50% data, decoder-only transformer with 2 layers/128 width/4 heads, AdamW with lr=1e-3, wd=1, 10⁶ steps.
2. **Reduce steps for faster iteration**: Try 10⁵ steps first to see if memorization phase works, then scale up.
3. **Try modular addition first**: Simpler operation, might grok faster — good for debugging.
4. **Weight decay ablation**: Paper emphasizes weight decay is crucial for grokking.
