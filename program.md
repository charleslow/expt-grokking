# Autonomous ML Experiment: Grokking in RNNs

This is an experiment to have an autonomous LLM agent discover whether RNN-based models can exhibit the grokking phenomenon (delayed generalization long after memorization), as originally demonstrated with transformers in Power et al. (2022).

## Background

We have already replicated the original grokking result with a decoder-only transformer on modular division (mod 97). The transformer baseline (commit 42753ca) achieves 97.8% validation accuracy at 100k steps with wd=0, exhibiting the characteristic memorize→plateau→generalize curve.

**New goal**: Find an RNN architecture that exhibits grokking on the same task, or rigorously demonstrate that it cannot.

## Setup

To set up the RNN grokking experiment:

1. **Read master.md and the Paper**: Read `master.md` (the master reference — contains motivation, milestones, and project map). Re-read `grokking.pdf`, focusing on the task setup, data splits, and what defines grokking (generalization long after memorization). The RNN exploration is our own extension — the paper only uses transformers.
2. **Read the existing codebase**: Understand the current transformer training code (`train.py`) and data pipeline. The RNN model must use the **exact same task, data splits, and evaluation** so results are directly comparable.
3. **Create the branch**: `git checkout -b replication/rnn_grokking_mar26` from the current main branch.
4. **Initialize tracking**: Reset `.replication_logs/` for this new experiment:
    - Update `insights.md` with a new "Experiment Goal" section and record the transformer baseline as the reference point.
    - Clear `ideas_queue.md` and populate it with an initial prioritized list of RNN architectures to try (e.g., vanilla RNN, LSTM, GRU, and variants).
    - Define `results.tsv` columns: `commit`, `description`, `model_type`, `hidden_dim`, `num_layers`, `num_params`, `weight_decay`, `num_steps`, `train_acc`, `val_acc`, `grokking_observed`, `grokking_onset_step`, `notes`.
5. **Set up the environment**: Use `uv` for Python packaging. Install any additional dependencies needed for RNN models.
6. **Confirm and go**: Confirm setup looks good and begin experimentation.

## Constraints

- **Same task**: Modular division x/y (mod 97), same 50/50 train/val split, same tokenization.
- **Same evaluation**: Loss and accuracy computed only on the answer token, same as the transformer baseline.
- **Same optimizer defaults**: Start with Adam, lr=1e-3, wd=0, β₁=0.9, β₂=0.98, linear warmup 10 steps, batch size 512. You may vary these later but always record them.
- **Parameter budget**: Keep non-embedding parameters in the same order of magnitude as the transformer (~4×10⁵). You may explore smaller/larger models but always log the parameter count.

## Experimentation

The goal is to determine whether RNNs can grok, and if so, under what conditions.

**Run command** (always run in detached mode so you are not blocked):
```shell
nohup uv run python train.py > run.log 2>&1 &
```
*Since it runs in detached mode, you must monitor the `run.log` or use `ps` to check when the process finishes before analyzing results.*

**What you CAN do:**
* Implement RNN model variants (vanilla RNN, LSTM, GRU, minimal gated units, etc.).
* Modify hyperparameters (hidden size, layers, learning rate, weight decay, etc.).
* Write and run Python analysis scripts to explore training dynamics.
* Generate plots comparing RNN training curves against the transformer baseline.
* Vary the training budget — some models may grok faster or slower.

**What you CANNOT do:**
* Modify the evaluation harness in a way that artificially inflates the metric. The evaluation must remain identical to the transformer baseline.
* Change the data splits or task definition.

**Simplicity criterion**: All else being equal, simpler is better. Start with the simplest RNN variant and only add complexity if needed.

**One change at a time**: Each commit should change exactly one variable or hypothesis. Never change multiple things at once.

## Tracking and Analysis

All experiment data lives in `.replication_logs/`:

```text
.replication_logs/
  results.tsv              # Master log (append-only TSV)
  insights.md              # Validated insights + transformer baseline + best RNN result
  ideas_queue.md           # Queue of ideas to try next
  <commit>/
    analysis.md            # YOUR investigation notes (you must write to this!)
    run.log                # Full training output
    metrics.jsonl          # Per-step metrics (if implemented)
```

**results.tsv**: Append a row after every completed run. Never delete rows.

## The Experiment Step

Each iteration of the experiment is a single step. The user will call you in a loop externally.

### 0. REFRESH: Re-read master.md, program.md, and the Paper
At the start of every iteration, re-read `master.md` (the master reference for this project), this file (`program.md`), and the relevant sections of the paper to stay aligned with the experiment goal.

### 1. RUN: Launch the experiment
Launch the experiment in detached mode:
```shell
nohup uv run python train.py > run.log 2>&1 &
```
Wait for the process to finish by periodically checking `ps` or the end of `run.log`.

### 2. ANALYZE: Read the results
Once finished, extract the key metrics from `run.log` or any output files. Compare them to:
- The transformer baseline (does this RNN grok at all?)
- Previous RNN runs (is this better or worse?)

### 3. INVESTIGATE: Dig deeper
If the RNN doesn't grok, investigate why:
* Does it memorize the training set? (If not, the model may be too small or training is broken.)
* Does it memorize but never generalize? (This is the interesting negative result.)
* Does it generalize quickly without a grokking delay? (Different from grokking — record this.)
* Are there signs of grokking starting but not completing? (May need more steps.)

### 4. SYNTHESIZE: Write your findings to analysis.md
**MANDATORY**: Create/Open `.replication_logs/<commit>/analysis.md` and write:
* What you observed in the logs.
* How this run compares to the transformer baseline and previous RNN runs.
* What discrepancies exist and hypotheses for why.

### 5. UPDATE your research notes
**insights.md** — Your validated knowledge base. Things you've confirmed about RNN grokking.
**ideas_queue.md** — Prioritized list of what to try next.

### 6. DECIDE: Keep or reject?
Compare the current run against the ALL-TIME BEST RNN result in `insights.md`.
* If it's closer to exhibiting grokking: KEEP. Update the "Current Best RNN" in `insights.md`.
* If it's worse or crashed: REJECT. Revert the change with `git revert`.

### 7. IMPLEMENT: Make your next change
Pick the top idea from `ideas_queue.md`. Modify the code. `git commit` your change with a clear description.

### 8. CHECK: Should you stop or continue?
After each iteration, assess whether you have enough evidence to write the final report:
- **Stop if**: You have found a clear grokking result with at least one RNN variant AND have tried enough variants/ablations to draw meaningful comparisons. OR you have exhausted your ideas queue and have strong evidence that RNNs cannot grok under these conditions.
- **Continue if**: There are still promising ideas to explore or results are ambiguous.

When you decide to stop, proceed to the Final Report section below.

## Final Report

When experimentation is complete, generate `report.pdf` in the repository root. This is the primary deliverable.

### Report generation
Write the report as `report.tex` using LaTeX, then compile to PDF:
```shell
uv run pdflatex report.tex
```
If `pdflatex` is not available, install it (`sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended`) or use an alternative method (e.g., matplotlib to generate a PDF with text and figures). The key requirement is that `report.pdf` exists and is readable.

### Report structure
The report must be a succinct academic report. Every claim must be rigorous and supported by experimental evidence. Include all details necessary to replicate the results.

1. **Title**: "Grokking in Recurrent Neural Networks: An Empirical Investigation"
2. **Abstract**: 3–5 sentences. State the question, approach, and key finding.
3. **Introduction**: Brief context on grokking (cite Power et al. 2022). State the research question: can RNNs exhibit grokking? Why this matters (architecture-dependence of grokking).
4. **Experimental Setup**:
   - Task description (modular division mod 97, data splits, tokenization).
   - Transformer baseline (architecture, hyperparameters, result).
   - RNN architectures tested (each variant, with exact hyperparameters and parameter counts).
   - Optimization details (optimizer, lr, wd, warmup, batch size, training budget).
   - Hardware and software environment.
5. **Results**:
   - Table of all runs (from `results.tsv`): model type, params, key hyperparameters, final train/val accuracy, whether grokking was observed, grokking onset step.
   - Learning curve plots for the most important runs (train acc and val acc vs. step), compared against the transformer baseline.
   - Highlight the key finding: which RNN(s) grok, or evidence that none do.
6. **Discussion**: Interpret the results. Why might RNNs grok or not? What does this tell us about the grokking phenomenon? Be honest about limitations.
7. **Conclusion**: 2–3 sentences. Restate the key finding and its implication.
8. **Appendix** (optional): Additional plots, full hyperparameter tables, or ablation results.

### Report quality criteria
- **Rigor**: Every claim must be supported by a specific experiment (cite the commit hash or run ID).
- **Reproducibility**: A reader should be able to replicate every result using only the information in the report and the code in this repository.
- **Conciseness**: No filler. Every sentence should convey information. Target 3–5 pages.
- **Honesty**: Report negative results faithfully. If RNNs don't grok, that is a valid finding.

## Important Rules

**Detached Mode**: Always run experiments using `nohup ... &` so you are not blocked. You must actively monitor the process to know when it finishes.

**Crashes**: If a run crashes, fix trivial issues and re-run. If fundamentally broken, log the crash and revert.

**Be a researcher**: Form hypotheses. Test them systematically. Don't just try random hyperparameters — reason about why an RNN might or might not grok based on its inductive biases.

**Environment**: Use `uv` for all Python packaging and running. Do not use `pip` directly.

**Transformer baseline is sacred**: Do not modify the transformer code or results. It serves as the fixed reference point.
