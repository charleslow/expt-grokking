# Master Reference

**This is the master reference for the project. Read this file first at the start of every conversation and every experiment iteration. Only record invariant, durable facts here — not run-level details, hyperparameters, or anything derivable from the code or git history.**

## Motivation

Grokking — generalization long after memorization — was demonstrated by Power et al. (2022) using transformers on algorithmic tasks (`grokking.pdf`). This project investigates whether grokking is architecture-dependent: can simpler recurrent models (RNNs, LSTMs, GRUs) exhibit the same phenomenon?

## Milestones

| Milestone | Commit | Description |
|-----------|--------|-------------|
| Transformer baseline | `42753ca` | Transformer replicates grokking on modular division (mod 97). Reference point for all future comparisons. |
| RNN exploration start | (pending) | Branch: `replication/rnn_grokking_mar26`. Goal: find an RNN variant that groks, or rigorously show none can. |

*Update this table as new milestones are reached.*

## Project Map

| File | Purpose |
|------|---------|
| `master.md` | This file. Read first. |
| `program.md` | Experiment protocol and agent loop instructions. |
| `grokking.pdf` | The original paper. |
| `train.py` | Training script. |
| `.replication_logs/` | Run-level tracking: `insights.md`, `ideas_queue.md`, `results.tsv`. |
| `report.tex` / `report.pdf` | Final deliverable (generated at end of experimentation). |
