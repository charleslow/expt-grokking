# Autonomous ML Paper Replication

This is an experiment to have an autonomous LLM agent replicate the results of a specific Machine Learning paper.

## Setup

To set up a new replication experiment, work with the user to:

1. **Read the Paper**: The user will provide a PDF of the paper to replicate. Read it thoroughly, focusing on the architecture, hyperparameters, training details, and the specific table or figure to replicate.
2. **Agree on a run tag**: Propose a tag based on the paper and today's date (e.g., `paper_name_mar5`). The branch `replication/<tag>` must not already exist.
3. **Create the branch**: `git checkout -b replication/<tag>` from the current main branch.
4. **Read the in-scope files**: Read the existing codebase to understand the baseline setup. Identify which files are for data preparation, which are for training, and which are for evaluation.
5. **Initialize tracking**: Create a `.replication_logs/` directory with empty `insights.md` and `ideas_queue.md`. In `insights.md`, add a "Current Best" section at the top:
    ```markdown
    ## Current Best
    commit: (none yet -- baseline pending)
    metric: (none)
    ```
6. **Confirm and go**: Confirm setup looks good and the target metric (from the paper's table/figure) is clearly defined.

Once you get confirmation, kick off the experimentation.

## Experimentation

The goal is to get as close to the target results (from a given table or figure in the paper) as possible.

**Run command** (always run in detached mode so you are not blocked):
```shell
nohup python train.py > run.log 2>&1 &
```
*Note: Adjust `python train.py` to the actual training command for the repository.*
*Since it runs in detached mode, you must monitor the `run.log` or use `ps` to check when the process finishes before analyzing results.*

**What you CAN do:**
* Modify the training and model files to match the paper's description.
* Write and run Python analysis scripts to explore training dynamics.
* Generate plots to compare your replication progress against the paper's figures.

**What you CANNOT do:**
* Modify the evaluation harness in a way that artificially inflates the metric. The evaluation must remain rigorous and comparable to the paper.

**Simplicity criterion**: All else being equal, simpler is better. Only add complexity if it is explicitly mentioned in the paper or absolutely necessary to match the results.

**The first run**: Your very first run should always be to establish the baseline with the current code, so you will run the training script as is.

## Tracking and Analysis

All experiment data lives in `.replication_logs/`:

```text
.replication_logs/
  results.tsv              # Master log (append-only TSV)
  insights.md              # Your validated insights + CURRENT BEST tracking
  ideas_queue.md           # Queue of ideas to try next
  <commit>/
    analysis.md            # YOUR investigation notes (you must write to this!)
    run.log                # Full training output
    metrics.jsonl          # Per-step metrics (if implemented)
```

**results.tsv** has 5 tab-separated columns:
```text
commit	metric_value	status	description
```

## The Experiment Loop

The experiment runs on a dedicated branch.

**LOOP FOREVER:**

### 0. REFRESH: Re-read program.md and the Paper
At the start of every iteration, re-read this file (`program.md`) and the relevant sections of the paper to stay aligned with the replication goal.

### 1. RUN: Launch the experiment
Launch the experiment in detached mode:
```shell
nohup python train.py > run.log 2>&1 &
```
Wait for the process to finish by periodically checking `ps` or the end of `run.log`.

### 2. ANALYZE: Read the results
Once finished, extract the key metrics from `run.log` or any output files. Compare them to the target table/figure in the paper.

### 3. INVESTIGATE: Dig deeper
If the results don't match the paper, investigate why. 
* Did the loss plateau too early? 
* Is there a bug in the architecture implementation? 
* Are the hyperparameters exactly as described in the paper?

### 4. SYNTHESIZE: Write your findings to analysis.md
**MANDATORY**: Create/Open `.replication_logs/<commit>/analysis.md` and write your investigation notes. Write:
* What you observed in the logs.
* How this run compares to the paper's target.
* What discrepancies exist and hypotheses for why.

### 5. UPDATE your research notes
**insights.md** -- Your validated knowledge base. Things you've confirmed work or don't work for this replication.
**ideas_queue.md** -- Prioritized list of what to try next based on the paper and your analysis.

### 6. DECIDE: Keep or reject?
Compare the current run against the ALL-TIME BEST in `insights.md`.
* If it's closer to the paper's results: KEEP. Update the "Current Best" in `insights.md`.
* If it's worse or crashed: REJECT. `git reset --hard <best_commit>`.

### 7. IMPLEMENT: Make your next change
Pick the top idea from `ideas_queue.md`. Modify the code. `git commit` your change with a clear description.

### 8. REPEAT
Go back to step 0.

## Important Rules

**Detached Mode**: Always run experiments using `nohup ... &` so you are not blocked. You must actively monitor the process to know when it finishes.

**Crashes**: If a run crashes, fix trivial issues and re-run. If fundamentally broken, log the crash and revert.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, re-read the paper carefully. The loop runs until the human interrupts you, period.

**Be a researcher**: Don't just guess hyperparameters. Read the paper closely. Look for hidden details in the appendix. Form hypotheses based on the paper's text and test them.

