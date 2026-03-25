#!/bin/bash
# Run claude -p in a loop for N hours
# Usage: ./run_loop.sh <hours>

if [ -z "$1" ]; then
    echo "Usage: ./run_loop.sh <hours>"
    exit 1
fi

HOURS=$1
END_TIME=$(( $(date +%s) + HOURS * 3600 ))
ITERATION=0

echo "Starting experiment loop for $HOURS hours (until $(date -d @$END_TIME '+%Y-%m-%d %H:%M:%S'))"

while [ $(date +%s) -lt $END_TIME ]; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "=== Iteration $ITERATION | $(date '+%Y-%m-%d %H:%M:%S') | $(( (END_TIME - $(date +%s)) / 60 )) min remaining ==="
    echo ""

    # Kill any leftover training processes from a previous iteration
    pkill -f "python train.py" 2>/dev/null || true
    sleep 1

    claude -p \
        --allowedTools "Read,Write,Edit,Bash,Glob,Grep" \
        "Read program.md thoroughly, then execute one full experiment step (sections 0-7). You are iteration $ITERATION." \
        || echo "WARNING: claude exited with non-zero status on iteration $ITERATION, continuing..."

    echo ""
    echo "=== Iteration $ITERATION completed at $(date '+%Y-%m-%d %H:%M:%S') ==="
done

echo ""
echo "Loop finished after $ITERATION iterations."
