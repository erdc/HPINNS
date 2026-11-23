#!/bin/bash
# Run the Optuna search as N processes, then train the best trial once.
#
# Each process is one copy of run_pinn.py. Together they stop near n_trials
# completed trials in user_input.py. They share the SQLite study. Only the
# last command writes the final weights and archives.
# Quantum trials run a compiled JAX circuit. OMP_NUM_THREADS=1 keeps each
# process from spawning extra thread pools.
#
# Usage, from this directory:
#   ./run_cores.sh 8

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 N" >&2
  echo "N is the number of search workers, a positive integer." >&2
  exit 1
fi

cores="$1"

cd "$(dirname "$0")"
mkdir -p logs
rm -f logs/trial_table_rows

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

pids=()

echo "Starting ${cores} search workers. One line prints here as each trial finishes."
echo "Full epoch logs: logs/worker_0.txt ... logs/worker_$((cores - 1)).txt"

for i in $(seq 0 $((cores - 1))); do
  python3 run_pinn.py --search-only --worker-id "$i" > "logs/worker_${i}.txt" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "One or more workers failed. See logs/worker_*.txt. Final training was not started." >&2
  exit 1
fi

echo "Search finished. Training the best completed trial."
python3 run_pinn.py --finalize
