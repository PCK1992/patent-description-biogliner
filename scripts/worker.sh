#!/usr/bin/env bash
set -euo pipefail
umask 077
export HF_HOME="$PWD/.hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_JIT=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset PYTHONOPTIMIZE  # the Helix runner uses Python assertions for its audits
py=${PYTHON:?Use submit.sh with an active virtual environment}
case "$1" in
    year) exec "$py" -u code/build_year.py "${2:-$((1976 + SLURM_ARRAY_TASK_ID))}" ;;
    pilot) exec "$py" -u code/score.py 1976 --pilot ;;
    full) exec "$py" -u code/score.py "$((1976 + SLURM_ARRAY_TASK_ID / 8))" --nshards 8 --shard "$((SLURM_ARRAY_TASK_ID % 8))" ;;
    audit) shift; exec "$py" -u code/audit.py "$@" ;;
    *) exit 2 ;;
esac
