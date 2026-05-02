#!/bin/bash
# ----------------------------------------------------------------------
# Tier-2 baseline tuning — SLURM array submitter.
#
# Submits 5 baselines x 10 datasets = 50 tasks as a single array job.
# Each task runs Optuna with 15 trials on its (baseline, dataset) pair
# and appends one row to baselines/results/tier2_tuned.csv.
#
# Launch (from the repo root):
#     sbatch baselines/submit_tier2.sh
#
# Monitor:
#     squeue -u $USER -j <jobid>
#     tail -f results/tier2/<baseline>_<dataset>.out
#
# After all 50 tasks complete:
#     /Users/arbili/opt/anaconda3/envs/bgu/bin/python baselines/tier2_aggregate.py
# ----------------------------------------------------------------------
#SBATCH --job-name=tier2_tune
#SBATCH --array=0-49
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=results/tier2/slurm_%A_%a.out
#SBATCH --error=results/tier2/slurm_%A_%a.err

set -euo pipefail

# --- Repo root --------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

mkdir -p results/tier2
mkdir -p baselines/results

# --- Conda env (per memory/feedback_python_path.md) -------------------
PY=/Users/arbili/opt/anaconda3/envs/bgu/bin/python
# On the cluster the user must adjust PY to the cluster-side bgu env path.
# The fallback below uses `conda activate bgu` if the explicit binary
# doesn't exist on this host.
if [[ ! -x "${PY}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate bgu
        PY=$(which python)
    fi
fi
echo "Python: ${PY}"
"${PY}" -V

# --- Build the (baseline, dataset) grid -------------------------------
BASELINES=(axgb pufe ocds emli gbdt_il)
DATASETS=(BankLoanSta Weather DiabetesRecord CreditRisk HRAnalytics \
          ClientRecord MovieAugV2 WeatherAUS WIDS FlightDelay)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
N_DATASETS=${#DATASETS[@]}            # 10
B_IDX=$(( TASK_ID / N_DATASETS ))     # 0..4
D_IDX=$(( TASK_ID % N_DATASETS ))     # 0..9
BASELINE=${BASELINES[$B_IDX]}
DATASET=${DATASETS[$D_IDX]}

OUT_CSV="${REPO_ROOT}/baselines/results/tier2_tuned.csv"
LOG_OUT="${REPO_ROOT}/results/tier2/${BASELINE}_${DATASET}.out"

echo "TASK ${TASK_ID}: baseline=${BASELINE}  dataset=${DATASET}"
echo "Output CSV: ${OUT_CSV}"
echo "Log:        ${LOG_OUT}"

# --- Sandbox CWD per-task (avoid model-file collisions on shared FS) --
SANDBOX="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tier2_${TASK_ID}_$$"
mkdir -p "${SANDBOX}"
cd "${SANDBOX}"

# --- Threading (32 CPU allocation) ------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export EMLI_TORCH_THREADS=${SLURM_CPUS_PER_TASK:-32}

# --- Run --------------------------------------------------------------
"${PY}" -m baselines.tune_baseline \
    --baseline "${BASELINE}" \
    --dataset "${DATASET}" \
    --n_trials 15 \
    --seed 42 \
    --out "${OUT_CSV}" \
    2>&1 | tee "${LOG_OUT}"

echo "TASK ${TASK_ID} DONE."
