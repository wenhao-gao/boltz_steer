#!/usr/bin/env bash
#
# Slurm script to run Boltz-2 inference.
#
# Usage examples:
#   sbatch --export=INPUT_PATH=/path/to/input.yaml scripts/slurm/boltz2_infer.sbatch
#   sbatch --export=INPUT_PATH=/path/to/folder,OUT_DIR=/path/to/out,BOLTZ_CACHE=/path/to/cache \
#          scripts/slurm/boltz2_infer.sbatch
#   sbatch --export=INPUT_PATH=/path/in,USE_MSA_SERVER=1,MSA_SERVER_URL=https://api.colabfold.com \
#          scripts/slurm/boltz2_infer.sbatch
#
# Required export variables:
#   - INPUT_PATH: path to a single .yaml/.fasta or a directory of inputs
#
# Optional export variables (sane defaults provided):
#   - OUT_DIR: output directory (default: $PWD/boltz_outputs)
#   - BOLTZ_CACHE: cache dir for models/data (default: $HOME/.boltz)
#   - DEVICES: number of GPUs to use (default: $SLURM_GPUS_ON_NODE or 1)
#   - NUM_WORKERS: dataloader workers (default: $SLURM_CPUS_PER_TASK or 2)
#   - DIFFUSION_SAMPLES: number of samples (default: 1)
#   - RECYCLING_STEPS: recycling steps (default: 3)
#   - SAMPLING_STEPS: diffusion sampling steps (default: 200)
#   - MAX_PARALLEL_SAMPLES: max samples evaluated in parallel (default: 5)
#   - OUTPUT_FORMAT: mmcif or pdb (default: mmcif)
#   - USE_POTENTIALS: set 1 to enable inference-time potentials (default: 0)
#   - CHECKPOINT: path to custom Boltz-2 checkpoint (default: implicit built-in)
#   - AFFINITY_CHECKPOINT: path to custom Boltz-2 affinity checkpoint (default: implicit built-in)
#   - USE_MSA_SERVER: set 1 to enable mmseqs2 server (default: 0)
#   - MSA_SERVER_URL: URL for msa server (default: https://api.colabfold.com)
#   - MSA_PAIRING_STRATEGY: greedy or complete (default: greedy)
#   - MSA_SERVER_USERNAME / MSA_SERVER_PASSWORD: basic auth (optional)
#   - API_KEY_HEADER / API_KEY_VALUE: API key auth for MSA server (optional)
#
# SBATCH resources — adjust to your cluster/queue.
# NOTE: If your site uses account/partition/QoS, uncomment and set accordingly.
#SBATCH --job-name=boltz2_infer
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=32G
##SBATCH --partition=gpu
##SBATCH --account=<your_account>
##SBATCH --qos=<your_qos>

set -euo pipefail

echo "[Slurm] Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname)"
date

#
# Load/activate GPU environment
# Assumes this script sets up CUDA, Python/conda env, and compiler libs as needed.
#
if [[ -f "$HOME/gpu_env_multiple.sh" ]]; then
  echo "[Slurm] Sourcing ~/gpu_env_multiple.sh"
  # shellcheck disable=SC1090
  source "$HOME/gpu_env_multiple.sh"
else
  echo "[WARN] ~/gpu_env_multiple.sh not found. Proceeding with current environment."
fi

#
# Inputs and defaults
#
: "${INPUT_PATH:?Please set INPUT_PATH to a .yaml/.fasta file or a directory}"
OUT_DIR=${OUT_DIR:-"$PWD/boltz_outputs"}
BOLTZ_CACHE=${BOLTZ_CACHE:-"$HOME/.boltz"}
DEVICES=${DEVICES:-"${SLURM_GPUS_ON_NODE:-1}"}
NUM_WORKERS=${NUM_WORKERS:-"${SLURM_CPUS_PER_TASK:-2}"}
DIFFUSION_SAMPLES=${DIFFUSION_SAMPLES:-1}
RECYCLING_STEPS=${RECYCLING_STEPS:-3}
SAMPLING_STEPS=${SAMPLING_STEPS:-200}
MAX_PARALLEL_SAMPLES=${MAX_PARALLEL_SAMPLES:-5}
OUTPUT_FORMAT=${OUTPUT_FORMAT:-mmcif}
USE_POTENTIALS=${USE_POTENTIALS:-0}
CHECKPOINT=${CHECKPOINT:-}
AFFINITY_CHECKPOINT=${AFFINITY_CHECKPOINT:-}
USE_MSA_SERVER=${USE_MSA_SERVER:-0}
MSA_SERVER_URL=${MSA_SERVER_URL:-https://api.colabfold.com}
MSA_PAIRING_STRATEGY=${MSA_PAIRING_STRATEGY:-greedy}
MSA_SERVER_USERNAME=${MSA_SERVER_USERNAME:-}
MSA_SERVER_PASSWORD=${MSA_SERVER_PASSWORD:-}
API_KEY_HEADER=${API_KEY_HEADER:-}
API_KEY_VALUE=${API_KEY_VALUE:-}

mkdir -p "$OUT_DIR" "$BOLTZ_CACHE"

# Threading hints for PyTorch/NumPy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export BOLTZ_CACHE

echo "[Slurm] Using devices: $DEVICES GPU(s)"
echo "[Slurm] INPUT_PATH: $INPUT_PATH"
echo "[Slurm] OUT_DIR:    $OUT_DIR"
echo "[Slurm] BOLTZ_CACHE: $BOLTZ_CACHE"

#
# Assemble CLI
#
cmd=(
  boltz predict "$INPUT_PATH"
  --out_dir "$OUT_DIR"
  --cache "$BOLTZ_CACHE"
  --devices "$DEVICES"
  --accelerator gpu
  --recycling_steps "$RECYCLING_STEPS"
  --sampling_steps "$SAMPLING_STEPS"
  --diffusion_samples "$DIFFUSION_SAMPLES"
  --max_parallel_samples "$MAX_PARALLEL_SAMPLES"
  --output_format "$OUTPUT_FORMAT"
  --num_workers "$NUM_WORKERS"
)

if [[ -n "$CHECKPOINT" ]]; then
  cmd+=(--checkpoint "$CHECKPOINT")
fi

if [[ -n "$AFFINITY_CHECKPOINT" ]]; then
  cmd+=(--affinity_checkpoint "$AFFINITY_CHECKPOINT")
fi

if [[ "$USE_POTENTIALS" == "1" || "$USE_POTENTIALS" == "true" ]]; then
  cmd+=(--use_potentials)
fi

if [[ "$USE_MSA_SERVER" == "1" || "$USE_MSA_SERVER" == "true" ]]; then
  cmd+=(
    --use_msa_server
    --msa_server_url "$MSA_SERVER_URL"
    --msa_pairing_strategy "$MSA_PAIRING_STRATEGY"
  )
  if [[ -n "$MSA_SERVER_USERNAME" && -n "$MSA_SERVER_PASSWORD" ]]; then
    cmd+=(
      --msa_server_username "$MSA_SERVER_USERNAME"
      --msa_server_password "$MSA_SERVER_PASSWORD"
    )
  fi
  if [[ -n "$API_KEY_VALUE" ]]; then
    # Header key defaults to X-API-Key if not set
    if [[ -n "$API_KEY_HEADER" ]]; then
      cmd+=(--api_key_header "$API_KEY_HEADER")
    fi
    cmd+=(--api_key_value "$API_KEY_VALUE")
  fi
fi

#
# Fallback to python -m if CLI entrypoint not on PATH
#
if ! command -v boltz >/dev/null 2>&1; then
  echo "[Slurm] 'boltz' CLI not found on PATH; using 'python -m boltz.main'"
  cmd=(python -m boltz.main predict "${cmd[@]:2}")
fi

echo "[Slurm] Running: ${cmd[*]}"

#
# Launch with srun to bind resources correctly
#
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus="$DEVICES" "${cmd[@]}"

rc=$?
echo "[Slurm] Exit code: $rc"
date
exit $rc

