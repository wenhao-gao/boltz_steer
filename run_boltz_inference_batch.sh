#!/usr/bin/env bash
#
# Batch launcher for Boltz-2 inference across available GPUs.
# Run directly (no sbatch needed); the script will detect GPUs or honour
# NUM_GPUS and submit parallel workers that share a single output directory.
#
# Required environment variables / arguments:
#   YAML_DIR      Directory containing .yaml inputs.
#
# Optional environment variables:
#   OUT_DIR       Output directory (default: $PWD/boltz_outputs).
#   BOLTZ_CACHE   Cache directory (default: $HOME/.boltz).
#   NUM_GPUS      GPUs to use. If unset, auto-detects.
#   NUM_WORKERS   Data loader workers per GPU (default: floor(SLURM_CPUS_PER_TASK / GPUs) or 2).
#   DIFFUSION_SAMPLES, RECYCLING_STEPS, SAMPLING_STEPS, MAX_PARALLEL_SAMPLES,
#   OUTPUT_FORMAT, USE_POTENTIALS, CHECKPOINT, AFFINITY_CHECKPOINT,
#   USE_MSA_SERVER, MSA_SERVER_URL, MSA_PAIRING_STRATEGY,
#   MSA_SERVER_USERNAME, MSA_SERVER_PASSWORD, API_KEY_HEADER, API_KEY_VALUE
#                 Forwarded to the boltz CLI.
#
set -euo pipefail

log() {
  echo "[boltz-batch] $*"
}

usage() {
  cat <<'HELP'
Usage: YAML_DIR=/path/to/yamls [OUT_DIR=/path/to/out] [NUM_GPUS=N] bash run_boltz_inference_batch.sh

Provides simple batching of Boltz-2 inference runs over multiple GPUs.
Set NUM_GPUS to limit usage; otherwise the script detects an available count.
HELP
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

: "${YAML_DIR:?Please set YAML_DIR to a directory containing .yaml inputs}"
if [[ ! -d "$YAML_DIR" ]]; then
  echo "YAML_DIR '$YAML_DIR' is not a directory" >&2
  exit 1
fi

OUT_DIR=${OUT_DIR:-"$PWD/boltz_outputs"}
BOLTZ_CACHE=${BOLTZ_CACHE:-"$HOME/.boltz"}

# Detect GPU ids/count available on this node.
detect_physical_gpus() {
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    echo "$SLURM_GPUS_ON_NODE"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local tmp=${CUDA_VISIBLE_DEVICES//,/ }
    set -- $tmp
    echo $#
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count=$(command nvidia-smi -L | wc -l)
    if (( count > 0 )); then
      echo "$count"
      return
    fi
  fi
  echo 0
}

collect_candidate_gpu_ids() {
  local fallback_count=${1:-0}
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local IFS=','
    read -r -a ids <<<"${CUDA_VISIBLE_DEVICES}"
    for id in "${ids[@]}"; do
      [[ -z "$id" ]] && continue
      echo "$id"
    done
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local smi_output
    smi_output=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$smi_output" ]]; then
      printf '%s\n' "$smi_output" | awk 'NF {gsub(/ /,""); print}'
    fi
    return
  fi
  if (( fallback_count > 0 )); then
    for ((i=0; i<fallback_count; i++)); do
      echo "$i"
    done
  fi
}

count_unused_gpus() {
  local ids=("$@")
  if (( ${#ids[@]} == 0 )); then
    echo 0
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ${#ids[@]}
    return
  fi
  mapfile -t busy_uuids < <(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | awk 'NF' || true)
  if (( ${#busy_uuids[@]} == 0 )); then
    echo ${#ids[@]}
    return
  fi
  declare -A busy_set=()
  local uuid
  for uuid in "${busy_uuids[@]}"; do
    uuid=${uuid//[[:space:]]/}
    [[ -z "$uuid" ]] && continue
    busy_set[$uuid]=1
  done
  declare -A index_to_uuid=()
  while IFS=',' read -r idx uuid; do
    idx=${idx//[[:space:]]/}
    uuid=${uuid//[[:space:]]/}
    [[ -z "$idx" || -z "$uuid" ]] && continue
    index_to_uuid[$idx]=$uuid
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)
  local unused=0
  local id uuid_lookup
  for id in "${ids[@]}"; do
    id=${id//[[:space:]]/}
    [[ -z "$id" ]] && continue
    uuid_lookup=${index_to_uuid[$id]}
    if [[ -z "$uuid_lookup" || -z ${busy_set[$uuid_lookup]:-} ]]; then
      ((unused++))
    fi
  done
  echo $unused
}

available_gpu_count=$(detect_physical_gpus)

mapfile -t CANDIDATE_GPU_IDS < <(collect_candidate_gpu_ids "$available_gpu_count")
missing_gpu_metadata=0
if (( ${#CANDIDATE_GPU_IDS[@]} == 0 )); then
  missing_gpu_metadata=1
fi
available_gpu_count=${#CANDIDATE_GPU_IDS[@]}
unused_gpu_count=$(count_unused_gpus "${CANDIDATE_GPU_IDS[@]}")

if [[ -n "${NUM_GPUS:-}" ]]; then
  if ! [[ $NUM_GPUS =~ ^[0-9]+$ ]] || (( NUM_GPUS < 1 )); then
    echo "NUM_GPUS must be a positive integer" >&2
    exit 1
  fi
  TARGET_GPUS=$NUM_GPUS
else
  if (( missing_gpu_metadata )); then
    log "Warning: unable to determine unused GPUs; defaulting to 1 concurrent submission. Set NUM_GPUS to adjust."
    TARGET_GPUS=1
    unused_gpu_count=$TARGET_GPUS
  else
    TARGET_GPUS=$unused_gpu_count
    if (( TARGET_GPUS < 1 )); then
      echo "No unused GPUs detected. Set NUM_GPUS to override or try again later." >&2
      exit 1
    fi
  fi
fi

if (( ! missing_gpu_metadata && available_gpu_count > 0 && TARGET_GPUS > available_gpu_count )); then
  log "Requested $TARGET_GPUS GPU(s) but only $available_gpu_count visible; capping to $available_gpu_count"
  TARGET_GPUS=$available_gpu_count
fi

if (( ! missing_gpu_metadata )); then
  log "Detected $unused_gpu_count unused GPU(s) out of $available_gpu_count visible."
fi

mapfile -t YAML_FILES < <(find "$YAML_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
if (( ${#YAML_FILES[@]} == 0 )); then
  echo "No .yaml files found in $YAML_DIR" >&2
  exit 1
fi

if (( TARGET_GPUS > ${#YAML_FILES[@]} )); then
  TARGET_GPUS=${#YAML_FILES[@]}
fi

TOTAL_CPUS=${SLURM_CPUS_PER_TASK:-0}
if (( TOTAL_CPUS <= 0 )); then
  TOTAL_CPUS=$(nproc --all 2>/dev/null || echo 2)
fi

if [[ -n "${NUM_WORKERS:-}" ]]; then
  WORKERS_PER_GPU=$NUM_WORKERS
  if ! [[ $WORKERS_PER_GPU =~ ^[0-9]+$ ]] || (( WORKERS_PER_GPU < 1 )); then
    echo "NUM_WORKERS must be a positive integer" >&2
    exit 1
  fi
else
  WORKERS_PER_GPU=$(( TOTAL_CPUS / TARGET_GPUS ))
  if (( WORKERS_PER_GPU < 1 )); then WORKERS_PER_GPU=1; fi
fi

mkdir -p "$OUT_DIR" "$BOLTZ_CACHE"

export OUT_DIR BOLTZ_CACHE WORKERS_PER_GPU DIFFUSION_SAMPLES RECYCLING_STEPS SAMPLING_STEPS \
  MAX_PARALLEL_SAMPLES OUTPUT_FORMAT USE_POTENTIALS CHECKPOINT AFFINITY_CHECKPOINT \
  USE_MSA_SERVER MSA_SERVER_URL MSA_PAIRING_STRATEGY MSA_SERVER_USERNAME MSA_SERVER_PASSWORD \
  API_KEY_HEADER API_KEY_VALUE

export OMP_NUM_THREADS=$WORKERS_PER_GPU
export MKL_NUM_THREADS=$WORKERS_PER_GPU

log "Using $TARGET_GPUS GPU(s)"
log "Found ${#YAML_FILES[@]} YAML file(s) in $YAML_DIR"
log "Outputs -> $OUT_DIR"
log "Cache   -> $BOLTZ_CACHE"
log "Workers per GPU -> $WORKERS_PER_GPU"

TMP_DIR=$(mktemp -d -p "$PWD" boltz_batch_tmp.XXXXXX)
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

for idx in $(seq 0 $((TARGET_GPUS - 1))); do
  : >"$TMP_DIR/chunk_${idx}.txt"
done

for i in "${!YAML_FILES[@]}"; do
  gpu=$(( i % TARGET_GPUS ))
  printf '%s\n' "${YAML_FILES[$i]}" >>"$TMP_DIR/chunk_${gpu}.txt"
done

WORKER_SCRIPT="$TMP_DIR/boltz_worker.slurm"
cat <<WORKER >"$WORKER_SCRIPT"
#!/usr/bin/env bash
#SBATCH --job-name=boltz2_worker
#SBATCH --output=boltz_worker-%j.out
#SBATCH --error=boltz_worker-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=${WORKERS_PER_GPU}
#SBATCH --mem=32G

set -euo pipefail
chunk_file="\$1"
gpu_label="\$2"
while IFS= read -r input || [[ -n "\$input" ]]; do
  [[ -z "\$input" ]] && continue
  echo "[boltz-gpu-\${gpu_label}] \$(date "+%F %T") -> \$input"
  if command -v boltz >/dev/null 2>&1; then
    cmd=(boltz predict "\$input")
  else
    cmd=(python -m boltz.main predict "\$input")
  fi
  cmd+=(
    --out_dir "$OUT_DIR"
    --cache "$BOLTZ_CACHE"
    --devices 1
    --accelerator gpu
    --num_workers "$WORKERS_PER_GPU"
    --diffusion_samples "${DIFFUSION_SAMPLES:-1}"
    --recycling_steps "${RECYCLING_STEPS:-3}"
    --sampling_steps "${SAMPLING_STEPS:-200}"
    --max_parallel_samples "${MAX_PARALLEL_SAMPLES:-5}"
    --output_format "${OUTPUT_FORMAT:-mmcif}"
  )
  if [[ "${USE_POTENTIALS:-0}" == "1" || "${USE_POTENTIALS:-}" == "true" ]]; then
    cmd+=(--use_potentials)
  fi
  if [[ -n "${CHECKPOINT:-}" ]]; then
    cmd+=(--checkpoint "${CHECKPOINT}")
  fi
  if [[ -n "${AFFINITY_CHECKPOINT:-}" ]]; then
    cmd+=(--affinity_checkpoint "${AFFINITY_CHECKPOINT}")
  fi
  if [[ "${USE_MSA_SERVER:-0}" == "1" || "${USE_MSA_SERVER:-}" == "true" ]]; then
    cmd+=(
      --use_msa_server
      --msa_server_url "${MSA_SERVER_URL:-https://api.colabfold.com}"
      --msa_pairing_strategy "${MSA_PAIRING_STRATEGY:-greedy}"
    )
    if [[ -n "${MSA_SERVER_USERNAME:-}" && -n "${MSA_SERVER_PASSWORD:-}" ]]; then
      cmd+=(
        --msa_server_username "${MSA_SERVER_USERNAME}"
        --msa_server_password "${MSA_SERVER_PASSWORD}"
      )
    fi
    if [[ -n "${API_KEY_VALUE:-}" ]]; then
      if [[ -n "${API_KEY_HEADER:-}" ]]; then
        cmd+=(--api_key_header "${API_KEY_HEADER}")
      fi
      cmd+=(--api_key_value "${API_KEY_VALUE}")
    fi
  fi
  echo "[boltz-gpu-\${gpu_label}] Running: \${cmd[*]}"
  "\${cmd[@]}"
done <"\$chunk_file"
WORKER
chmod +x "$WORKER_SCRIPT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch command not found; please ensure Slurm client tools are available." >&2
  exit 1
fi

launch_gpu_job() {
  local chunk_file=$1
  local gpu_index=$2
  if [[ ! -s "$chunk_file" ]]; then
    return
  fi
  local count
  count=$(wc -l <"$chunk_file")
  log "GPU $gpu_index submitting ${count} file(s)"
  local -a submit_args=("$WORKER_SCRIPT" "$chunk_file" "$gpu_index")
  (
    job_id=$(sbatch --parsable --wait --job-name="boltz2_gpu${gpu_index}" "${submit_args[@]}")
    status=$?
    if (( status == 0 )); then
      log "Job ${job_id} finished for GPU $gpu_index"
    else
      log "sbatch failed for GPU $gpu_index (exit $status)"
    fi
    exit $status
  ) &
  echo $!
}

job_pids=()
for idx in $(seq 0 $((TARGET_GPUS - 1))); do
  pid=$(launch_gpu_job "$TMP_DIR/chunk_${idx}.txt" "$idx")
  if [[ -n "$pid" ]]; then
    job_pids+=($pid)
  fi
done

status=0
for pid in "${job_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

log "Batch inference completed with status $status"
exit $status
