#!/usr/bin/env bash
#
# Slurm script to run Boltz-2 inference.
#
#SBATCH --job-name=boltz2_infer
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

echo "[Slurm] Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname)"
date

set +e
boltz predict inputs/test \
    --override \
    --accelerator gpu \
    --devices 1 \
    --out_dir outputs/test_boltz_steer \
    --steering_strategy boltz \
    --output_format mmcif
rc=$?
set -e

set +e
boltz predict inputs/test \
    --override \
    --accelerator gpu \
    --devices 1 \
    --out_dir outputs/test_boltz
    --steering_strategy no_steering \
    --output_format mmcif
rc=$?
set -e

python -m scripts.eval.run_physicalsim_metrics outputs/test_boltz_steer --num-workers 16
python -m scripts.eval.run_physicalsim_metrics outputs/test_boltz --num-workers 16

date
exit $rc



