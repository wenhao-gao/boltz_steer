#!/bin/bash

# CUDA_VISIBLE_DEVICES=5 boltz predict inputs/test \
#     --override \
#     --accelerator gpu \
#     --devices 1 \
#     --seed 1 \
#     --out_dir outputs/test_boltz \
#     --steering_strategy no_steering \
#     --output_format mmcif

# CUDA_VISIBLE_DEVICES=5 boltz predict inputs/test \
#     --override \
#     --accelerator gpu \
#     --devices 1 \
#     --seed 1 \
#     --out_dir outputs/test_boltz_steer \
#     --steering_strategy boltz \
#     --output_format mmcif

CUDA_VISIBLE_DEVICES=5 boltz predict inputs/test \
    --override \
    --accelerator gpu \
    --devices 1 \
    --seed 1 \
    --out_dir outputs/test_boltz_fks \
    --steering_strategy fks \
    --output_format mmcif

CUDA_VISIBLE_DEVICES=5 boltz predict inputs/test \
    --override \
    --accelerator gpu \
    --devices 1 \
    --seed 1 \
    --out_dir outputs/test_boltz_gbd \
    --steering_strategy gbd \
    --output_format mmcif

# CUDA_VISIBLE_DEVICES=5 boltz predict inputs/test \
#     --override \
#     --accelerator gpu \
#     --devices 1 \
#     --seed 1 \
#     --out_dir outputs/test_boltz_vm \
#     --steering_strategy vm \
#     --output_format mmcif

# python -m scripts.eval.run_physicalsim_metrics outputs/test_boltz --num-workers 16
# python -m scripts.eval.run_physicalsim_metrics outputs/test_boltz_steer --num-workers 16
python -m scripts.eval.run_physicalsim_metrics outputs/test_boltz_fks --num-workers 16
python -m scripts.eval.run_physicalsim_metrics outputs/test_boltz_gbd --num-workers 16
# python -m scripts.eval.run_physicalsim_metrics outputs/test_boltz_vm --num-workers 16

# Check the output
if [ $? -eq 0 ]; then
  echo "Prediction completed successfully"
else
  echo "Prediction failed"
fi







