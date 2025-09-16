#!/bin/bash

# Run the prediction
# CUDA_VISIBLE_DEVICES=6,7 boltz predict examples/test/pb_subset \
#     --override \
#     --use_msa_server \
#     --accelerator gpu \
#     --devices 2 \
#     --out_dir test_pb_subset \
#     --output_format mmcif 

# Run the prediction
CUDA_VISIBLE_DEVICES=7 boltz predict examples/test/pb_subset/8slg.yaml \
    --override \
    --accelerator gpu \
    --devices 1 \
    --out_dir test_pb_subset \
    --output_format mmcif 

# Check the output
if [ $? -eq 0 ]; then
  echo "Prediction completed successfully"
else
  echo "Prediction failed"
fi







