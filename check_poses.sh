#!/bin/bash

python -m scripts.eval.run_physicalsim_metrics outputs/posebusters --num-workers 32

python -m scripts.eval.run_physicalsim_metrics outputs/posebusters_steer --num-workers 32