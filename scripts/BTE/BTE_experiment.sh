#!/bin/bash
n_shifts=6
for ((target_shift=0; target_shift<$n_shifts; target_shift++))
do
  echo "Starting Shift: $target_shift"
  CUDA_VISIBLE_DEVICES=3 python BTE_experiment.py --target_shift $target_shift --n_shifts $n_shifts
done
