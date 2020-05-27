#!/bin/bash
source activate tensorflow_p36
n_shifts=6
if [[ $# == 0 ]]
then
    for ((target_shift=0; target_shift<$n_shifts; target_shift++))
    do
      echo "Starting Shift: $target_shift"
      CUDA_VISIBLE_DEVICES=3 python experiment.py --target_shift $target_shift --n_shifts $n_shifts &
    done
else
    for target_shift in $*
    do
      echo "Starting Shift: $target_shift"
      dvc=$(($target_shift % 4))
      CUDA_VISIBLE_DEVICES=$dvc python experiment.py --target_shift $target_shift --n_shifts $n_shifts &
    done
fi