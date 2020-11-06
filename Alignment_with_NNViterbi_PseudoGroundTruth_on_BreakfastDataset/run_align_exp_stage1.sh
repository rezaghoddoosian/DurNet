#!/bin/sh


for split_idx in 1 2 3 4
  do
  echo "Runnning train_stage1 $split_idx"
  python train_stage1.py --split_id $split_idx
  echo "Runnning inference_stage1_getsoftmax_vals $split_idx"
  python inference_stage1_getsoftmax_vals.py --split_id $split_idx
  echo "Runnning align_test_split_inference $split_idx"
  python align_test_split_inference.py --split_id 4 --stage_id 1
  echo "Runnning align_test_split $split_idx"
  python align_test_split.py --split_id 4 --stage_id 1
done
