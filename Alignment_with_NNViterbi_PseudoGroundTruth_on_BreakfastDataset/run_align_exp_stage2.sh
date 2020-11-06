#!/bin/sh

for split_idx in 1 2 3 4
do
  python train_stage2.py --split_id $split_idx
  python inference_stage2.py --split_id $split_idx
  python align_test_split_inference.py --split_id $split_idx --stage_id 2
  python align_test_split.py --split_id split_idx --stage_id 2
done

python check_accuracies_align_stage_1_2.py --split_id 4