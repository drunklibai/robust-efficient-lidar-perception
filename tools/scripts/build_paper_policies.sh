#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root after local data preparation.
dataset="${1:?Usage: bash tools/scripts/build_paper_policies.sh kitti|once}"
case "$dataset" in
  kitti) search=tools/search_sampling_policy.py ;;
  once) search=tools/search_sampling_policy_once.py ;;
  *) echo "Expected kitti or once" >&2; exit 2 ;;
esac
for model in pointrcnn second; do
  # PointRCNN budgets start from the full-size network. SECOND keeps the
  # compact BEV widths while estimating its voxel budget from clean data.
  variant=original
  if [ "$model" = second ]; then variant=ours; fi
  python "$search" \
    --cfg_file "tools/cfgs/paper/$dataset/${model}_${variant}.yaml" \
    --data_path "data/$dataset" --split train_mixed_reference_ori \
    --num_frames 0 --seed 1234 \
    --budget_npoint_ratio 0.75 --budget_nsample_ratio 0.75 \
    --output "output/policies/${dataset}_${model}.yaml"
done
