# Reproducing the technical route

Complete [installation](INSTALL.md) and [data preparation](DATA_PREPARATION.md) first. All commands below run from the repository root in the Linux GPU environment. Generated files belong under ignored `data/` or `output/` directories.

## 1. Reference predictions, fusion and ordering

Make a local copy of `tools/cfgs/paper/references_kitti.example.yaml` and provide actual checkpoint paths. Each reference configuration must match its checkpoint. The example paths are locations to fill, not bundled downloads.

```bash
python tools/run_references.py --dataset kitti --data_root data/kitti \
  --references /path/to/references_kitti.yaml \
  --output_dir output/references/kitti --seed 1234
```

This sequentially runs seven references on the clean and corrupted training splits with augmentation disabled, records frame identities, fuses each domain, and writes:

```text
output/references/kitti/<reference>/ori.pkl
output/references/kitti/<reference>/cor.pkl
output/references/kitti/fused_ori.pkl
output/references/kitti/fused_cor.pkl
data/kitti/ImageSets/train_mixed_sorted_uncertainty.txt
```

For ONCE, supply a separate seven-entry YAML containing **ONCE-trained configuration/checkpoint pairs** and use `--dataset once --data_root data/once --output_dir output/references/once`. KITTI weights/configurations are not interchangeable with ONCE ones. Existing ONCE detector configurations are under `tools/cfgs/once_models/`; dataset-specific configurations/checkpoints for all seven references must be supplied or trained. The original server's full ONCE reference bundle is not included.

The release default is confidence-weighted box averaging, maximum member confidence, at least ceil(7/2) reference votes, BEV IoU 0.5 grouping, and classwise BEV NMS 0.1. `--fusion_mode` can select another retained fusion mode. These make the regeneration recipe explicit; the original server fusion settings have not been recovered. Rotated IoU failures raise an error rather than silently changing to axis-aligned IoU.

You can run individual stages with `tools/paper_pipeline.py predict`, `fuse`, or `rank`; each exposes `--help`. Prediction records are checked against both frame and sequence identity, preventing accidental mixing of datasets or reordered caches.

Ranking uses the paper's score:

```text
u = mean(abs(clean_score - corrupted_score))
  + mean(1 - matched_BEV_IoU)
  + 0.5 * unmatched_fraction
```

Matches are classwise, one-to-one, and thresholded at 0.5. The release uses descending-IoU greedy matching. If both sides are empty, the unmatched term is zero. Both domains of a pair receive the same sensitivity. All 2N samples are retained in ascending order; there is no hard-sample subset selection.

## 2. Generate sampling policies

```bash
bash tools/scripts/build_paper_policies.sh kitti
bash tools/scripts/build_paper_policies.sh once
```

The scripts inspect all clean training frames (`--num_frames 0`) and write:

```text
output/policies/kitti_pointrcnn.yaml
output/policies/kitti_second.yaml
output/policies/once_pointrcnn.yaml
output/policies/once_second.yaml
```

PointRCNN starts from full-size sampling budgets while the Ours configuration defines compact channel widths. SECOND uses the compact BEV configuration as the policy source, so policy application preserves those widths. Only training points inside the configured range are used for offline statistics. Budgets are generated once, then remain fixed throughout training and inference.

The recipe uses 0.75 point/neighbor budget caps, 0.6 minimum ratios, seed 1234, and the implementation's rounding/minimum integer bounds. These are source defaults, not recovered original-run hyperparameters. The scripts also support `--target_cfg` and `--match_mode` for experiments that constrain total budgets. Do not mix those modes with the default recipe without recording the choice. Preserve the generated YAML with every run.

## 3. Train from scratch

Example KITTI PointRCNN:

```bash
python tools/train.py --cfg_file tools/cfgs/paper/kitti/pointrcnn_ours.yaml \
  --batch_size 2 --epochs 80 --workers 4 --fix_random_seed --seed 666 \
  --extra_tag paper_seed666 --wo_gpu_stat
```

Use `second_ours.yaml` for SECOND or the `paper/once/` directory for ONCE. Use the epochs and per-GPU batch size appropriate to the selected configuration; the explicit example above is KITTI. Repeat with multiple seeds for accuracy experiments and retain each run's metrics.

Ours configurations explicitly disable **training-frame shuffling**. Point-level augmentation/shuffling inside an individual frame is a separate mechanism and remains configured as in the detector. Do not pass `--pretrained_model` or `--ckpt` for the Ours from-scratch experiment. Use a fresh `--extra_tag`: the upstream trainer can automatically resume checkpoints already present in a run directory.

Check the startup log for the applied policy, 2N training samples, and `SHUFFLE_TRAIN: False`. The canonical protocol uses one GPU. Although the sampler now honors the shuffle flag in distributed mode, distributed batches need not have the same optimization trajectory.

## 4. Evaluate AP on both domains

After training, set `CKPT` to the checkpoint saved under the printed output directory. For the root-relative KITTI PointRCNN example:

```bash
CKPT=output/cfgs/paper/kitti/pointrcnn_ours/paper_seed666/ckpt/checkpoint_epoch_80.pth
python tools/test.py --cfg_file tools/cfgs/paper/kitti/pointrcnn_ours.yaml \
  --ckpt "$CKPT" --batch_size 1 --set DATA_CONFIG.DATA_SPLIT.test val_ori
python tools/test.py --cfg_file tools/cfgs/paper/kitti/pointrcnn_ours.yaml \
  --ckpt "$CKPT" --batch_size 1 --set DATA_CONFIG.DATA_SPLIT.test val_cor
```

Keep the same generated policy when loading the checkpoint. For ONCE, the paper recipe explicitly evaluates Car/Pedestrian/Cyclist using orientation-aware matching, 40 recall samples, and no Vehicle superclass merging. Report the 0–30 m, 30–50 m and >50 m values; the manuscript's range average is their unweighted arithmetic mean. KITTI uses its normal AP_R40 matching and difficulty levels.

Corruption evaluation is one pass over the fixed mixed-corruption validation split, not an average of six independent per-type evaluations.

## 5. Benchmark latency and memory

```bash
python tools/test_speed.py --cfg_file tools/cfgs/paper/kitti/pointrcnn_ours.yaml \
  --ckpt "$CKPT" --batch_size 1 --warmup_iters 100 --num_iters 500 --repeat 5 \
  --set DATA_CONFIG.DATA_SPLIT.test val_ori
python tools/test_speed.py --cfg_file tools/cfgs/paper/kitti/pointrcnn_ours.yaml \
  --ckpt "$CKPT" --batch_size 1 --warmup_iters 100 --num_iters 500 --repeat 5 \
  --set DATA_CONFIG.DATA_SPLIT.test val_cor
```

The timer encloses `model(batch)` with CUDA events and synchronizes each iteration. CPU data loading and the host-to-device transfer happen outside the timed interval. Memory is PyTorch peak **allocated** memory, not total device usage. Record hardware, software, batch size and the per-repeat summaries with the results.

## Keep a run record

Archive the source commit, reference YAML and checkpoint hashes, corruption assignment/simulator versions, sorted training list, policy YAML, effective config, random seeds, final checkpoint and clean/corrupted metrics. They are local experiment artifacts and are intentionally excluded from Git. A new generated recipe is not evidence that the original paper's exact numbers have been reproduced.
