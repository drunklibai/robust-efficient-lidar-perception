# Baselines and ablations

The first release prioritizes the complete Ours technical route. It also retains baseline implementations and candidate configurations; the retained historical filenames do not establish a one-to-one mapping to the paper's original checkpoints.

## Baselines

| Method | Entry points |
| --- | --- |
| Original | `tools/cfgs/paper/{kitti,once}/*_original.yaml`, with matching full-size released weights |
| Retrain | `tools/cfgs/paper/{kitti,once}/*_retrain.yaml`: full-size detector, complete mixed list, ordinary shuffled training |
| GMP | `tools/export_gmp_ckpt.py`, `pcdet/utils/gmp_pruning_utils.py`, `*gmp*.yaml` |
| LTH/IMP | `tools/export_lth_imp_ckpt.py`, `pcdet/utils/imp_pruning_utils.py`, `*lth_imp*.yaml` |
| Network Slimming | `tools/export_slimming_pruned_cfg.py`, `tools/export_slimming_pruned_cfg_second.py`, `pcdet/utils/slimming_utils.py`, `*slimming*` configs |

Use `--help` on the exporter before running it. Magnitude masks, weight rewinding and BN-gamma channel selection require the corresponding training checkpoints. For Slimming, train with its regularizer, export both the smaller configuration and transferred checkpoint, and fine-tune that pair. The main Ours route instead trains the compact detector from scratch.

Historical `tools/cfgs/{kitti,once}_models/` baseline configurations include multiple pruning ratios, regularization strengths and training variants. Before using one, set its data path, complete mixed split, desired shuffle policy, clean/corrupted evaluation split, and ONCE R40 settings as in the paper recipes. Do not infer a baseline's data protocol from the word `mixed` in its filename. Select and report a concrete configuration rather than claiming all retained variants reproduce the same table row.

## Training order comparison

Using the fused training predictions, generate a separate list for each control:

```bash
python tools/paper_pipeline.py rank --dataset kitti --data_root data/kitti \
  --clean output/references/kitti/fused_ori.pkl \
  --corrupted output/references/kitti/fused_cor.pkl \
  --order confidence --output data/kitti/ImageSets/train_mixed_confidence.txt
python tools/paper_pipeline.py rank --dataset kitti --data_root data/kitti \
  --clean output/references/kitti/fused_ori.pkl \
  --corrupted output/references/kitti/fused_cor.pkl \
  --order random --seed 1234 --output data/kitti/ImageSets/train_mixed_random.txt
```

Train the full-size PointRCNN configuration on the selected complete list. For a fixed order comparison, set `SHUFFLE_TRAIN: false` for all three orders. The confidence control uses mean top-three confidence within each present class, averages across present classes, and sorts individual clean/corrupted samples from high to low confidence. The sensitivity route sorts pairs from low to high sensitivity.

## Component ablations

- **No ensemble reference:** use one reference's clean/corrupted predictions directly with `rank`, keeping the compact architecture and all data. Record which reference is used.
- **No data prioritization:** train the same compact model/policy with `train_mixed_nosel` and `SHUFFLE_TRAIN: true`.
- **No density-guided sampling:** keep channel widths and the sensitivity-ordered data, restore full-size PointRCNN `NPOINTS`, `NSAMPLE`, and ROI pool sampling from the original configuration, and clear `SAMPLING_POLICY_PATH`. Do not reset the compact MLP/FC widths.

Create separate configuration copies and fresh output tags for each variant. These instructions define explicit reproducible controls; exact correspondence to the original ablation checkpoints requires the original server run records, which are not included.
