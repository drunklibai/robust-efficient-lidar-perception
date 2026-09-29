# Data and corruption preparation

## 1. Download and prepare clean data

No dataset content or split file is committed. Obtain KITTI/ONCE from their providers and use the [OpenPCDet data preparation instructions](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md). Download the corresponding upstream `ImageSets` files into your local dataset directory; do not invent a new train/validation partition.

Expected KITTI layout after clean metadata generation:

```text
data/kitti/
  ImageSets/train.txt, val.txt, test.txt
  training/velodyne/          # original float32 [x,y,z,intensity]
  training/image_2/, calib/, label_2/
  testing/                   # if included in the selected metadata procedure
  kitti_infos_train.pkl, kitti_infos_val.pkl
  kitti_dbinfos_train.pkl, gt_database/
```

Generate metadata with the base dataset configuration, before enabling mixed data:

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

The paper recipes read clean points from `training/velodyne_ori`. Preserve `velodyne/` for standard preparation and create a local alias:

```bash
ln -s velodyne data/kitti/training/velodyne_ori
```

For ONCE, follow upstream instructions to generate `once_infos_train.pkl`, `once_infos_val.pkl` and the ground-truth database. Expected point paths are:

```text
data/once/ImageSets/train.txt, val.txt, test.txt
data/once/data/<sequence_id>/lidar_roof/<frame_id>.bin
data/once/once_infos_train.pkl
data/once/once_infos_val.pkl
data/once/once_dbinfos_train.pkl
```

Use the original ONCE sequence-level split files. Metadata must include annotations for the labelled training and validation frames. The pipeline keeps their frame order explicit.

## 2. Generate six severity-4 corruption caches

Use the externally maintained [robustness_pc_detector toolkit](https://github.com/Castiel-Lee/robustness_pc_detector) referenced by the paper. It supplies scene-level density/noise corruptions and weather simulation based on LISA. Follow its installation and per-directory instructions:

- [Density](https://github.com/Castiel-Lee/robustness_pc_detector/tree/main/scene/density): `density_dec`, `density_inc`.
- [Noise](https://github.com/Castiel-Lee/robustness_pc_detector/tree/main/scene/noise): Gaussian and background noise.
- [Weather](https://github.com/Castiel-Lee/robustness_pc_detector/tree/main/scene): rain and snow via its LISA integration.

For example, the upstream density command (run in its density directory) is:

```bash
python Simulation_density.py --path_data /absolute/path/to/clean/velodyne \
  --save_dir /absolute/path/to/cache --noise_model density_dec -r 4
```

Generate each of the six corruption types at **severity 4 of 5**. Use the toolkit's documented noise/weather commands and actual output directories rather than assuming all scripts have identical flags. Preserve original annotations. Record the external toolkit commit, simulator settings and random seeds alongside your generated artifacts.

The toolkit documents KITTI input; for ONCE, process each sequence's `lidar_roof` directory and preserve sequence IDs and the four-channel binary format. Weather simulator outputs with auxiliary label columns must be converted to float32 x/y/z/intensity before caching. Dataset-specific simulator adaptation is external to this repository; the original server adapter is not included.

## 3. Build the fixed mixed-corruption dataset

Create a local JSON mapping the six names to the directories containing your cached severity-4 point clouds:

```json
{
  "density_dec": "/path/to/cache/density_dec_4",
  "density_inc": "/path/to/cache/density_inc_4",
  "gaussian": "/path/to/cache/gaussian_4",
  "background": "/path/to/cache/background_4",
  "rain": "/path/to/cache/rain_4",
  "snow": "/path/to/cache/snow_4"
}
```

Each KITTI source directory must contain `<frame_id>.bin`. Each ONCE source directory must contain `<sequence_id>/lidar_roof/<frame_id>.bin`. Keep train and validation frame identities disjoint.

```bash
python tools/paper_pipeline.py assemble-corruptions --dataset kitti --data_root data/kitti \
  --sources /path/to/corruption_sources.json --split train --seed 1234
python tools/paper_pipeline.py assemble-corruptions --dataset kitti --data_root data/kitti \
  --sources /path/to/corruption_sources.json --split val --seed 5678
```

For ONCE, replace `kitti` with `once` and provide the corresponding source directories. The script selects one cached corruption per frame, writes `training/velodyne_cor/` (KITTI) or `data_cor/` (ONCE), and saves the assignment, severity and file checksums. It refuses to overwrite existing corrupted frames. It assembles external simulator outputs; it does not substitute an approximate weather generator.

The balanced seeded assignment is an explicit release recipe, not a recovered assignment from the original server. If you already have the original fixed corruption dataset, use it directly and preserve its provenance instead of reassembling it.

## 4. Generate local split lists

```bash
python tools/paper_pipeline.py prepare --dataset kitti --data_root data/kitti
python tools/paper_pipeline.py prepare --dataset once --data_root data/once
```

This creates:

- `train_mixed_reference_ori.txt` / `train_mixed_reference_cor.txt`: same training frames, separately on the two domains.
- `train_mixed_nosel.txt`: complete 1:1 mixture for Retrain.
- `val_ori.txt` / `val_cor.txt`: clean/corrupted validation splits.
- `paper_train_frames.json` / `paper_val_frames.json`: explicit identities and order.

Original `train.txt` and `val.txt` remain unchanged. No generated file needs to be uploaded to Git. Before moving to reference prediction, ensure all required point files and metadata are present.
