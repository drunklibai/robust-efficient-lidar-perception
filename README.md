# Robust and Efficient LiDAR Perception

Code for **Provisioning Robust and Efficient LiDAR Perception for Autonomous Driving**.

This project constructs a fixed detector offline to improve robustness to point-cloud corruption while reducing inference latency and GPU memory use. It is built on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Technical route

1. Prepare KITTI or ONCE and cache one corrupted counterpart per clean frame.
2. Run seven fixed reference detectors on both domains and fuse their predictions offline.
3. Score clean/corrupted prediction differences and order the complete 1:1 training mixture from low to high perturbation sensitivity.
4. Reduce channel widths and generate point/neighbor budgets (PointRCNN) or voxel budgets (SECOND) from training-data statistics.
5. Train the compact detector from scratch, preserving the training order.
6. Evaluate clean/corrupted AP, inference latency, p95 latency, and peak allocated GPU memory.

Reference detectors are used during offline preparation only. Deployment runs a single detector. The released main route does not use knowledge distillation or safety-weighted channel pruning.

## Start here

- [Installation](docs/INSTALL.md)
- [Data and corruption preparation](docs/DATA_PREPARATION.md)
- [Complete reproduction commands](docs/REPRODUCING.md)
- [Baselines and ablations](docs/BASELINES.md)
- [Release scope and validation](docs/RELEASE_NOTES.md)

Run the documented commands **from the repository root**. Training and inference require a Linux CUDA environment with compiled OpenPCDet operators.

```bash
git clone https://github.com/drunklibai/robust-efficient-lidar-perception.git
cd robust-efficient-lidar-perception
python tools/paper_pipeline.py --help
```

`data/` contains only a `.gitkeep` placeholder. Download datasets and upstream splits yourself; all metadata, corrupted point clouds, prediction caches, sorted lists, and sampling policies are generated locally. See the data guide before running training.

## Repository layout

```text
data/                         # empty placeholder; all local contents ignored
pcdet/                        # detector framework and CUDA/C++ operators
tools/cfgs/paper/              # explicit KITTI/ONCE reproduction recipes
tools/paper_pipeline.py        # prepare, assemble-corruptions, predict, fuse, rank
tools/run_references.py        # seven-reference offline preparation
tools/search_sampling_policy*.py
tools/train.py
tools/test.py
tools/test_speed.py
docs/
tests/                        # CPU preparation and protocol checks
```

## Models

Obtain reference checkpoints from the [OpenPCDet Model Zoo](https://github.com/open-mmlab/OpenPCDet#model-zoo) or train the reference detectors on the target dataset. A checkpoint must match its architecture, class ordering, input features, and dataset configuration. No paper-specific checkpoint download is included in this release.

The reference list used in the manuscript is PointRCNN, PointRCNN-IoU, PartA2, PointPillars, SECOND, SECOND-IoU, and PV-RCNN. The orchestration script accepts dataset-specific configuration/checkpoint pairs for these references.

## Acknowledgements and license

We acknowledge OpenPCDet and the detector implementations it includes, KITTI, ONCE, and the [point-cloud corruption toolkit](https://github.com/Castiel-Lee/robustness_pc_detector). Keep the existing [Apache-2.0 license](LICENSE) and individual third-party notices when using this code. External datasets, checkpoints, and corruption tools retain their own terms.

Please cite the paper by its title above and acknowledge OpenPCDet when using this implementation. Publication metadata will be added when available.
