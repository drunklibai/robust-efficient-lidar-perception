# Source release scope

This repository is a curated source release of the research workspace. It includes the detector framework, main technical route, baseline tools, and reproducibility documentation. All dataset contents, split files, checkpoints, prediction caches, logs, paper drafts, internal analysis notes, editor files, safety-pruning experiments and backup versions are excluded. `data/.gitkeep` is the only committed file under `data/`.

## Preparation changes

- Added `paper_pipeline.py` and `run_references.py` for explicit frame identities, full 1:1 mixtures, corruption-cache assembly, reference inference, fusion and ranking.
- Added standalone root-relative KITTI/ONCE recipe configurations and policy generation commands.
- Removed the inactive KD branches from the PointRCNN head, KITTI dataset and collate path; ordinary detection losses and model parameters are retained.
- Honored `SHUFFLE_TRAIN` in the distributed sampler and disabled frame shuffling in the Ours recipes.
- Made three-class ONCE R40 evaluation explicit in the paper recipes.
- Matched the manuscript's empty/empty sensitivity boundary: unmatched fraction is zero. The release ranking uses descending-IoU greedy matching; older KITTI helpers used row-order matching.
- Prevented a silent rotated-IoU to axis-aligned-IoU fallback in the fusion helper.
- Allowed KITTI metadata generation before info files exist and corrected an existing invalid relative import in an auxiliary KITTI evaluator.
- Made policy generation fail on missing selected KITTI point files instead of silently skipping them.

## Evidence boundary

The original experiments were conducted on a separate server. The release does not contain recovered server policies, final checkpoints, simulator adapters, reference bundles or random assignments. Defaults in the new commands are explicit regeneration choices, not claims about unknown original settings. In particular, the ONCE-specific external weather adaptation and the complete seven-model ONCE checkpoint/configuration bundle remain reader-provided dependencies.

Validation during release preparation: 14 CPU tests passed, including real sampling-policy generation for KITTI/ONCE and PointRCNN/SECOND on synthetic point files, internal module dependency checks, and presence of every declared CUDA extension source. Python source compilation passed. These synthetic tests exercise preparation and configuration behavior, not detection accuracy. CUDA extension compilation, seven-model GPU inference, training convergence, checkpoint compatibility and reproduction of the manuscript tables require the actual Linux GPU/data environment and have not been certified by those CPU tests.

Do not treat generated configurations as verified original-run artifacts. Preserve the actual configuration, policy, split, seed and checkpoints for every experiment. Original third-party license and copyright notices are retained.
