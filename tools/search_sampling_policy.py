import _init_path  # noqa: F401
import argparse
import math
from pathlib import Path

import numpy as np
import yaml

from offline_config import cfg, cfg_from_yaml_file


def parse_args():
    parser = argparse.ArgumentParser(description='Offline data-adaptive sampling policy search')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=None, help='KITTI root path, e.g. data/kitti')
    parser.add_argument('--split', type=str, default=None, help='train split stem in ImageSets (without .txt)')
    parser.add_argument('--num_frames', type=int, default=800, help='0 means use all frames')
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--budget_npoint_ratio', type=float, default=0.75)
    parser.add_argument('--budget_nsample_ratio', type=float, default=0.75)
    parser.add_argument('--min_npoint_ratio', type=float, default=0.6)
    parser.add_argument('--min_nsample_ratio', type=float, default=0.6)
    parser.add_argument('--min_npoint', type=int, default=32)
    parser.add_argument('--min_nsample', type=int, default=8)
    parser.add_argument('--round_to', type=int, default=8)

    parser.add_argument('--num_query', type=int, default=256, help='query points per frame for neighbor stats')
    parser.add_argument('--num_ref', type=int, default=2048, help='reference points per frame for neighbor stats')
    parser.add_argument('--output', type=str, required=True, help='output policy yaml file')
    parser.add_argument('--target_cfg', type=str, default=None,
                        help='reference pruned cfg for matching pruning magnitude')
    parser.add_argument('--match_mode', type=str, default='exact', choices=['exact', 'sum'],
                        help='exact: same NPOINTS/NSAMPLE numbers as target_cfg; '
                             'sum: keep the same total budget and reallocate by data statistics')
    return parser.parse_args()


def _round_to_multiple(x, multiple):
    if multiple <= 1:
        return int(x)
    return int(max(multiple, round(float(x) / multiple) * multiple))


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _mask_by_range(points_xyz, point_cloud_range):
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    mask = (
        (points_xyz[:, 0] >= x_min) & (points_xyz[:, 0] <= x_max) &
        (points_xyz[:, 1] >= y_min) & (points_xyz[:, 1] <= y_max) &
        (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max)
    )
    return points_xyz[mask]


def _load_split_items(split_file):
    items = []
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) == 1:
                items.append((toks[0], None))
            else:
                items.append((toks[0], toks[1]))
    return items


def _pick_items(items, num_frames, rng):
    if num_frames <= 0 or num_frames >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, num_frames).astype(np.int64)
    # de-duplicate while preserving order
    unique = []
    seen = set()
    for i in indices.tolist():
        if i not in seen:
            unique.append(items[i])
            seen.add(i)
    rng.shuffle(unique)
    return unique


def _resolve_lidar_file(kitti_root, sample_id, domain, mixed_enabled):
    training_root = kitti_root / 'training'
    if mixed_enabled:
        if domain == 'cor':
            return training_root / 'velodyne_cor' / f'{sample_id}.bin'
        if domain == 'ori':
            return training_root / 'velodyne_ori' / f'{sample_id}.bin'
        # fallback for one-column split lines
        candidate = training_root / 'velodyne_ori' / f'{sample_id}.bin'
        if candidate.exists():
            return candidate
    return training_root / 'velodyne' / f'{sample_id}.bin'


def _neighbor_count_stats(points_xyz, radii, num_query, num_ref, rng):
    n = points_xyz.shape[0]
    if n < 2:
        return [0.0 for _ in radii]

    q = min(num_query, n)
    r = min(num_ref, n)
    q_idx = rng.choice(n, size=q, replace=False)
    r_idx = rng.choice(n, size=r, replace=False)
    q_pts = points_xyz[q_idx]
    r_pts = points_xyz[r_idx]

    diff = q_pts[:, None, :] - r_pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    means = []
    for radius in radii:
        c = (dist2 <= (radius * radius)).sum(axis=1).mean()
        means.append(float(c))
    return means


def _allocate_with_sum(total, weights, lowers, uppers):
    weights = np.array(weights, dtype=np.float64)
    lowers = np.array(lowers, dtype=np.int64)
    uppers = np.array(uppers, dtype=np.int64)

    assert lowers.shape == uppers.shape == weights.shape
    total = int(total)
    min_sum = int(lowers.sum())
    max_sum = int(uppers.sum())
    total = max(min_sum, min(max_sum, total))

    free = total - min_sum
    cap = (uppers - lowers).astype(np.int64)
    out = lowers.copy()
    if free <= 0:
        return out.tolist()

    w = np.maximum(weights, 1e-9)
    w = w / w.sum()
    add = np.floor(w * free).astype(np.int64)
    add = np.minimum(add, cap)
    out += add
    rem = free - int(add.sum())

    if rem > 0:
        frac = (w * free) - add
        order = np.argsort(-frac)
        for idx in order:
            if rem <= 0:
                break
            can = int(uppers[idx] - out[idx])
            if can <= 0:
                continue
            inc = min(can, rem)
            out[idx] += inc
            rem -= inc

    return out.tolist()


def _find_processor(data_cfg, name):
    processors = data_cfg.get('DATA_PROCESSOR', [])
    for i, proc in enumerate(processors):
        if proc.get('NAME', '') == name:
            return i, proc
    return -1, None


def _to_builtin_obj(x):
    if hasattr(x, 'items'):
        return {k: _to_builtin_obj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_builtin_obj(v) for v in x]
    return x


def _search_policy_pointrcnn(args, cfg, kitti_root, split_name, items, point_counts, neigh_mean):
    model_cfg = cfg.MODEL
    data_cfg = cfg.DATA_CONFIG

    _, sample_processor = _find_processor(data_cfg, 'sample_points')
    if sample_processor is None:
        raise ValueError('PointRCNN requires a sample_points processor')
    ref_num_points = sample_processor.NUM_POINTS['train']
    density_ratio = (np.median(point_counts) / float(max(ref_num_points, 1))) ** 0.5
    density_ratio = _clip(density_ratio, args.min_npoint_ratio, 1.0)
    npoint_ratio = min(args.budget_npoint_ratio, density_ratio)

    base_npoints = list(model_cfg.BACKBONE_3D.SA_CONFIG.NPOINTS)
    new_npoints = []
    for n in base_npoints:
        nn = _round_to_multiple(max(args.min_npoint, int(round(n * npoint_ratio))), args.round_to)
        nn = min(nn, n)
        new_npoints.append(nn)

    base_nsample_layers = [list(x) for x in model_cfg.BACKBONE_3D.SA_CONFIG.NSAMPLE]
    new_nsample_layers = []
    idx = 0
    nsample_ratios = []
    for layer in base_nsample_layers:
        cur = []
        for base_ns in layer:
            mean_nb = float(neigh_mean[idx])
            redundancy = max(mean_nb / float(max(base_ns, 1)), 1.0)
            data_ratio = 1.0 / math.sqrt(redundancy)
            data_ratio = _clip(data_ratio, args.min_nsample_ratio, 1.0)
            final_ratio = min(args.budget_nsample_ratio, data_ratio)
            ns = int(round(base_ns * final_ratio))
            ns = max(args.min_nsample, min(base_ns, ns))
            cur.append(ns)
            nsample_ratios.append(final_ratio)
            idx += 1
        new_nsample_layers.append(cur)

    roi_sa_cfg = model_cfg.ROI_HEAD.SA_CONFIG
    roi_npoints = list(roi_sa_cfg.NPOINTS)
    roi_npoints_new = []
    for n in roi_npoints:
        if n == -1:
            roi_npoints_new.append(-1)
        else:
            nn = _round_to_multiple(max(args.min_npoint, int(round(n * npoint_ratio))), args.round_to)
            nn = min(nn, n)
            roi_npoints_new.append(nn)

    roi_nsample = list(roi_sa_cfg.NSAMPLE)
    nsample_ratio_global = float(np.mean(nsample_ratios)) if len(nsample_ratios) > 0 else args.budget_nsample_ratio
    roi_nsample_new = []
    for ns in roi_nsample:
        nns = int(round(ns * nsample_ratio_global))
        nns = max(args.min_nsample, min(ns, nns))
        roi_nsample_new.append(nns)

    roi_pool_num = int(model_cfg.ROI_HEAD.ROI_POINT_POOL.NUM_SAMPLED_POINTS)
    roi_pool_new = _round_to_multiple(max(args.min_npoint, int(round(roi_pool_num * npoint_ratio))), args.round_to)
    roi_pool_new = min(roi_pool_new, roi_pool_num)

    if args.target_cfg is not None:
        target_cfg = cfg_from_yaml_file(args.target_cfg, type(cfg)())
        t_model = target_cfg.MODEL
        t_backbone_npoints = list(t_model.BACKBONE_3D.SA_CONFIG.NPOINTS)
        t_backbone_nsample = [list(x) for x in t_model.BACKBONE_3D.SA_CONFIG.NSAMPLE]
        t_roi_npoints = list(t_model.ROI_HEAD.SA_CONFIG.NPOINTS)
        t_roi_nsample = list(t_model.ROI_HEAD.SA_CONFIG.NSAMPLE)
        t_roi_pool = int(t_model.ROI_HEAD.ROI_POINT_POOL.NUM_SAMPLED_POINTS)

        if args.match_mode == 'exact':
            new_npoints = t_backbone_npoints
            new_nsample_layers = t_backbone_nsample
            roi_npoints_new = t_roi_npoints
            roi_nsample_new = t_roi_nsample
            roi_pool_new = t_roi_pool
        else:
            layer_red = []
            idx2 = 0
            for layer in base_nsample_layers:
                red_list = []
                for base_ns in layer:
                    mean_nb = float(neigh_mean[idx2])
                    red_list.append(max(mean_nb / float(max(base_ns, 1)), 1.0))
                    idx2 += 1
                layer_red.append(float(np.mean(red_list)))
            layer_importance = [1.0 / math.sqrt(r) for r in layer_red]
            npoint_lowers = [args.min_npoint] * len(base_npoints)
            npoint_uppers = list(base_npoints)
            total_backbone_npoints = int(sum([x for x in t_backbone_npoints if x > 0]))
            new_npoints = _allocate_with_sum(total_backbone_npoints, layer_importance, npoint_lowers, npoint_uppers)

            for i in range(1, len(new_npoints)):
                if new_npoints[i] > new_npoints[i - 1]:
                    new_npoints[i] = new_npoints[i - 1]

            new_nsample_layers = []
            idx2 = 0
            for layer_i, layer in enumerate(base_nsample_layers):
                target_sum = int(sum(t_backbone_nsample[layer_i]))
                w, lowers, uppers = [], [], []
                for base_ns in layer:
                    mean_nb = float(neigh_mean[idx2])
                    redundancy = max(mean_nb / float(max(base_ns, 1)), 1.0)
                    w.append(1.0 / math.sqrt(redundancy))
                    lowers.append(args.min_nsample)
                    uppers.append(base_ns)
                    idx2 += 1
                alloc = _allocate_with_sum(target_sum, w, lowers, uppers)
                new_nsample_layers.append(alloc)

            roi_active_idx = [i for i, n in enumerate(roi_npoints) if n != -1]
            roi_target_total = int(sum([t_roi_npoints[i] for i in roi_active_idx]))
            roi_base_active = [roi_npoints[i] for i in roi_active_idx]
            roi_alloc = _allocate_with_sum(
                roi_target_total,
                weights=roi_base_active,
                lowers=[args.min_npoint] * len(roi_active_idx),
                uppers=roi_base_active
            )
            roi_npoints_new = list(roi_npoints)
            for k, idx3 in enumerate(roi_active_idx):
                roi_npoints_new[idx3] = roi_alloc[k]

            roi_target_nsample_total = int(sum(t_roi_nsample))
            roi_w = [1.15, 1.0, 0.85][:len(roi_nsample)]
            roi_nsample_new = _allocate_with_sum(
                roi_target_nsample_total,
                roi_w,
                [args.min_nsample] * len(roi_nsample),
                list(roi_nsample)
            )
            roi_pool_new = t_roi_pool

    out_cfg = {
        'META': {
            'policy_type': 'offline_data_adaptive_sampling',
            'model_name': 'PointRCNN',
            'source_cfg': args.cfg_file,
            'data_path': str(kitti_root),
            'split': split_name,
            'used_frames': int(len(items)),
            'point_count_median': float(np.median(point_counts)),
            'point_count_mean': float(np.mean(point_counts)),
            'npoint_ratio': float(npoint_ratio),
            'nsample_ratio_mean': float(nsample_ratio_global),
            'target_cfg': args.target_cfg if args.target_cfg is not None else '',
            'match_mode': args.match_mode if args.target_cfg is not None else '',
        },
        'MODEL': {
            'BACKBONE_3D': {
                'SA_CONFIG': {
                    'NPOINTS': new_npoints,
                    'NSAMPLE': new_nsample_layers
                }
            },
            'ROI_HEAD': {
                'ROI_POINT_POOL': {
                    'NUM_SAMPLED_POINTS': roi_pool_new
                },
                'SA_CONFIG': {
                    'NPOINTS': roi_npoints_new,
                    'NSAMPLE': roi_nsample_new
                }
            }
        }
    }

    summary = [
        f'Backbone NPOINTS: {base_npoints} -> {new_npoints}',
        f'Backbone NSAMPLE: {base_nsample_layers} -> {new_nsample_layers}',
        f'ROI NPOINTS: {roi_npoints} -> {roi_npoints_new}',
        f'ROI NSAMPLE: {roi_nsample} -> {roi_nsample_new}',
        f'ROI NUM_SAMPLED_POINTS: {roi_pool_num} -> {roi_pool_new}',
    ]
    return out_cfg, summary


def _search_policy_second(args, cfg, kitti_root, split_name, items, point_counts):
    data_cfg = cfg.DATA_CONFIG
    model_cfg = cfg.MODEL
    median_points = float(np.median(point_counts))

    idx_sp, sp_cfg = _find_processor(data_cfg, 'sample_points')
    idx_vox, vox_cfg = _find_processor(data_cfg, 'transform_points_to_voxels')
    if idx_vox < 0:
        raise RuntimeError('SECOND policy search requires DATA_PROCESSOR.transform_points_to_voxels')

    base_sample_train = None
    base_sample_test = None
    if idx_sp >= 0:
        base_sample_train = int(sp_cfg.NUM_POINTS['train'])
        base_sample_test = int(sp_cfg.NUM_POINTS['test'])

    base_vox_train = int(vox_cfg.MAX_NUMBER_OF_VOXELS['train'])
    base_vox_test = int(vox_cfg.MAX_NUMBER_OF_VOXELS['test'])

    ref_num_points = base_sample_train if base_sample_train is not None else max(int(median_points), 1)
    density_ratio = (median_points / float(max(ref_num_points, 1))) ** 0.5
    density_ratio = _clip(density_ratio, args.min_npoint_ratio, 1.0)
    budget_ratio = min(args.budget_npoint_ratio, density_ratio)

    new_sample_train = base_sample_train
    new_sample_test = base_sample_test
    if base_sample_train is not None:
        new_sample_train = max(args.min_npoint, _round_to_multiple(int(round(base_sample_train * budget_ratio)), args.round_to))
        new_sample_train = min(new_sample_train, base_sample_train)
        new_sample_test = max(args.min_npoint, _round_to_multiple(int(round(base_sample_test * budget_ratio)), args.round_to))
        new_sample_test = min(new_sample_test, base_sample_test)

    new_vox_train = max(1000, _round_to_multiple(int(round(base_vox_train * budget_ratio)), args.round_to))
    new_vox_train = min(new_vox_train, base_vox_train)
    new_vox_test = max(1000, _round_to_multiple(int(round(base_vox_test * budget_ratio)), args.round_to))
    new_vox_test = min(new_vox_test, base_vox_test)

    base_num_filters = [int(x) for x in model_cfg.BACKBONE_2D.NUM_FILTERS]
    base_num_up_filters = [int(x) for x in model_cfg.BACKBONE_2D.NUM_UPSAMPLE_FILTERS]
    new_num_filters = list(base_num_filters)
    new_num_up_filters = list(base_num_up_filters)

    if args.target_cfg is not None:
        target_cfg = cfg_from_yaml_file(args.target_cfg, type(cfg)())
        t_data = target_cfg.DATA_CONFIG
        t_model = target_cfg.MODEL
        t_idx_sp, t_sp = _find_processor(t_data, 'sample_points')
        t_idx_vox, t_vox = _find_processor(t_data, 'transform_points_to_voxels')
        if t_idx_vox < 0:
            raise RuntimeError('target_cfg for SECOND must include transform_points_to_voxels')

        t_vox_train = int(t_vox.MAX_NUMBER_OF_VOXELS['train'])
        t_vox_test = int(t_vox.MAX_NUMBER_OF_VOXELS['test'])
        t_sample_train = int(t_sp.NUM_POINTS['train']) if t_idx_sp >= 0 and idx_sp >= 0 else None
        t_sample_test = int(t_sp.NUM_POINTS['test']) if t_idx_sp >= 0 and idx_sp >= 0 else None
        t_num_filters = [int(x) for x in t_model.BACKBONE_2D.NUM_FILTERS]
        t_num_up_filters = [int(x) for x in t_model.BACKBONE_2D.NUM_UPSAMPLE_FILTERS]

        if len(t_num_filters) != len(base_num_filters) or len(t_num_up_filters) != len(base_num_up_filters):
            raise RuntimeError('SECOND target_cfg BACKBONE_2D filter lengths must match baseline cfg')

        if args.match_mode == 'exact':
            new_vox_train, new_vox_test = t_vox_train, t_vox_test
            if t_sample_train is not None:
                new_sample_train, new_sample_test = t_sample_train, t_sample_test
            new_num_filters = t_num_filters
            new_num_up_filters = t_num_up_filters
        else:
            # keep total budget, reallocate train/test by complexity preference
            # train split is optimization target, slightly favor train budget.
            w_train = 1.2 + (median_points / float(max(ref_num_points, 1)))
            w_test = 1.0
            vox_alloc = _allocate_with_sum(
                t_vox_train + t_vox_test,
                [w_train, w_test],
                [1000, 1000],
                [base_vox_train, base_vox_test]
            )
            new_vox_train, new_vox_test = int(vox_alloc[0]), int(vox_alloc[1])

            if t_sample_train is not None:
                sp_alloc = _allocate_with_sum(
                    t_sample_train + t_sample_test,
                    [w_train, w_test],
                    [args.min_npoint, args.min_npoint],
                    [base_sample_train, base_sample_test]
                )
                new_sample_train, new_sample_test = int(sp_alloc[0]), int(sp_alloc[1])

            # keep total structural budget and redistribute by data complexity.
            # shallower stage usually dominates detail sensitivity; give it slightly larger weight.
            stage_w = [1.1 + 0.15 * (len(base_num_filters) - 1 - i) for i in range(len(base_num_filters))]
            up_w = [1.0 for _ in range(len(base_num_up_filters))]
            all_w = stage_w + up_w
            lowers = [args.round_to] * (len(base_num_filters) + len(base_num_up_filters))
            uppers = base_num_filters + base_num_up_filters
            target_total = int(sum(t_num_filters) + sum(t_num_up_filters))
            alloc = _allocate_with_sum(target_total, all_w, lowers, uppers)
            alloc = [_round_to_multiple(v, args.round_to) for v in alloc]
            alloc = [max(args.round_to, min(v, uppers[i])) for i, v in enumerate(alloc)]
            # one more pass to correct rounded-sum drift
            drift = target_total - int(sum(alloc))
            if drift != 0:
                step = args.round_to
                order = list(range(len(alloc)))
                if drift < 0:
                    order = order[::-1]
                while drift != 0:
                    updated = False
                    for i in order:
                        if drift > 0 and alloc[i] + step <= uppers[i]:
                            alloc[i] += step
                            drift -= step
                            updated = True
                        elif drift < 0 and alloc[i] - step >= args.round_to:
                            alloc[i] -= step
                            drift += step
                            updated = True
                        if drift == 0:
                            break
                    if not updated:
                        break
            new_num_filters = alloc[:len(base_num_filters)]
            new_num_up_filters = alloc[len(base_num_filters):]

    data_override = {'DATA_PROCESSOR': []}
    for i, proc in enumerate(data_cfg.DATA_PROCESSOR):
        proc_dict = _to_builtin_obj(proc)
        if i == idx_sp and base_sample_train is not None:
            proc_dict['NUM_POINTS']['train'] = int(new_sample_train)
            proc_dict['NUM_POINTS']['test'] = int(new_sample_test)
        if i == idx_vox:
            proc_dict['MAX_NUMBER_OF_VOXELS']['train'] = int(new_vox_train)
            proc_dict['MAX_NUMBER_OF_VOXELS']['test'] = int(new_vox_test)
        data_override['DATA_PROCESSOR'].append(proc_dict)

    model_override = {
        'BACKBONE_2D': {
            'NUM_FILTERS': [int(x) for x in new_num_filters],
            'NUM_UPSAMPLE_FILTERS': [int(x) for x in new_num_up_filters]
        }
    }

    out_cfg = {
        'META': {
            'policy_type': 'offline_data_adaptive_sampling',
            'model_name': str(cfg.MODEL.NAME),
            'source_cfg': args.cfg_file,
            'data_path': str(kitti_root),
            'split': split_name,
            'used_frames': int(len(items)),
            'point_count_median': float(np.median(point_counts)),
            'point_count_mean': float(np.mean(point_counts)),
            'budget_ratio': float(budget_ratio),
            'target_cfg': args.target_cfg if args.target_cfg is not None else '',
            'match_mode': args.match_mode if args.target_cfg is not None else '',
        },
        'MODEL': model_override,
        'DATA_CONFIG': data_override
    }

    summary = []
    summary.append(f'BACKBONE_2D.NUM_FILTERS: {base_num_filters} -> {new_num_filters}')
    summary.append(f'BACKBONE_2D.NUM_UPSAMPLE_FILTERS: {base_num_up_filters} -> {new_num_up_filters}')
    if base_sample_train is not None:
        summary.append(f'sample_points.NUM_POINTS train/test: [{base_sample_train}, {base_sample_test}] -> [{new_sample_train}, {new_sample_test}]')
    summary.append(f'transform_points_to_voxels.MAX_NUMBER_OF_VOXELS train/test: [{base_vox_train}, {base_vox_test}] -> [{new_vox_train}, {new_vox_test}]')
    return out_cfg, summary


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cfg_from_yaml_file(args.cfg_file, cfg)
    model_cfg = cfg.MODEL
    data_cfg = cfg.DATA_CONFIG

    kitti_root = Path(args.data_path) if args.data_path is not None else (cfg.ROOT_DIR / data_cfg.DATA_PATH)
    split_name = args.split if args.split is not None else data_cfg.DATA_SPLIT['train']
    split_file = kitti_root / 'ImageSets' / f'{split_name}.txt'
    if not split_file.exists():
        raise FileNotFoundError(f'Split file not found: {split_file}')

    items = _load_split_items(split_file)
    if len(items) == 0:
        raise RuntimeError(f'No frame in split: {split_file}')
    items = _pick_items(items, args.num_frames, rng)

    point_cloud_range = data_cfg.POINT_CLOUD_RANGE
    mixed_enabled = bool(data_cfg.get('MIXED_DATASET', {}).get('ENABLED', False))

    model_name = str(model_cfg.NAME)
    flat_radii = []
    if model_name == 'PointRCNN':
        backbone_radius = model_cfg.BACKBONE_3D.SA_CONFIG.RADIUS
        for layer_r in backbone_radius:
            flat_radii.extend(layer_r)
    else:
        # SECOND and other voxel models: use coarse radii for local density proxy.
        flat_radii = [0.2, 0.4, 0.8]

    point_counts = []
    neigh_sum = np.zeros(len(flat_radii), dtype=np.float64)
    used_frames = 0

    for sid, domain in items:
        lidar_file = _resolve_lidar_file(kitti_root, sid, domain, mixed_enabled)
        if not lidar_file.exists():
            raise FileNotFoundError(lidar_file)

        pts = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)[:, :3]
        pts = _mask_by_range(pts, point_cloud_range)
        if pts.shape[0] < 4:
            continue

        point_counts.append(pts.shape[0])
        neigh_stats = _neighbor_count_stats(pts, flat_radii, args.num_query, args.num_ref, rng)
        neigh_sum += np.array(neigh_stats, dtype=np.float64)
        used_frames += 1

    if used_frames == 0:
        raise RuntimeError('No valid frame is found for policy search.')

    point_counts = np.array(point_counts, dtype=np.float64)
    neigh_mean = neigh_sum / float(used_frames)

    if model_name == 'PointRCNN':
        out_cfg, summary = _search_policy_pointrcnn(
            args=args, cfg=cfg, kitti_root=kitti_root, split_name=split_name,
            items=items, point_counts=point_counts, neigh_mean=neigh_mean
        )
    elif model_name.startswith('SECOND'):
        out_cfg, summary = _search_policy_second(
            args=args, cfg=cfg, kitti_root=kitti_root, split_name=split_name,
            items=items, point_counts=point_counts
        )
    else:
        raise ValueError(f'Unsupported model for policy search: {model_name}')

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)

    print('=== Offline Sampling Policy Search Summary ===')
    print(f'Frames used: {used_frames}')
    print(f'Median points in range: {np.median(point_counts):.1f}')
    for line in summary:
        print(line)
    print(f'Policy saved to: {out_file}')


if __name__ == '__main__':
    main()
