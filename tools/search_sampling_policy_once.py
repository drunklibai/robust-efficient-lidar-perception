import _init_path  # noqa: F401
import argparse
import math
from pathlib import Path

import numpy as np
import yaml

from offline_config import cfg, cfg_from_yaml_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Offline data-adaptive sampling policy search for ONCE'
    )
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=None, help='ONCE root path, e.g. ../data/once')
    parser.add_argument('--split', type=str, default=None, help='Frame-level split stem in ImageSets')
    parser.add_argument('--num_frames', type=int, default=800, help='0 means use all split entries')
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--budget_npoint_ratio', type=float, default=0.75)
    parser.add_argument('--budget_nsample_ratio', type=float, default=0.75)
    parser.add_argument('--min_npoint_ratio', type=float, default=0.6)
    parser.add_argument('--min_nsample_ratio', type=float, default=0.6)
    parser.add_argument('--min_npoint', type=int, default=32)
    parser.add_argument('--min_nsample', type=int, default=8)
    parser.add_argument('--round_to', type=int, default=8)

    parser.add_argument('--num_query', type=int, default=256, help='Query points per frame for neighbor stats')
    parser.add_argument('--num_ref', type=int, default=2048, help='Reference points per frame for neighbor stats')
    parser.add_argument('--output', type=str, required=True, help='Output policy yaml file')
    parser.add_argument('--target_cfg', type=str, default=None, help='Reference hand-pruned cfg for budget matching')
    parser.add_argument(
        '--match_mode', type=str, default='exact', choices=['exact', 'sum', 'groupwise_sum'],
        help='exact: use the target sampling values; sum: preserve target total sampling budget '
             'and reallocate it using ONCE mixed-data statistics; '
             'groupwise_sum: for SECOND, preserve separate backbone and upsample channel budgets'
    )
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


def _load_once_split_items(split_file):
    items = []
    with open(split_file, 'r') as f:
        for line_no, line in enumerate(f, start=1):
            toks = line.strip().split()
            if not toks:
                continue
            if len(toks) == 2:
                seq_id, frame_id = toks
                domain = 'ori'
            elif len(toks) >= 3:
                seq_id, frame_id, domain = toks[:3]
            else:
                raise ValueError(
                    f'ONCE policy search requires frame-level split entries, but line {line_no} '
                    f'in {split_file} contains only a sequence id: {line.strip()}'
                )
            if domain not in ['ori', 'cor']:
                raise ValueError(
                    f'Unsupported ONCE domain at line {line_no} in {split_file}: {domain}'
                )
            items.append((seq_id, frame_id, domain))
    return items


def _pick_items(items, num_frames, rng):
    if num_frames <= 0 or num_frames >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, num_frames).astype(np.int64)
    unique = []
    seen = set()
    for idx in indices.tolist():
        if idx not in seen:
            unique.append(items[idx])
            seen.add(idx)
    rng.shuffle(unique)
    return unique


def _resolve_once_lidar_file(once_root, seq_id, frame_id, domain):
    lidar_root = 'data_cor' if domain == 'cor' else 'data'
    return once_root / lidar_root / seq_id / 'lidar_roof' / f'{frame_id}.bin'


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
        means.append(float((dist2 <= radius * radius).sum(axis=1).mean()))
    return means


def _allocate_with_sum(total, weights, lowers, uppers):
    weights = np.array(weights, dtype=np.float64)
    lowers = np.array(lowers, dtype=np.int64)
    uppers = np.array(uppers, dtype=np.int64)

    assert lowers.shape == uppers.shape == weights.shape
    total = int(max(int(lowers.sum()), min(int(uppers.sum()), int(total))))

    free = total - int(lowers.sum())
    caps = uppers - lowers
    out = lowers.copy()
    if free <= 0:
        return out.tolist()

    weights = np.maximum(weights, 1e-9)
    weights = weights / weights.sum()
    raw_add = weights * free
    add = np.minimum(np.floor(raw_add).astype(np.int64), caps)
    out += add
    remaining = free - int(add.sum())

    if remaining > 0:
        order = np.argsort(-(raw_add - add))
        for idx in order:
            if remaining <= 0:
                break
            can_add = int(uppers[idx] - out[idx])
            if can_add <= 0:
                continue
            delta = min(can_add, remaining)
            out[idx] += delta
            remaining -= delta

    return out.tolist()


def _allocate_rounded_with_sum(total, weights, lowers, uppers, round_to):
    """Allocate a channel budget in round_to-sized units without total drift."""
    if round_to <= 1:
        return _allocate_with_sum(total, weights, lowers, uppers)
    if total % round_to != 0:
        raise ValueError(f'Channel budget {total} must be divisible by round_to={round_to}')

    lower_units = [int(math.ceil(float(value) / round_to)) for value in lowers]
    upper_units = [int(math.floor(float(value) / round_to)) for value in uppers]
    if any(lo > hi for lo, hi in zip(lower_units, upper_units)):
        raise ValueError('Rounded lower channel bound exceeds its upper bound')

    allocated_units = _allocate_with_sum(
        total // round_to, weights, lower_units, upper_units
    )
    return [int(value * round_to) for value in allocated_units]


def _find_processor(data_cfg, name):
    for proc in data_cfg.get('DATA_PROCESSOR', []):
        if proc.get('NAME', '') == name:
            return proc
    return None


def _find_processor_with_index(data_cfg, name):
    processors = data_cfg.get('DATA_PROCESSOR', [])
    for idx, proc in enumerate(processors):
        if proc.get('NAME', '') == name:
            return idx, proc
    return -1, None


def _to_builtin_obj(obj):
    if hasattr(obj, 'items'):
        return {k: _to_builtin_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin_obj(v) for v in obj]
    return obj


def _sampling_budget(model_cfg):
    backbone_npoints = list(model_cfg.BACKBONE_3D.SA_CONFIG.NPOINTS)
    backbone_nsample = [list(x) for x in model_cfg.BACKBONE_3D.SA_CONFIG.NSAMPLE]
    roi_npoints = list(model_cfg.ROI_HEAD.SA_CONFIG.NPOINTS)
    roi_nsample = list(model_cfg.ROI_HEAD.SA_CONFIG.NSAMPLE)
    roi_pool = int(model_cfg.ROI_HEAD.ROI_POINT_POOL.NUM_SAMPLED_POINTS)
    return {
        'backbone_npoints': int(sum(x for x in backbone_npoints if x > 0)),
        'backbone_nsample': [int(sum(x)) for x in backbone_nsample],
        'roi_npoints': int(sum(x for x in roi_npoints if x > 0)),
        'roi_nsample': int(sum(roi_nsample)),
        'roi_pool': roi_pool,
    }


def _search_policy_pointrcnn(args, config, once_root, split_name, used_frames, point_counts, neigh_mean):
    model_cfg = config.MODEL
    data_cfg = config.DATA_CONFIG

    sample_points_cfg = _find_processor(data_cfg, 'sample_points')
    if sample_points_cfg is None:
        raise RuntimeError('PointRCNN policy search requires DATA_PROCESSOR.sample_points in cfg_file')
    ref_num_points = int(sample_points_cfg.NUM_POINTS['train'])
    density_ratio = (np.median(point_counts) / float(max(ref_num_points, 1))) ** 0.5
    density_ratio = _clip(density_ratio, args.min_npoint_ratio, 1.0)
    npoint_ratio = min(args.budget_npoint_ratio, density_ratio)

    base_npoints = list(model_cfg.BACKBONE_3D.SA_CONFIG.NPOINTS)
    new_npoints = []
    for n in base_npoints:
        val = _round_to_multiple(max(args.min_npoint, int(round(n * npoint_ratio))), args.round_to)
        new_npoints.append(min(val, n))

    base_nsample_layers = [list(x) for x in model_cfg.BACKBONE_3D.SA_CONFIG.NSAMPLE]
    new_nsample_layers = []
    nsample_ratios = []
    flat_idx = 0
    for layer in base_nsample_layers:
        cur = []
        for base_ns in layer:
            redundancy = max(float(neigh_mean[flat_idx]) / float(max(base_ns, 1)), 1.0)
            data_ratio = _clip(1.0 / math.sqrt(redundancy), args.min_nsample_ratio, 1.0)
            final_ratio = min(args.budget_nsample_ratio, data_ratio)
            val = max(args.min_nsample, min(base_ns, int(round(base_ns * final_ratio))))
            cur.append(val)
            nsample_ratios.append(final_ratio)
            flat_idx += 1
        new_nsample_layers.append(cur)

    roi_npoints = list(model_cfg.ROI_HEAD.SA_CONFIG.NPOINTS)
    roi_npoints_new = []
    for n in roi_npoints:
        if n == -1:
            roi_npoints_new.append(-1)
            continue
        val = _round_to_multiple(max(args.min_npoint, int(round(n * npoint_ratio))), args.round_to)
        roi_npoints_new.append(min(val, n))

    roi_nsample = list(model_cfg.ROI_HEAD.SA_CONFIG.NSAMPLE)
    nsample_ratio_global = float(np.mean(nsample_ratios)) if nsample_ratios else args.budget_nsample_ratio
    roi_nsample_new = [
        max(args.min_nsample, min(ns, int(round(ns * nsample_ratio_global))))
        for ns in roi_nsample
    ]

    roi_pool_num = int(model_cfg.ROI_HEAD.ROI_POINT_POOL.NUM_SAMPLED_POINTS)
    roi_pool_new = _round_to_multiple(
        max(args.min_npoint, int(round(roi_pool_num * npoint_ratio))), args.round_to
    )
    roi_pool_new = min(roi_pool_new, roi_pool_num)
    target_budget = None

    if args.target_cfg is not None:
        target_cfg = cfg_from_yaml_file(args.target_cfg, type(config)())
        target_model = target_cfg.MODEL
        target_npoints = list(target_model.BACKBONE_3D.SA_CONFIG.NPOINTS)
        target_nsample = [list(x) for x in target_model.BACKBONE_3D.SA_CONFIG.NSAMPLE]
        target_roi_npoints = list(target_model.ROI_HEAD.SA_CONFIG.NPOINTS)
        target_roi_nsample = list(target_model.ROI_HEAD.SA_CONFIG.NSAMPLE)
        target_roi_pool = int(target_model.ROI_HEAD.ROI_POINT_POOL.NUM_SAMPLED_POINTS)
        target_budget = _sampling_budget(target_model)

        if args.match_mode == 'exact':
            new_npoints = target_npoints
            new_nsample_layers = target_nsample
            roi_npoints_new = target_roi_npoints
            roi_nsample_new = target_roi_nsample
            roi_pool_new = target_roi_pool
        else:
            layer_redundancy = []
            flat_idx = 0
            for layer in base_nsample_layers:
                cur = []
                for base_ns in layer:
                    cur.append(max(float(neigh_mean[flat_idx]) / float(max(base_ns, 1)), 1.0))
                    flat_idx += 1
                layer_redundancy.append(float(np.mean(cur)))
            layer_importance = [1.0 / math.sqrt(x) for x in layer_redundancy]
            new_npoints = _allocate_with_sum(
                sum(x for x in target_npoints if x > 0),
                layer_importance,
                [args.min_npoint] * len(base_npoints),
                base_npoints,
            )
            for idx in range(1, len(new_npoints)):
                if new_npoints[idx] > new_npoints[idx - 1]:
                    new_npoints[idx] = new_npoints[idx - 1]

            new_nsample_layers = []
            flat_idx = 0
            for layer_idx, layer in enumerate(base_nsample_layers):
                weights = []
                for base_ns in layer:
                    redundancy = max(float(neigh_mean[flat_idx]) / float(max(base_ns, 1)), 1.0)
                    weights.append(1.0 / math.sqrt(redundancy))
                    flat_idx += 1
                new_nsample_layers.append(_allocate_with_sum(
                    sum(target_nsample[layer_idx]),
                    weights,
                    [args.min_nsample] * len(layer),
                    layer,
                ))

            active_idx = [idx for idx, n in enumerate(roi_npoints) if n != -1]
            allocated = _allocate_with_sum(
                sum(target_roi_npoints[idx] for idx in active_idx),
                [roi_npoints[idx] for idx in active_idx],
                [args.min_npoint] * len(active_idx),
                [roi_npoints[idx] for idx in active_idx],
            )
            roi_npoints_new = list(roi_npoints)
            for out_idx, roi_idx in enumerate(active_idx):
                roi_npoints_new[roi_idx] = allocated[out_idx]
            roi_nsample_new = _allocate_with_sum(
                sum(target_roi_nsample),
                [1.15, 1.0, 0.85][:len(roi_nsample)],
                [args.min_nsample] * len(roi_nsample),
                roi_nsample,
            )
            roi_pool_new = target_roi_pool

    policy_model = {
        'BACKBONE_3D': {'SA_CONFIG': {'NPOINTS': new_npoints, 'NSAMPLE': new_nsample_layers}},
        'ROI_HEAD': {
            'ROI_POINT_POOL': {'NUM_SAMPLED_POINTS': roi_pool_new},
            'SA_CONFIG': {'NPOINTS': roi_npoints_new, 'NSAMPLE': roi_nsample_new},
        },
    }
    generated_budget = _sampling_budget(type(config)({
        'BACKBONE_3D': type(config)({'SA_CONFIG': type(config)({
            'NPOINTS': new_npoints, 'NSAMPLE': new_nsample_layers
        })}),
        'ROI_HEAD': type(config)({
            'ROI_POINT_POOL': type(config)({'NUM_SAMPLED_POINTS': roi_pool_new}),
            'SA_CONFIG': type(config)({'NPOINTS': roi_npoints_new, 'NSAMPLE': roi_nsample_new}),
        }),
    }))

    out_cfg = {
        'META': {
            'policy_type': 'offline_data_adaptive_sampling',
            'dataset': 'ONCE',
            'model_name': 'PointRCNN',
            'source_cfg': args.cfg_file,
            'data_path': str(once_root),
            'split': split_name,
            'used_frames': int(used_frames),
            'point_count_median': float(np.median(point_counts)),
            'point_count_mean': float(np.mean(point_counts)),
            'npoint_ratio': float(npoint_ratio),
            'nsample_ratio_mean': float(nsample_ratio_global),
            'target_cfg': args.target_cfg if args.target_cfg is not None else '',
            'match_mode': args.match_mode if args.target_cfg is not None else '',
            'generated_budget': generated_budget,
            'target_budget': target_budget if target_budget is not None else {},
        },
        'MODEL': policy_model,
    }
    summary = [
        f'Backbone NPOINTS: {base_npoints} -> {new_npoints}',
        f'Backbone NSAMPLE: {base_nsample_layers} -> {new_nsample_layers}',
        f'ROI NPOINTS: {roi_npoints} -> {roi_npoints_new}',
        f'ROI NSAMPLE: {roi_nsample} -> {roi_nsample_new}',
        f'ROI NUM_SAMPLED_POINTS: {roi_pool_num} -> {roi_pool_new}',
        f'Generated sampling budget: {generated_budget}',
    ]
    if target_budget is not None:
        summary.append(f'Target sampling budget: {target_budget}')
    return out_cfg, summary


def _search_policy_second(args, config, once_root, split_name, selected_items, used_frames, point_counts):
    data_cfg = config.DATA_CONFIG
    model_cfg = config.MODEL
    median_points = float(np.median(point_counts))

    idx_sp, sp_cfg = _find_processor_with_index(data_cfg, 'sample_points')
    idx_vox, vox_cfg = _find_processor_with_index(data_cfg, 'transform_points_to_voxels')
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
        new_sample_train = max(
            args.min_npoint,
            _round_to_multiple(int(round(base_sample_train * budget_ratio)), args.round_to)
        )
        new_sample_train = min(new_sample_train, base_sample_train)
        new_sample_test = max(
            args.min_npoint,
            _round_to_multiple(int(round(base_sample_test * budget_ratio)), args.round_to)
        )
        new_sample_test = min(new_sample_test, base_sample_test)

    new_vox_train = max(1000, _round_to_multiple(int(round(base_vox_train * budget_ratio)), args.round_to))
    new_vox_train = min(new_vox_train, base_vox_train)
    new_vox_test = max(1000, _round_to_multiple(int(round(base_vox_test * budget_ratio)), args.round_to))
    new_vox_test = min(new_vox_test, base_vox_test)

    base_num_filters = [int(x) for x in model_cfg.BACKBONE_2D.NUM_FILTERS]
    base_num_up_filters = [int(x) for x in model_cfg.BACKBONE_2D.NUM_UPSAMPLE_FILTERS]
    new_num_filters = list(base_num_filters)
    new_num_up_filters = list(base_num_up_filters)
    target_filter_budget = None
    target_upsample_budget = None

    if args.target_cfg is not None:
        target_cfg = cfg_from_yaml_file(args.target_cfg, type(config)())
        target_data = target_cfg.DATA_CONFIG
        target_model = target_cfg.MODEL
        target_idx_sp, target_sp = _find_processor_with_index(target_data, 'sample_points')
        target_idx_vox, target_vox = _find_processor_with_index(target_data, 'transform_points_to_voxels')
        if target_idx_vox < 0:
            raise RuntimeError('target_cfg for SECOND must include transform_points_to_voxels')

        target_vox_train = int(target_vox.MAX_NUMBER_OF_VOXELS['train'])
        target_vox_test = int(target_vox.MAX_NUMBER_OF_VOXELS['test'])
        target_sample_train = int(target_sp.NUM_POINTS['train']) if target_idx_sp >= 0 and idx_sp >= 0 else None
        target_sample_test = int(target_sp.NUM_POINTS['test']) if target_idx_sp >= 0 and idx_sp >= 0 else None
        target_num_filters = [int(x) for x in target_model.BACKBONE_2D.NUM_FILTERS]
        target_num_up_filters = [int(x) for x in target_model.BACKBONE_2D.NUM_UPSAMPLE_FILTERS]
        target_filter_budget = int(sum(target_num_filters))
        target_upsample_budget = int(sum(target_num_up_filters))

        if len(target_num_filters) != len(base_num_filters) or len(target_num_up_filters) != len(base_num_up_filters):
            raise RuntimeError('SECOND target_cfg BACKBONE_2D filter lengths must match baseline cfg')

        if args.match_mode == 'exact':
            new_vox_train, new_vox_test = target_vox_train, target_vox_test
            if target_sample_train is not None:
                new_sample_train, new_sample_test = target_sample_train, target_sample_test
            new_num_filters = target_num_filters
            new_num_up_filters = target_num_up_filters
        else:
            train_weight = 1.2 + (median_points / float(max(ref_num_points, 1)))
            test_weight = 1.0
            vox_alloc = _allocate_with_sum(
                target_vox_train + target_vox_test,
                [train_weight, test_weight],
                [1000, 1000],
                [base_vox_train, base_vox_test],
            )
            new_vox_train, new_vox_test = int(vox_alloc[0]), int(vox_alloc[1])

            if target_sample_train is not None:
                sample_alloc = _allocate_with_sum(
                    target_sample_train + target_sample_test,
                    [train_weight, test_weight],
                    [args.min_npoint, args.min_npoint],
                    [base_sample_train, base_sample_test],
                )
                new_sample_train, new_sample_test = int(sample_alloc[0]), int(sample_alloc[1])

            stage_weights = [
                1.1 + 0.15 * (len(base_num_filters) - 1 - idx)
                for idx in range(len(base_num_filters))
            ]
            upsample_weights = [1.0 for _ in base_num_up_filters]
            if args.match_mode == 'sum':
                all_weights = stage_weights + upsample_weights
                lowers = [args.round_to] * (len(base_num_filters) + len(base_num_up_filters))
                uppers = base_num_filters + base_num_up_filters
                target_total = target_filter_budget + target_upsample_budget
                allocated = _allocate_with_sum(target_total, all_weights, lowers, uppers)
                allocated = [_round_to_multiple(val, args.round_to) for val in allocated]
                allocated = [max(args.round_to, min(val, uppers[idx])) for idx, val in enumerate(allocated)]

                drift = target_total - int(sum(allocated))
                if drift != 0:
                    step = args.round_to
                    order = list(range(len(allocated)))
                    if drift < 0:
                        order = order[::-1]
                    while drift != 0:
                        updated = False
                        for idx in order:
                            if drift > 0 and allocated[idx] + step <= uppers[idx]:
                                allocated[idx] += step
                                drift -= step
                                updated = True
                            elif drift < 0 and allocated[idx] - step >= args.round_to:
                                allocated[idx] -= step
                                drift += step
                                updated = True
                            if drift == 0:
                                break
                        if not updated:
                            break

                new_num_filters = allocated[:len(base_num_filters)]
                new_num_up_filters = allocated[len(base_num_filters):]
            elif args.match_mode == 'groupwise_sum':
                new_num_filters = _allocate_rounded_with_sum(
                    target_filter_budget,
                    stage_weights,
                    [args.round_to] * len(base_num_filters),
                    base_num_filters,
                    args.round_to,
                )
                new_num_up_filters = _allocate_rounded_with_sum(
                    target_upsample_budget,
                    upsample_weights,
                    [args.round_to] * len(base_num_up_filters),
                    base_num_up_filters,
                    args.round_to,
                )

    data_override = {'DATA_PROCESSOR': []}
    for idx, proc in enumerate(data_cfg.DATA_PROCESSOR):
        proc_dict = _to_builtin_obj(proc)
        if idx == idx_sp and base_sample_train is not None:
            proc_dict['NUM_POINTS']['train'] = int(new_sample_train)
            proc_dict['NUM_POINTS']['test'] = int(new_sample_test)
        if idx == idx_vox:
            proc_dict['MAX_NUMBER_OF_VOXELS']['train'] = int(new_vox_train)
            proc_dict['MAX_NUMBER_OF_VOXELS']['test'] = int(new_vox_test)
        data_override['DATA_PROCESSOR'].append(proc_dict)

    model_override = {
        'BACKBONE_2D': {
            'NUM_FILTERS': [int(x) for x in new_num_filters],
            'NUM_UPSAMPLE_FILTERS': [int(x) for x in new_num_up_filters],
        }
    }

    out_cfg = {
        'META': {
            'policy_type': 'offline_data_adaptive_sampling',
            'dataset': 'ONCE',
            'model_name': str(model_cfg.NAME),
            'source_cfg': args.cfg_file,
            'data_path': str(once_root),
            'split': split_name,
            'entries_selected': int(len(selected_items)),
            'used_frames': int(used_frames),
            'point_count_median': float(np.median(point_counts)),
            'point_count_mean': float(np.mean(point_counts)),
            'budget_ratio': float(budget_ratio),
            'target_cfg': args.target_cfg if args.target_cfg is not None else '',
            'match_mode': args.match_mode if args.target_cfg is not None else '',
            'target_filter_budget': target_filter_budget if target_filter_budget is not None else 0,
            'target_upsample_budget': target_upsample_budget if target_upsample_budget is not None else 0,
            'generated_filter_budget': int(sum(new_num_filters)),
            'generated_upsample_budget': int(sum(new_num_up_filters)),
        },
        'MODEL': model_override,
        'DATA_CONFIG': data_override,
    }

    summary = [
        f'BACKBONE_2D.NUM_FILTERS: {base_num_filters} -> {new_num_filters}',
        f'BACKBONE_2D.NUM_UPSAMPLE_FILTERS: {base_num_up_filters} -> {new_num_up_filters}',
    ]
    if target_filter_budget is not None:
        summary.append(
            f'BEV channel budgets filter/upsample: '
            f'{sum(new_num_filters)}/{sum(new_num_up_filters)} '
            f'(target {target_filter_budget}/{target_upsample_budget})'
        )
    if base_sample_train is not None:
        summary.append(
            f'sample_points.NUM_POINTS train/test: [{base_sample_train}, {base_sample_test}] '
            f'-> [{new_sample_train}, {new_sample_test}]'
        )
    summary.append(
        f'transform_points_to_voxels.MAX_NUMBER_OF_VOXELS train/test: '
        f'[{base_vox_train}, {base_vox_test}] -> [{new_vox_train}, {new_vox_test}]'
    )
    return out_cfg, summary


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cfg_from_yaml_file(args.cfg_file, cfg)

    model_name = str(cfg.MODEL.NAME)

    once_root = Path(args.data_path) if args.data_path is not None else Path(cfg.DATA_CONFIG.DATA_PATH)
    split_name = args.split if args.split is not None else cfg.DATA_CONFIG.DATA_SPLIT['train']
    split_file = once_root / 'ImageSets' / f'{split_name}.txt'
    if not split_file.exists():
        raise FileNotFoundError(
            f'Split file not found: {split_file}. Run from tools/ as in OpenPCDet training, '
            f'or pass --data_path explicitly.'
        )

    items = _load_once_split_items(split_file)
    if not items:
        raise RuntimeError(f'No frame-level samples in split: {split_file}')
    selected_items = _pick_items(items, args.num_frames, rng)

    radii = []
    if model_name == 'PointRCNN':
        for layer_radii in cfg.MODEL.BACKBONE_3D.SA_CONFIG.RADIUS:
            radii.extend(layer_radii)
    elif model_name.startswith('SECOND'):
        radii = [0.2, 0.4, 0.8]
    else:
        raise ValueError(f'Unsupported ONCE model for policy search: {model_name}')

    point_counts = []
    neigh_sum = np.zeros(len(radii), dtype=np.float64)
    missing_files = []
    for seq_id, frame_id, domain in selected_items:
        lidar_file = _resolve_once_lidar_file(once_root, seq_id, frame_id, domain)
        if not lidar_file.exists():
            missing_files.append(str(lidar_file))
            continue
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)[:, :3]
        points = _mask_by_range(points, cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        if points.shape[0] < 4:
            continue
        point_counts.append(points.shape[0])
        neigh_sum += np.array(
            _neighbor_count_stats(points, radii, args.num_query, args.num_ref, rng),
            dtype=np.float64,
        )

    if missing_files:
        raise FileNotFoundError(
            f'{len(missing_files)} selected ONCE lidar files are missing, example: {missing_files[0]}'
        )
    used_frames = len(point_counts)
    if used_frames == 0:
        raise RuntimeError('No valid ONCE frame was found for policy search.')

    point_counts = np.asarray(point_counts, dtype=np.float64)
    neigh_mean = neigh_sum / float(used_frames)
    if model_name == 'PointRCNN':
        out_cfg, summary = _search_policy_pointrcnn(
            args=args,
            config=cfg,
            once_root=once_root,
            split_name=split_name,
            used_frames=used_frames,
            point_counts=point_counts,
            neigh_mean=neigh_mean,
        )
    elif model_name.startswith('SECOND'):
        out_cfg, summary = _search_policy_second(
            args=args,
            config=cfg,
            once_root=once_root,
            split_name=split_name,
            selected_items=selected_items,
            used_frames=used_frames,
            point_counts=point_counts,
        )
    else:
        raise ValueError(f'Unsupported ONCE model for policy search: {model_name}')

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)

    print('=== ONCE Offline Sampling Policy Search Summary ===')
    print(f'Split file: {split_file}')
    print(f'Entries selected: {len(selected_items)}')
    print(f'Frames used: {used_frames}')
    print(f'Median points in range: {np.median(point_counts):.1f}')
    for line in summary:
        print(line)
    print(f'Policy saved to: {out_file}')


if __name__ == '__main__':
    main()
