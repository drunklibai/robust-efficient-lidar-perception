"""
Export a pruned SECOND config and checkpoint from Network Slimming BN-gamma
statistics.

This script is SECOND-specific. It prunes every part that has BN gamma and a
tractable channel dependency in the default VoxelBackBone8x + BaseBEVBackbone:

  - backbone_3d conv_input / conv1 / conv2 / conv3 / conv4 / conv_out
  - backbone_2d blocks / deblocks
  - dense_head input channels are sliced to match the pruned BEV output

The generated config relies on the optional MODEL.BACKBONE_3D.CHANNELS field.
If that field is absent, VoxelBackBone8x keeps its original hard-coded channel
sizes, so existing configs are unchanged.
"""
import argparse
import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Export SECOND Network Slimming pruned cfg/ckpt')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_cfg', type=str, required=True)
    parser.add_argument('--output_ckpt', type=str, required=True)
    parser.add_argument('--prune_ratio', type=float, default=0.3)
    parser.add_argument('--round_to', type=int, default=8)
    parser.add_argument('--min_channel', type=int, default=8)
    return parser.parse_args()


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def get_model_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        return ckpt, ckpt['model_state']
    return {'model_state': ckpt}, ckpt


def round_channels(n, round_to, min_channel, max_channel):
    n = max(min_channel, int(round(n)))
    if round_to > 1:
        n = max(round_to, int(round(n / round_to) * round_to))
    return min(n, max_channel)


def keep_indices_from_score(score, threshold, round_to, min_channel):
    score = np.asarray(score, dtype=np.float64)
    keep_n = int((score > threshold).sum())
    keep_n = round_channels(keep_n, round_to, min_channel, len(score))
    idx = np.argsort(-score)[:keep_n]
    return np.sort(idx)


def bn_score(sd, bn_prefix):
    return sd[f'{bn_prefix}.weight'].detach().abs().cpu().numpy()


def compute_threshold(sd, bn_prefixes, prune_ratio):
    all_gamma = [bn_score(sd, x) for x in bn_prefixes]
    all_gamma = np.concatenate(all_gamma)
    sorted_gamma = np.sort(all_gamma)
    idx = min(int(len(sorted_gamma) * prune_ratio), len(sorted_gamma) - 1)
    return float(sorted_gamma[idx]), all_gamma


def t(idx):
    return torch.from_numpy(np.asarray(idx, dtype=np.int64))


def copy_bn(old_sd, new_sd, bn_prefix, keep_idx):
    keep = t(keep_idx)
    for suffix in ['weight', 'bias', 'running_mean', 'running_var']:
        key = f'{bn_prefix}.{suffix}'
        if key in old_sd:
            new_sd[key] = old_sd[key].index_select(0, keep).contiguous()
    key = f'{bn_prefix}.num_batches_tracked'
    if key in old_sd:
        new_sd[key] = old_sd[key]


def slice_sparse_weight(weight, in_idx, out_idx, orig_in, orig_out):
    w = weight
    # spconv 1.x commonly stores [kz, ky, kx, in, out].
    if w.dim() == 5:
        if w.shape[-2] == orig_in and w.shape[-1] == orig_out:
            if in_idx is not None:
                w = w.index_select(-2, t(in_idx))
            w = w.index_select(-1, t(out_idx))
            return w.contiguous()
        if w.shape[1] == orig_in and w.shape[0] == orig_out:
            w = w.index_select(0, t(out_idx))
            if in_idx is not None:
                w = w.index_select(1, t(in_idx))
            return w.contiguous()

    # Fallback for less common layouts: identify dimensions by original sizes.
    out_dims = [i for i, s in enumerate(w.shape) if s == orig_out]
    in_dims = [i for i, s in enumerate(w.shape) if s == orig_in]
    if not out_dims:
        raise RuntimeError(f'Cannot find output channel dim in sparse weight shape {tuple(weight.shape)}')
    out_dim = out_dims[-1] if w.dim() == 5 else out_dims[0]
    if in_idx is not None:
        in_dim = next((i for i in in_dims if i != out_dim), None)
        if in_dim is None:
            raise RuntimeError(f'Cannot find input channel dim in sparse weight shape {tuple(weight.shape)}')
        if in_dim > out_dim:
            w = w.index_select(out_dim, t(out_idx))
            w = w.index_select(in_dim, t(in_idx))
        else:
            w = w.index_select(in_dim, t(in_idx))
            w = w.index_select(out_dim, t(out_idx))
    else:
        w = w.index_select(out_dim, t(out_idx))
    return w.contiguous()


def slice_conv2d_weight(weight, in_idx, out_idx):
    w = weight.index_select(0, t(out_idx))
    if in_idx is not None:
        w = w.index_select(1, t(in_idx))
    return w.contiguous()


def slice_deconv2d_weight(weight, in_idx, out_idx):
    # ConvTranspose2d weight shape is [in_channels, out_channels, kH, kW].
    w = weight
    if in_idx is not None:
        w = w.index_select(0, t(in_idx))
    w = w.index_select(1, t(out_idx))
    return w.contiguous()


def replace_conv_bias(old_sd, new_sd, conv_prefix, out_idx):
    key = f'{conv_prefix}.bias'
    if key in old_sd:
        new_sd[key] = old_sd[key].index_select(0, t(out_idx)).contiguous()


def prune_sparse_conv_bn(old_sd, new_sd, conv_prefix, bn_prefix, in_idx, out_idx, orig_in, orig_out):
    new_sd[f'{conv_prefix}.weight'] = slice_sparse_weight(
        old_sd[f'{conv_prefix}.weight'], in_idx, out_idx, orig_in, orig_out
    )
    replace_conv_bias(old_sd, new_sd, conv_prefix, out_idx)
    copy_bn(old_sd, new_sd, bn_prefix, out_idx)


def prune_conv_bn(old_sd, new_sd, conv_prefix, bn_prefix, in_idx, out_idx):
    new_sd[f'{conv_prefix}.weight'] = slice_conv2d_weight(old_sd[f'{conv_prefix}.weight'], in_idx, out_idx)
    replace_conv_bias(old_sd, new_sd, conv_prefix, out_idx)
    copy_bn(old_sd, new_sd, bn_prefix, out_idx)


def prune_deconv_bn(old_sd, new_sd, conv_prefix, bn_prefix, in_idx, out_idx):
    new_sd[f'{conv_prefix}.weight'] = slice_deconv2d_weight(old_sd[f'{conv_prefix}.weight'], in_idx, out_idx)
    replace_conv_bias(old_sd, new_sd, conv_prefix, out_idx)
    copy_bn(old_sd, new_sd, bn_prefix, out_idx)


def aggregate_block_score(sd, bn_prefixes):
    vals = [bn_score(sd, x) for x in bn_prefixes]
    return np.stack(vals, axis=0).mean(axis=0)


def build_keep_info(cfg, sd, threshold, round_to, min_channel):
    layer_nums = cfg['MODEL']['BACKBONE_2D']['LAYER_NUMS']

    info = OrderedDict()
    sparse_specs = [
        ('conv_input', 'backbone_3d.conv_input.0', 'backbone_3d.conv_input.1', 4, 16),
        ('conv1', 'backbone_3d.conv1.0.0', 'backbone_3d.conv1.0.1', 16, 16),
        ('conv2_0', 'backbone_3d.conv2.0.0', 'backbone_3d.conv2.0.1', 16, 32),
        ('conv2_1', 'backbone_3d.conv2.1.0', 'backbone_3d.conv2.1.1', 32, 32),
        ('conv2_2', 'backbone_3d.conv2.2.0', 'backbone_3d.conv2.2.1', 32, 32),
        ('conv3_0', 'backbone_3d.conv3.0.0', 'backbone_3d.conv3.0.1', 32, 64),
        ('conv3_1', 'backbone_3d.conv3.1.0', 'backbone_3d.conv3.1.1', 64, 64),
        ('conv3_2', 'backbone_3d.conv3.2.0', 'backbone_3d.conv3.2.1', 64, 64),
        ('conv4_0', 'backbone_3d.conv4.0.0', 'backbone_3d.conv4.0.1', 64, 64),
        ('conv4_1', 'backbone_3d.conv4.1.0', 'backbone_3d.conv4.1.1', 64, 64),
        ('conv4_2', 'backbone_3d.conv4.2.0', 'backbone_3d.conv4.2.1', 64, 64),
        ('conv_out', 'backbone_3d.conv_out.0', 'backbone_3d.conv_out.1', 64, 128),
    ]

    info['sparse_specs'] = sparse_specs
    info['sparse_keep'] = OrderedDict()
    for name, _, bn_prefix, _, _ in sparse_specs:
        info['sparse_keep'][name] = keep_indices_from_score(
            bn_score(sd, bn_prefix), threshold, round_to, min_channel
        )

    info['block_keep'] = []
    info['block_bn_prefixes'] = []
    for block_idx, layer_num in enumerate(layer_nums):
        bn_prefixes = [f'backbone_2d.blocks.{block_idx}.2']
        for k in range(layer_num):
            bn_prefixes.append(f'backbone_2d.blocks.{block_idx}.{5 + 3 * k}')
        info['block_bn_prefixes'].append(bn_prefixes)
        info['block_keep'].append(keep_indices_from_score(
            aggregate_block_score(sd, bn_prefixes), threshold, round_to, min_channel
        ))

    info['deblock_keep'] = []
    for deblock_idx in range(len(cfg['MODEL']['BACKBONE_2D']['NUM_UPSAMPLE_FILTERS'])):
        info['deblock_keep'].append(keep_indices_from_score(
            bn_score(sd, f'backbone_2d.deblocks.{deblock_idx}.1'),
            threshold, round_to, min_channel
        ))

    return info


def make_bev_keep(conv_out_keep, depth_multiplier):
    keep = []
    for c in conv_out_keep:
        for d in range(depth_multiplier):
            keep.append(int(c) * depth_multiplier + d)
    return np.asarray(keep, dtype=np.int64)


def build_pruned_config(cfg, info):
    out = copy.deepcopy(cfg)
    sk = info['sparse_keep']
    out['MODEL']['BACKBONE_3D']['CHANNELS'] = {
        'conv_input': int(len(sk['conv_input'])),
        'conv1': int(len(sk['conv1'])),
        'conv2': [int(len(sk['conv2_0'])), int(len(sk['conv2_1'])), int(len(sk['conv2_2']))],
        'conv3': [int(len(sk['conv3_0'])), int(len(sk['conv3_1'])), int(len(sk['conv3_2']))],
        'conv4': [int(len(sk['conv4_0'])), int(len(sk['conv4_1'])), int(len(sk['conv4_2']))],
        'conv_out': int(len(sk['conv_out'])),
    }

    orig_bev = int(cfg['MODEL']['MAP_TO_BEV']['NUM_BEV_FEATURES'])
    depth_multiplier = orig_bev // 128
    out['MODEL']['MAP_TO_BEV']['NUM_BEV_FEATURES'] = int(len(sk['conv_out']) * depth_multiplier)
    out['MODEL']['BACKBONE_2D']['NUM_FILTERS'] = [int(len(x)) for x in info['block_keep']]
    out['MODEL']['BACKBONE_2D']['NUM_UPSAMPLE_FILTERS'] = [int(len(x)) for x in info['deblock_keep']]

    if 'SLIMMING' in out.get('MODEL', {}):
        out['MODEL']['SLIMMING']['ENABLED'] = False
    if 'GMP_PRUNING' in out.get('MODEL', {}):
        out['MODEL']['GMP_PRUNING']['ENABLED'] = False
    if 'IMP_PRUNING' in out.get('MODEL', {}):
        out['MODEL']['IMP_PRUNING']['ENABLED'] = False
    return out


def build_pruned_state_dict(cfg, old_sd, info):
    new_sd = OrderedDict((k, v) for k, v in old_sd.items())

    prev_keep = None
    for name, conv_prefix, bn_prefix, orig_in, orig_out in info['sparse_specs']:
        out_keep = info['sparse_keep'][name]
        prune_sparse_conv_bn(old_sd, new_sd, conv_prefix, bn_prefix, prev_keep, out_keep, orig_in, orig_out)
        prev_keep = out_keep

    orig_bev = int(cfg['MODEL']['MAP_TO_BEV']['NUM_BEV_FEATURES'])
    depth_multiplier = orig_bev // 128
    block_in_keep = make_bev_keep(info['sparse_keep']['conv_out'], depth_multiplier)

    for block_idx, layer_num in enumerate(cfg['MODEL']['BACKBONE_2D']['LAYER_NUMS']):
        out_keep = info['block_keep'][block_idx]
        prune_conv_bn(
            old_sd, new_sd,
            f'backbone_2d.blocks.{block_idx}.1',
            f'backbone_2d.blocks.{block_idx}.2',
            block_in_keep, out_keep
        )
        for k in range(layer_num):
            conv_id = 4 + 3 * k
            bn_id = 5 + 3 * k
            prune_conv_bn(
                old_sd, new_sd,
                f'backbone_2d.blocks.{block_idx}.{conv_id}',
                f'backbone_2d.blocks.{block_idx}.{bn_id}',
                out_keep, out_keep
            )
        block_in_keep = out_keep

    deblock_input_keeps = info['block_keep']
    deblock_output_keeps = info['deblock_keep']
    for deblock_idx, (in_keep, out_keep) in enumerate(zip(deblock_input_keeps, deblock_output_keeps)):
        conv_prefix = f'backbone_2d.deblocks.{deblock_idx}.0'
        bn_prefix = f'backbone_2d.deblocks.{deblock_idx}.1'
        prune_deconv_bn(old_sd, new_sd, conv_prefix, bn_prefix, in_keep, out_keep)

    dense_in_keep = []
    offset = 0
    orig_up_filters = cfg['MODEL']['BACKBONE_2D']['NUM_UPSAMPLE_FILTERS']
    for keep, orig_ch in zip(deblock_output_keeps, orig_up_filters):
        dense_in_keep.extend((keep + offset).tolist())
        offset += int(orig_ch)
    dense_in_keep = np.asarray(dense_in_keep, dtype=np.int64)

    for head in ['conv_cls', 'conv_box', 'conv_dir_cls']:
        key = f'dense_head.{head}.weight'
        if key not in old_sd:
            continue
        new_sd[key] = old_sd[key].index_select(1, t(dense_in_keep)).contiguous()
        bias_key = f'dense_head.{head}.bias'
        if bias_key in old_sd:
            new_sd[bias_key] = old_sd[bias_key]

    return new_sd


def collect_target_bn_prefixes(cfg):
    prefixes = []
    prefixes.extend([
        'backbone_3d.conv_input.1',
        'backbone_3d.conv1.0.1',
        'backbone_3d.conv2.0.1', 'backbone_3d.conv2.1.1', 'backbone_3d.conv2.2.1',
        'backbone_3d.conv3.0.1', 'backbone_3d.conv3.1.1', 'backbone_3d.conv3.2.1',
        'backbone_3d.conv4.0.1', 'backbone_3d.conv4.1.1', 'backbone_3d.conv4.2.1',
        'backbone_3d.conv_out.1',
    ])
    for block_idx, layer_num in enumerate(cfg['MODEL']['BACKBONE_2D']['LAYER_NUMS']):
        prefixes.append(f'backbone_2d.blocks.{block_idx}.2')
        for k in range(layer_num):
            prefixes.append(f'backbone_2d.blocks.{block_idx}.{5 + 3 * k}')
    for deblock_idx in range(len(cfg['MODEL']['BACKBONE_2D']['NUM_UPSAMPLE_FILTERS'])):
        prefixes.append(f'backbone_2d.deblocks.{deblock_idx}.1')
    return prefixes


def print_summary(cfg, out_cfg, info, threshold, all_gamma, args):
    print('=== SECOND Network Slimming Export Summary ===')
    print(f'Input cfg: {args.cfg_file}')
    print(f'Input ckpt: {args.ckpt}')
    print(f'Total BN gamma params: {len(all_gamma)}')
    print(f'Prune ratio: {args.prune_ratio}')
    print(f'Global gamma threshold: {threshold:.8f}')
    print('3D channels:')
    print(f"  conv_input: 16 -> {len(info['sparse_keep']['conv_input'])}")
    print(f"  conv1: 16 -> {len(info['sparse_keep']['conv1'])}")
    print(f"  conv2: [32, 32, 32] -> {out_cfg['MODEL']['BACKBONE_3D']['CHANNELS']['conv2']}")
    print(f"  conv3: [64, 64, 64] -> {out_cfg['MODEL']['BACKBONE_3D']['CHANNELS']['conv3']}")
    print(f"  conv4: [64, 64, 64] -> {out_cfg['MODEL']['BACKBONE_3D']['CHANNELS']['conv4']}")
    print(f"  conv_out: 128 -> {len(info['sparse_keep']['conv_out'])}")
    print('2D channels:')
    print(f"  NUM_FILTERS: {cfg['MODEL']['BACKBONE_2D']['NUM_FILTERS']} -> {out_cfg['MODEL']['BACKBONE_2D']['NUM_FILTERS']}")
    print(f"  NUM_UPSAMPLE_FILTERS: {cfg['MODEL']['BACKBONE_2D']['NUM_UPSAMPLE_FILTERS']} -> {out_cfg['MODEL']['BACKBONE_2D']['NUM_UPSAMPLE_FILTERS']}")
    print(f"  MAP_TO_BEV.NUM_BEV_FEATURES: {cfg['MODEL']['MAP_TO_BEV']['NUM_BEV_FEATURES']} -> {out_cfg['MODEL']['MAP_TO_BEV']['NUM_BEV_FEATURES']}")
    print(f'Output cfg: {args.output_cfg}')
    print(f'Output ckpt: {args.output_ckpt}')


def main():
    args = parse_args()
    cfg = load_yaml(args.cfg_file)
    ckpt, old_sd = get_model_state(args.ckpt)

    bn_prefixes = collect_target_bn_prefixes(cfg)
    missing = [x for x in bn_prefixes if f'{x}.weight' not in old_sd]
    if missing:
        raise KeyError(f'Missing BN weights in checkpoint: {missing[:5]}')

    threshold, all_gamma = compute_threshold(old_sd, bn_prefixes, args.prune_ratio)
    info = build_keep_info(cfg, old_sd, threshold, args.round_to, args.min_channel)
    out_cfg = build_pruned_config(cfg, info)
    new_sd = build_pruned_state_dict(cfg, old_sd, info)

    save_yaml(out_cfg, args.output_cfg)
    out_path = Path(args.output_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ckpt = {
        'model_state': new_sd,
        'slimming_meta': {
            'source_ckpt': args.ckpt,
            'prune_ratio': args.prune_ratio,
            'threshold': threshold,
            'round_to': args.round_to,
            'min_channel': args.min_channel,
        }
    }
    for key in ['version', 'epoch', 'it']:
        if key in ckpt:
            out_ckpt[key] = ckpt[key]
    torch.save(out_ckpt, str(out_path))
    print_summary(cfg, out_cfg, info, threshold, all_gamma, args)


if __name__ == '__main__':
    main()
