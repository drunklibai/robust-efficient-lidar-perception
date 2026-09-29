"""
Export pruned PointRCNN config + weight-transferred checkpoint from
slimming-trained BN-gamma statistics.

**True structured pruning**: each BN layer's gamma identifies WHICH channels
are important. The surviving channel indices are used to slice the corresponding
Conv/Linear/BN weights, producing a smaller checkpoint that inherits the
learned representations.  Only short fine-tuning (10-20 epochs) is needed.

Workflow:
  1. Train with SLIMMING.ENABLED=True  →  checkpoint with sparse BN gamma
  2. Run this script                    →  pruned yaml  +  pruned ckpt
  3. Fine-tune with the pruned yaml/ckpt→  final compact model

Usage:
    python export_slimming_pruned_cfg.py \\
        --cfg_file cfgs/kitti_models/pointrcnn_mixed_slimming_train.yaml \\
        --ckpt ../output/.../checkpoint_epoch_80.pth \\
        --output_cfg cfgs/kitti_models/pointrcnn_pruned_r30.yaml \\
        --output_ckpt ../output/pruned_r30_init.pth \\
        --prune_ratio 0.3
"""
import _init_path
import argparse
import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Export pruned PointRCNN cfg + checkpoint from per-layer BN-gamma')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_cfg', type=str, required=True)
    parser.add_argument('--output_ckpt', type=str, default=None,
                        help='Output pruned checkpoint path (weight transfer)')
    parser.add_argument('--prune_ratio', type=float, default=0.3)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--round_to', type=int, default=8)
    parser.add_argument('--min_channel', type=int, default=8)
    return parser.parse_args()


def _round_ch(c, round_to, min_ch):
    c = max(min_ch, int(round(c)))
    if round_to > 1:
        c = int(max(round_to, round(c / round_to) * round_to))
    return c


def _load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _save_yaml(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# ═══════════════════════════════════════════════════════════════════════════
# BN gamma collection — returns both gammas and keep-indices
# ═══════════════════════════════════════════════════════════════════════════

def collect_bn_gammas(model):
    """Return OrderedDict {name: abs(gamma) ndarray}."""
    gammas = OrderedDict()
    for name, m in model.named_modules():
        if m.__class__.__name__ in ('BatchNorm1d', 'BatchNorm2d'):
            if hasattr(m, 'weight') and m.weight is not None:
                gammas[name] = m.weight.detach().abs().cpu().numpy()
    return gammas


def compute_global_threshold(gammas, prune_ratio):
    all_g = np.concatenate(list(gammas.values()))
    sorted_g = np.sort(all_g)
    idx = min(int(len(sorted_g) * prune_ratio), len(sorted_g) - 1)
    return float(sorted_g[idx]), all_g


def gamma_keep_indices(gamma, threshold, round_to, min_ch):
    """
    Return indices of channels to keep.  Picks top-gamma channels, rounds
    count up to alignment, enforces minimum.
    """
    n_keep = _round_ch(int((gamma > threshold).sum()), round_to, min_ch)
    n_keep = min(n_keep, len(gamma))
    # Sort by gamma descending, take top n_keep
    idx = np.argsort(-gamma)[:n_keep]
    return np.sort(idx)  # return in ascending order for consistent slicing


# ═══════════════════════════════════════════════════════════════════════════
# Per-layer keep-index extraction
# ═══════════════════════════════════════════════════════════════════════════

def _seq_bn_indices(gammas, prefix, threshold, R, M):
    """Extract keep indices from Conv-BN-ReLU Sequential (stride 3)."""
    result = []
    j = 0
    while True:
        bn_name = f'{prefix}.{3 * j + 1}'
        if bn_name not in gammas:
            break
        result.append((bn_name, gamma_keep_indices(gammas[bn_name], threshold, R, M)))
        j += 1
    return result


def _roi_fc_bn_indices(gammas, prefix, threshold, R, M):
    """Extract keep indices from RoIHead FC layers (may have Dropout)."""
    bn_items = []
    for name in gammas:
        if name.startswith(prefix + '.'):
            suffix = name[len(prefix) + 1:]
            if suffix.isdigit():
                bn_items.append((int(suffix), name))
    bn_items.sort()
    return [(n, gamma_keep_indices(gammas[n], threshold, R, M)) for _, n in bn_items]


def extract_all_keep_indices(gammas, threshold, R, M):
    """
    Returns a structure mapping each BN layer name → keep indices,
    organized by model component.
    """
    info = {}

    # ── Backbone SA ── backbone_3d.SA_modules.{k}.mlps.{s}.{3j+1}
    sa = []
    k = 0
    while True:
        prefix = f'backbone_3d.SA_modules.{k}'
        if not any(n.startswith(prefix) for n in gammas):
            break
        level = []
        s = 0
        while True:
            items = _seq_bn_indices(gammas, f'{prefix}.mlps.{s}', threshold, R, M)
            if not items:
                break
            level.append(items)
            s += 1
        sa.append(level)
        k += 1
    info['backbone_sa'] = sa

    # ── Backbone FP ── backbone_3d.FP_modules.{k}.mlp.{3j+1}
    fp = []
    k = 0
    while True:
        items = _seq_bn_indices(gammas, f'backbone_3d.FP_modules.{k}.mlp', threshold, R, M)
        if not items:
            break
        fp.append(items)
        k += 1
    info['backbone_fp'] = fp

    # ── Point head ──
    info['ph_cls'] = _seq_bn_indices(gammas, 'point_head.cls_layers', threshold, R, M)
    info['ph_reg'] = _seq_bn_indices(gammas, 'point_head.box_layers', threshold, R, M)

    # ── ROI head SA ──
    roi_sa = []
    k = 0
    while True:
        items = _seq_bn_indices(gammas, f'roi_head.SA_modules.{k}.mlps.0', threshold, R, M)
        if not items:
            break
        roi_sa.append(items)
        k += 1
    info['roi_sa'] = roi_sa

    # ── ROI head FC ──
    info['roi_cls'] = _roi_fc_bn_indices(gammas, 'roi_head.cls_layers', threshold, R, M)
    info['roi_reg'] = _roi_fc_bn_indices(gammas, 'roi_head.reg_layers', threshold, R, M)

    # ── ROI xyz_up_layer (USE_BN=True) ──
    info['roi_xyz'] = _seq_bn_indices(gammas, 'roi_head.xyz_up_layer', threshold, R, M)

    return info


# ═══════════════════════════════════════════════════════════════════════════
# Weight transfer: slice old state_dict → new (smaller) state_dict
# ═══════════════════════════════════════════════════════════════════════════

def _t(idx):
    """Convert numpy index array to torch LongTensor."""
    return torch.from_numpy(idx.astype(np.int64))


def _prune_conv_bn_relu_seq(old_sd, prefix, bn_keep_list, in_keep_idx,
                             new_sd, conv_dims=4, use_xyz_offset=0):
    """
    Prune a Sequential of (Conv-BN-ReLU)* blocks.

    Args:
        old_sd: full state_dict
        prefix: e.g. 'backbone_3d.SA_modules.0.mlps.0'
        bn_keep_list: list of (bn_name, keep_indices) from _seq_bn_indices
        in_keep_idx: input channel keep indices (numpy), or None for first layer
        new_sd: output state_dict (mutated)
        conv_dims: 4 for Conv2d, 3 for Conv1d
        use_xyz_offset: if >0, the first conv's input has this many extra fixed
                        channels (xyz) that are always kept
    Returns:
        out_keep_idx: output channel keep indices of last BN
    """
    cur_in = in_keep_idx
    out_keep = None

    for j, (bn_name, keep_idx) in enumerate(bn_keep_list):
        conv_key = f'{prefix}.{3 * j}'
        bn_key = f'{prefix}.{3 * j + 1}'

        W = old_sd[f'{conv_key}.weight']  # (out, in, [1, 1]) or (out, in, 1)
        keep = _t(keep_idx)

        # Select output channels
        W = torch.index_select(W, 0, keep)

        # Select input channels
        if cur_in is not None:
            full_in = cur_in
            if j == 0 and use_xyz_offset > 0:
                # First conv gets extra xyz channels prepended by PointNet2
                xyz_idx = np.arange(use_xyz_offset)
                full_in = np.concatenate([xyz_idx, cur_in + use_xyz_offset])
            W = torch.index_select(W, 1, _t(full_in))

        new_sd[f'{conv_key}.weight'] = W
        if f'{conv_key}.bias' in old_sd:
            new_sd[f'{conv_key}.bias'] = old_sd[f'{conv_key}.bias'][keep]

        # BN params (weight, bias, running_mean, running_var)
        for suffix in ['weight', 'bias', 'running_mean', 'running_var']:
            key = f'{bn_key}.{suffix}'
            if key in old_sd:
                new_sd[key] = old_sd[key][keep]
        if f'{bn_key}.num_batches_tracked' in old_sd:
            new_sd[f'{bn_key}.num_batches_tracked'] = old_sd[f'{bn_key}.num_batches_tracked']

        cur_in = keep_idx
        out_keep = keep_idx

    return out_keep


def _prune_linear_bn_relu_seq(old_sd, prefix, bn_keep_list, in_keep_idx, new_sd):
    """
    Like _prune_conv_bn_relu_seq but for Linear layers (PointHeadTemplate.make_fc_layers).
    Linear weight shape: (out, in).
    """
    cur_in = in_keep_idx
    out_keep = None

    for j, (bn_name, keep_idx) in enumerate(bn_keep_list):
        lin_key = f'{prefix}.{3 * j}'
        bn_key = f'{prefix}.{3 * j + 1}'

        W = old_sd[f'{lin_key}.weight']  # (out, in)
        keep = _t(keep_idx)

        W = torch.index_select(W, 0, keep)
        if cur_in is not None:
            W = torch.index_select(W, 1, _t(cur_in))

        new_sd[f'{lin_key}.weight'] = W
        if f'{lin_key}.bias' in old_sd:
            new_sd[f'{lin_key}.bias'] = old_sd[f'{lin_key}.bias'][keep]

        for suffix in ['weight', 'bias', 'running_mean', 'running_var']:
            key = f'{bn_key}.{suffix}'
            if key in old_sd:
                new_sd[key] = old_sd[key][keep]
        if f'{bn_key}.num_batches_tracked' in old_sd:
            new_sd[f'{bn_key}.num_batches_tracked'] = old_sd[f'{bn_key}.num_batches_tracked']

        cur_in = keep_idx
        out_keep = keep_idx

    return out_keep


def _prune_final_layer(old_sd, key_prefix, in_keep_idx, new_sd, is_conv=True):
    """Prune the final output layer (no BN) — only slices input dim."""
    for suffix in ['weight', 'bias']:
        key = f'{key_prefix}.{suffix}'
        if key not in old_sd:
            continue
        W = old_sd[key]
        if suffix == 'weight' and in_keep_idx is not None:
            W = torch.index_select(W, 1, _t(in_keep_idx))
        new_sd[key] = W


def build_pruned_state_dict(old_sd, keep_info):
    """
    Build pruned state_dict by selecting surviving channels from the
    slimming-trained model.

    Transfers weights for:
      - backbone_3d  (SA + FP modules, with proper multi-scale concat
                      and skip-connection input tracking)
      - point_head   (cls_layers + box_layers)

    Skips (re-initialized by model's init_weights):
      - roi_head     (xyz_up_layer and merge_down_layer have no BN when
                      USE_BN=False, so gamma-based channel selection can't
                      determine which channels to keep; the entire ROI head
                      is excluded and will re-initialize via xavier_init
                      during model construction)

    Returns:
        OrderedDict with pruned weights for backbone_3d + point_head.
        The caller should load with strict=False so that ROI head keys
        fall back to their freshly-initialized values.
    """
    new_sd = OrderedDict()

    sa_info = keep_info['backbone_sa']
    fp_info = keep_info['backbone_fp']
    num_sa = len(sa_info)
    num_fp = len(fp_info)

    # ── Compute original channel bookkeeping from state_dict shapes ──

    # SA[k]: track per-scale original output channels and concatenated keep indices
    sa_orig_scale_out = []   # sa_orig_scale_out[k][s] = int
    sa_concat_keep = []      # sa_concat_keep[k] = np.array (keep indices in concat space)

    for k in range(num_sa):
        scales_ch = []
        parts = []
        offset = 0
        for s, bn_list in enumerate(sa_info[k]):
            last_bn_name = bn_list[-1][0]
            orig_ch = old_sd[f'{last_bn_name}.weight'].shape[0]
            scales_ch.append(orig_ch)
            parts.append(bn_list[-1][1] + offset)   # offset into concat space
            offset += orig_ch
        sa_orig_scale_out.append(scales_ch)
        sa_concat_keep.append(np.concatenate(parts))

    sa_orig_total = [sum(s) for s in sa_orig_scale_out]
    # sa_orig_total[k] = total original output channels of SA[k]

    # skip_channel_list (original) — used to decompose FP input into [deeper|skip]
    input_feat_ch = 1  # KITTI: 4 point features − 3 xyz
    skip_ch_list = [input_feat_ch] + sa_orig_total
    # skip_ch_list[k] == the original skip-connection channel count for FP[k]

    # FP original output channels (from last BN weight shape)
    fp_orig_out_ch = []
    for k in range(num_fp):
        last_bn_name = fp_info[k][-1][0]
        fp_orig_out_ch.append(old_sd[f'{last_bn_name}.weight'].shape[0])

    # ── SA modules: prune with input tracking ──
    for k in range(num_sa):
        # All scales in SA[k] share the same input: concat output of SA[k-1]
        sa_in = None if k == 0 else sa_concat_keep[k - 1]
        for s, bn_list in enumerate(sa_info[k]):
            _prune_conv_bn_relu_seq(
                old_sd, f'backbone_3d.SA_modules.{k}.mlps.{s}',
                bn_list, sa_in, new_sd,
                conv_dims=4, use_xyz_offset=3   # PointNet2 prepends 3 xyz channels
            )

    # ── FP modules: prune with [deeper|skip] input tracking ──
    # FP[k] in forward receives:
    #   deeper = SA[-1].out  if k == num_fp-1  else  FP[k+1].out
    #   skip   = raw(1ch)    if k == 0          else  SA[k-1].out
    # Input to mlp = torch.cat([interpolated_deeper, skip], dim=1)
    fp_out_keep = [None] * num_fp

    for k in range(num_fp - 1, -1, -1):
        # Deeper source
        if k == num_fp - 1:
            d_keep = sa_concat_keep[-1]
            d_orig = sa_orig_total[-1]
        else:
            d_keep = fp_out_keep[k + 1]
            d_orig = fp_orig_out_ch[k + 1]

        # Skip source
        s_orig = skip_ch_list[k]
        s_keep = sa_concat_keep[k - 1] if k > 0 else None  # k=0 → raw, all kept

        # Build concatenated input indices: [deeper | skip]
        if d_keep is not None and s_keep is not None:
            fp_in = np.concatenate([d_keep, s_keep + d_orig])
        elif d_keep is not None:
            # skip is raw features (all channels kept)
            fp_in = np.concatenate([d_keep, np.arange(s_orig) + d_orig])
        else:
            fp_in = None

        out = _prune_conv_bn_relu_seq(
            old_sd, f'backbone_3d.FP_modules.{k}.mlp',
            fp_info[k], fp_in, new_sd,
            conv_dims=4, use_xyz_offset=0
        )
        fp_out_keep[k] = out

    backbone_out = fp_out_keep[0]

    # ── Point head: cls_layers + box_layers ──
    ph_cls = keep_info['ph_cls']
    cls_out = _prune_linear_bn_relu_seq(
        old_sd, 'point_head.cls_layers', ph_cls, backbone_out, new_sd)
    _prune_final_layer(
        old_sd, f'point_head.cls_layers.{3 * len(ph_cls)}',
        cls_out, new_sd, is_conv=False)

    ph_reg = keep_info['ph_reg']
    reg_out = _prune_linear_bn_relu_seq(
        old_sd, 'point_head.box_layers', ph_reg, backbone_out, new_sd)
    _prune_final_layer(
        old_sd, f'point_head.box_layers.{3 * len(ph_reg)}',
        reg_out, new_sd, is_conv=False)

    # ── ROI head: SKIP ──
    # xyz_up_layer and merge_down_layer lack BN (USE_BN=False), making
    # gamma-based channel selection impossible. The ROI head is relatively
    # small and will re-initialize via xavier init + fine-tune quickly.
    # Keys NOT in new_sd will keep their model-initialized values when
    # loaded with strict=False.

    return new_sd


# ═══════════════════════════════════════════════════════════════════════════
# Build pruned config (yaml)
# ═══════════════════════════════════════════════════════════════════════════

def build_pruned_config(in_cfg, keep_info, R, M):
    out = in_cfg

    # ── Backbone SA ──
    sa_mlps = []
    for level_scales in keep_info['backbone_sa']:
        level = []
        for bn_list in level_scales:
            level.append([int(len(idx)) for _, idx in bn_list])
        sa_mlps.append(level)
    if sa_mlps:
        out['MODEL']['BACKBONE_3D']['SA_CONFIG']['MLPS'] = sa_mlps

    # ── Backbone FP ──
    fp_mlps = []
    for bn_list in keep_info['backbone_fp']:
        fp_mlps.append([int(len(idx)) for _, idx in bn_list])
    if fp_mlps:
        out['MODEL']['BACKBONE_3D']['FP_MLPS'] = fp_mlps

    # ── Point head ──
    ph_cls = [int(len(idx)) for _, idx in keep_info['ph_cls']]
    ph_reg = [int(len(idx)) for _, idx in keep_info['ph_reg']]
    if ph_cls:
        out['MODEL']['POINT_HEAD']['CLS_FC'] = ph_cls
    if ph_reg:
        out['MODEL']['POINT_HEAD']['REG_FC'] = ph_reg

    # ── ROI head XYZ_UP_LAYER ──
    fp_out = out['MODEL']['BACKBONE_3D']['FP_MLPS'][0][-1]
    if keep_info['roi_xyz']:
        xyz = [int(len(idx)) for _, idx in keep_info['roi_xyz']]
        xyz[-1] = int(fp_out)
        out['MODEL']['ROI_HEAD']['XYZ_UP_LAYER'] = xyz
    else:
        orig_xyz = in_cfg['MODEL']['ROI_HEAD']['XYZ_UP_LAYER']
        ratio = fp_out / orig_xyz[-1] if orig_xyz[-1] != 0 else 1.0
        new_xyz = [_round_ch(int(round(v * ratio)), R, M) for v in orig_xyz[:-1]]
        new_xyz.append(int(fp_out))
        out['MODEL']['ROI_HEAD']['XYZ_UP_LAYER'] = new_xyz

    # ── ROI head SA ──
    roi_sa = []
    for bn_list in keep_info['roi_sa']:
        roi_sa.append([int(len(idx)) for _, idx in bn_list])
    if roi_sa:
        out['MODEL']['ROI_HEAD']['SA_CONFIG']['MLPS'] = roi_sa

    # ── ROI head FC ──
    roi_cls = [int(len(idx)) for _, idx in keep_info['roi_cls']]
    roi_reg = [int(len(idx)) for _, idx in keep_info['roi_reg']]
    if roi_cls:
        out['MODEL']['ROI_HEAD']['CLS_FC'] = roi_cls
    if roi_reg:
        out['MODEL']['ROI_HEAD']['REG_FC'] = roi_reg

    # Disable slimming for fine-tuning
    if 'SLIMMING' in out.get('MODEL', {}):
        out['MODEL']['SLIMMING']['ENABLED'] = False

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def print_layer_diagnostics(gammas, threshold):
    print('\n╔══════════════════════════════════════════════════════════════════════╗')
    print('║                Per-Layer BN Gamma Diagnostics                       ║')
    print('╠══════════════════════════════════════════════════════════════════════╣')
    print(f'  {"Layer":<50s} {"Orig":>4s} {"Keep":>4s} {"Dead%":>6s} {"Mean":>8s} {"Min":>8s}')
    print(f'  {"─"*50} {"─"*4} {"─"*4} {"─"*6} {"─"*8} {"─"*8}')
    for name, gamma in gammas.items():
        n_ch = len(gamma)
        keep = int((gamma > threshold).sum())
        dead = (1 - keep / n_ch) * 100 if n_ch > 0 else 0
        print(f'  {name:<50s} {n_ch:>4d} {keep:>4d} {dead:>5.1f}% {gamma.mean():>8.5f} {gamma.min():>8.5f}')
    print('╚══════════════════════════════════════════════════════════════════════╝')


def print_channel_comparison(in_cfg, out_cfg):
    print('\n┌──────────────────────────────────────────────────────────────────┐')
    print('│            Original  →  Pruned  Channel Comparison              │')
    print('├──────────────────────────────────────────────────────────────────┤')

    def _flat(lst):
        if not isinstance(lst, list):
            return str(lst)
        if lst and isinstance(lst[0], list):
            return '[' + ', '.join(_flat(x) for x in lst) + ']'
        return str(lst)

    keys = [
        ('BACKBONE_3D.SA_CONFIG.MLPS', ['MODEL', 'BACKBONE_3D', 'SA_CONFIG', 'MLPS']),
        ('BACKBONE_3D.FP_MLPS',        ['MODEL', 'BACKBONE_3D', 'FP_MLPS']),
        ('POINT_HEAD.CLS_FC',           ['MODEL', 'POINT_HEAD', 'CLS_FC']),
        ('POINT_HEAD.REG_FC',           ['MODEL', 'POINT_HEAD', 'REG_FC']),
        ('ROI_HEAD.XYZ_UP_LAYER',       ['MODEL', 'ROI_HEAD', 'XYZ_UP_LAYER']),
        ('ROI_HEAD.SA_CONFIG.MLPS',     ['MODEL', 'ROI_HEAD', 'SA_CONFIG', 'MLPS']),
        ('ROI_HEAD.CLS_FC',             ['MODEL', 'ROI_HEAD', 'CLS_FC']),
        ('ROI_HEAD.REG_FC',             ['MODEL', 'ROI_HEAD', 'REG_FC']),
    ]
    for label, path in keys:
        orig, prun = in_cfg, out_cfg
        try:
            for p in path:
                orig = orig[p]
                prun = prun[p]
        except (KeyError, TypeError):
            continue
        print(f'  {label}:')
        print(f'    原始: {_flat(orig)}')
        print(f'    剪枝: {_flat(prun)}')
    print('└──────────────────────────────────────────────────────────────────┘')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.data_path is not None:
        cfg.DATA_CONFIG.DATA_PATH = args.data_path

    # Build model & load slimming checkpoint
    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=0, logger=None, training=False
    )
    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset
    )
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()

    # Collect BN gammas
    gammas = collect_bn_gammas(model)
    if not gammas:
        raise RuntimeError('No BN gamma found in model.')

    threshold, all_gamma = compute_global_threshold(gammas, args.prune_ratio)

    print('\n=== Slimming Pruning Export (Per-Layer Gamma + Weight Transfer) ===')
    print(f'Total BN params       : {len(all_gamma)}')
    print(f'Gamma range           : [{all_gamma.min():.6f}, {all_gamma.max():.6f}]')
    print(f'Gamma mean            : {all_gamma.mean():.6f}')
    print(f'Target prune ratio    : {args.prune_ratio}')
    print(f'Global threshold      : {threshold:.6f}')
    print(f'Actual dead channels  : {(all_gamma <= threshold).mean():.2%}')

    print_layer_diagnostics(gammas, threshold)

    # Extract per-layer keep indices
    keep_info = extract_all_keep_indices(
        gammas, threshold, args.round_to, args.min_channel)

    # Build pruned config
    in_cfg = _load_yaml(args.cfg_file)
    orig_cfg = copy.deepcopy(in_cfg)
    out_cfg = build_pruned_config(in_cfg, keep_info, args.round_to, args.min_channel)

    print_channel_comparison(orig_cfg, out_cfg)

    _save_yaml(out_cfg, args.output_cfg)
    print(f'\n✓ Pruned config saved to: {args.output_cfg}')

    # Build pruned checkpoint with weight transfer
    if args.output_ckpt:
        print('\n--- Building pruned checkpoint (weight transfer) ---')
        old_sd = ckpt['model_state']
        new_sd = build_pruned_state_dict(old_sd, keep_info)

        pruned_ckpt = {'model_state': new_sd}
        if 'version' in ckpt:
            pruned_ckpt['version'] = ckpt['version']

        ckpt_path = Path(args.output_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pruned_ckpt, str(ckpt_path))
        print(f'✓ Pruned checkpoint saved to: {args.output_ckpt}')

        # Stats
        transferred_params = sum(v.numel() for v in new_sd.values())
        total_orig_params = sum(v.numel() for v in old_sd.values())
        roi_params = sum(v.numel() for k, v in old_sd.items() if k.startswith('roi_head.'))
        print(f'  迁移参数量: {transferred_params:,}')
        print(f'  原始总参数: {total_orig_params:,}')
        print(f'  ROI head (将重新初始化): {roi_params:,}')
        print(f'  迁移覆盖率: {transferred_params / total_orig_params:.1%} (不含ROI head)')
        print(f'\n  ⚠ ROI head 权重未迁移 (USE_BN=False → xyz_up/merge_down 无BN)')
        print(f'  ⚠ 加载时请使用 strict=False, ROI head 将使用 xavier 初始化')
    else:
        print('\n(跳过权重迁移，未指定 --output_ckpt，重训练将从头开始)')


if __name__ == '__main__':
    main()
