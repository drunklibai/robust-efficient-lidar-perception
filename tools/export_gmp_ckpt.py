import _init_path
import argparse
from collections import OrderedDict
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network


def parse_args():
    parser = argparse.ArgumentParser(description='Export post-training GMP checkpoint')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--trained_ckpt', type=str, required=True)
    parser.add_argument('--output_ckpt', type=str, required=True)
    parser.add_argument('--sparsity', type=float, default=0.5)
    return parser.parse_args()


def _module_type_match(module, module_types):
    return module.__class__.__name__ in module_types


def _name_match(name, include_keywords, exclude_keywords):
    if include_keywords and not any(k in name for k in include_keywords):
        return False
    if exclude_keywords and any(k in name for k in exclude_keywords):
        return False
    return True


def collect_target_weight_keys(model, gmp_cfg):
    module_types = gmp_cfg.get('MODULE_TYPES', ['Conv1d', 'Conv2d', 'Linear'])
    include_keywords = gmp_cfg.get('INCLUDE_KEYWORDS', [])
    exclude_keywords = gmp_cfg.get('EXCLUDE_KEYWORDS', [])
    param_name = str(gmp_cfg.get('PARAM_NAME', 'weight'))

    keys = []
    for name, module in model.named_modules():
        if not _module_type_match(module, module_types):
            continue
        if not _name_match(name, include_keywords, exclude_keywords):
            continue
        if not hasattr(module, param_name):
            continue
        keys.append(f'{name}.{param_name}')
    if not keys:
        raise RuntimeError('No module matched GMP_PRUNING config; please check MODULE_TYPES/keywords.')
    return keys


def build_global_masks(trained_state, target_keys, sparsity):
    if sparsity <= 0 or sparsity >= 1:
        raise ValueError(f'--sparsity must be in (0, 1), got {sparsity}')

    scores = []
    shapes = []
    for key in target_keys:
        if key not in trained_state:
            raise KeyError(f'Trained checkpoint missing target weight: {key}')
        weight = trained_state[key].detach().cpu()
        scores.append(weight.abs().reshape(-1))
        shapes.append(weight.shape)

    flat_scores = torch.cat(scores, dim=0)
    total = flat_scores.numel()
    prune_count = int(total * sparsity)
    keep_count = max(total - prune_count, 1)

    keep_indices = torch.topk(flat_scores, k=keep_count, largest=True, sorted=False).indices
    flat_mask = torch.zeros(total, dtype=torch.float32)
    flat_mask[keep_indices] = 1.0

    masks = OrderedDict()
    offset = 0
    for key, shape in zip(target_keys, shapes):
        numel = int(torch.tensor(shape).prod().item())
        masks[key] = flat_mask[offset:offset + numel].reshape(shape)
        offset += numel
    return masks


def build_gmp_state(trained_state, target_keys, masks):
    out_state = OrderedDict()
    target_set = set(target_keys)

    for key, val in trained_state.items():
        if key in target_set:
            continue
        out_state[key] = val.detach().cpu() if torch.is_tensor(val) else val

    for key in target_keys:
        out_state[f'{key}_orig'] = trained_state[key].detach().cpu()
        out_state[f'{key}_mask'] = masks[key].detach().cpu()
    return out_state


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    gmp_cfg = cfg.MODEL.get('GMP_PRUNING', None)
    if gmp_cfg is None:
        raise ValueError('MODEL.GMP_PRUNING is required in cfg_file.')

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=None,
        training=False
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    target_keys = collect_target_weight_keys(model, gmp_cfg)

    trained_ckpt = torch.load(args.trained_ckpt, map_location='cpu')
    trained_state = trained_ckpt['model_state']
    masks = build_global_masks(trained_state, target_keys, args.sparsity)
    out_state = build_gmp_state(trained_state, target_keys, masks)

    zero_cnt = sum(int((m == 0).sum().item()) for m in masks.values())
    total_cnt = sum(int(m.numel()) for m in masks.values())
    actual_sparsity = zero_cnt / max(total_cnt, 1)

    out_ckpt = {
        'model_state': out_state,
        'gmp_meta': {
            'trained_ckpt': args.trained_ckpt,
            'target_sparsity': args.sparsity,
            'actual_sparsity': actual_sparsity,
            'target_modules': len(target_keys),
            'target_weights': total_cnt,
        }
    }
    if 'version' in trained_ckpt:
        out_ckpt['version'] = trained_ckpt['version']

    output_path = Path(args.output_ckpt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(output_path))

    print('=== GMP checkpoint export ===')
    print(f'Target modules  : {len(target_keys)}')
    print(f'Target weights  : {total_cnt}')
    print(f'Target sparsity : {args.sparsity:.4f}')
    print(f'Actual sparsity : {actual_sparsity:.4f}')
    print(f'Saved to        : {output_path}')


if __name__ == '__main__':
    main()
