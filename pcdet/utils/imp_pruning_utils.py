from pathlib import Path

import torch
import torch.nn.utils.prune as prune


def _module_type_match(module, module_types):
    return module.__class__.__name__ in module_types


def _name_match(name, include_keywords, exclude_keywords):
    if include_keywords and not any(k in name for k in include_keywords):
        return False
    if exclude_keywords and any(k in name for k in exclude_keywords):
        return False
    return True


def _resolve_path(path, cfg):
    path = Path(path)
    if path.exists():
        return path
    candidate = cfg.ROOT_DIR / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f'IMP mask checkpoint not found: {path}')


def _collect_target_modules(model, module_types, include_keywords, exclude_keywords, param_name):
    targets = []
    for name, module in model.named_modules():
        if not _module_type_match(module, module_types):
            continue
        if not _name_match(name, include_keywords, exclude_keywords):
            continue
        if not hasattr(module, param_name):
            continue
        targets.append((name, module))
    return targets


def apply_imp_pruning_from_cfg(cfg, model, logger=None, stage='train'):
    imp_cfg = cfg.MODEL.get('IMP_PRUNING', None)
    if imp_cfg is None or not bool(imp_cfg.get('ENABLED', False)):
        return 0

    if stage == 'test' and not bool(imp_cfg.get('APPLY_ON_TEST', True)):
        return 0

    mask_ckpt = str(imp_cfg.get('MASK_CKPT', '')).strip()
    if mask_ckpt == '':
        raise ValueError('MODEL.IMP_PRUNING.MASK_CKPT must be set when IMP_PRUNING is enabled.')

    module_types = imp_cfg.get('MODULE_TYPES', ['Conv1d', 'Conv2d', 'Linear'])
    include_keywords = imp_cfg.get('INCLUDE_KEYWORDS', [])
    exclude_keywords = imp_cfg.get('EXCLUDE_KEYWORDS', [])
    param_name = str(imp_cfg.get('PARAM_NAME', 'weight'))

    mask_path = _resolve_path(mask_ckpt, cfg)
    checkpoint = torch.load(mask_path, map_location='cpu')
    model_state = checkpoint.get('model_state', checkpoint)

    targets = _collect_target_modules(
        model, module_types, include_keywords, exclude_keywords, param_name
    )
    if len(targets) == 0:
        raise RuntimeError('No module matched IMP_PRUNING config; please check MODULE_TYPES/keywords.')

    zero_cnt = 0
    total_cnt = 0
    applied = 0
    missing_masks = []
    for name, module in targets:
        mask_key = f'{name}.{param_name}_mask'
        if mask_key not in model_state:
            missing_masks.append(mask_key)
            continue
        mask = model_state[mask_key].detach().cpu()
        prune.custom_from_mask(module, name=param_name, mask=mask)
        zero_cnt += int((mask == 0).sum().item())
        total_cnt += int(mask.numel())
        applied += 1

    if missing_masks:
        preview = ', '.join(missing_masks[:5])
        raise KeyError(f'IMP mask checkpoint is missing {len(missing_masks)} masks, e.g. {preview}')

    sparsity = zero_cnt / max(total_cnt, 1)
    msg = (
        f'[IMP] stage={stage}, mask_ckpt={mask_path}, '
        f'target_modules={applied}, mask_sparsity={sparsity:.4f}'
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
    return applied
