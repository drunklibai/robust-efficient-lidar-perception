import torch.nn as nn
import torch.nn.utils.prune as prune
from pathlib import Path

import torch


def _norm_type(method_name):
    method_name = str(method_name).lower()
    if method_name in ['l1', 'l1_unstructured']:
        return 'l1_unstructured'
    if method_name in ['random', 'random_unstructured']:
        return 'random_unstructured'
    raise ValueError(f'Unsupported GMP method: {method_name}')


def _module_type_match(module, module_types):
    module_type_name = module.__class__.__name__
    return module_type_name in module_types


def _name_match(name, include_keywords, exclude_keywords):
    if include_keywords:
        ok = any(k in name for k in include_keywords)
        if not ok:
            return False
    if exclude_keywords:
        bad = any(k in name for k in exclude_keywords)
        if bad:
            return False
    return True


def _resolve_mask_path(path, cfg):
    path = Path(path)
    if path.exists():
        return path
    candidate = cfg.ROOT_DIR / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f'GMP mask checkpoint not found: {path}')


def apply_gmp_pruning_from_cfg(cfg, model, logger=None, stage='train'):
    gmp_cfg = cfg.MODEL.get('GMP_PRUNING', None)
    if gmp_cfg is None or not bool(gmp_cfg.get('ENABLED', False)):
        return 0

    if stage == 'test' and not bool(gmp_cfg.get('APPLY_ON_TEST', True)):
        return 0

    method = _norm_type(gmp_cfg.get('METHOD', 'l1_unstructured'))
    amount = float(gmp_cfg.get('AMOUNT', 0.5))
    if amount <= 0 or amount >= 1:
        raise ValueError(f'GMP amount should be in (0, 1), got {amount}')
    scope = str(gmp_cfg.get('SCOPE', 'global')).lower()
    if scope not in ['global', 'layerwise']:
        raise ValueError(f"GMP SCOPE should be 'global' or 'layerwise', got {scope}")

    module_types = gmp_cfg.get('MODULE_TYPES', ['Conv1d', 'Conv2d', 'Linear'])
    include_keywords = gmp_cfg.get('INCLUDE_KEYWORDS', [])
    exclude_keywords = gmp_cfg.get('EXCLUDE_KEYWORDS', [])
    param_name = str(gmp_cfg.get('PARAM_NAME', 'weight'))

    parameters_to_prune = []
    target_names = []
    for name, module in model.named_modules():
        if not _module_type_match(module, module_types):
            continue
        if not _name_match(name, include_keywords, exclude_keywords):
            continue
        if not hasattr(module, param_name):
            continue
        parameters_to_prune.append((module, param_name))
        target_names.append(name)

    if len(parameters_to_prune) == 0:
        raise RuntimeError('No module matched GMP_PRUNING config; please check MODULE_TYPES/keywords.')

    mask_ckpt = str(gmp_cfg.get('MASK_CKPT', '')).strip()
    if mask_ckpt != '':
        mask_path = _resolve_mask_path(mask_ckpt, cfg)
        checkpoint = torch.load(mask_path, map_location='cpu')
        model_state = checkpoint.get('model_state', checkpoint)

        missing_masks = []
        zero_cnt = 0
        total_cnt = 0
        for target_name, (module, pname) in zip(target_names, parameters_to_prune):
            mask_key = f'{target_name}.{pname}_mask'
            if mask_key not in model_state:
                missing_masks.append(mask_key)
                continue
            mask = model_state[mask_key].detach().cpu()
            prune.custom_from_mask(module, name=pname, mask=mask)
            total_cnt += int(mask.numel())
            zero_cnt += int((mask == 0).sum().item())

        if missing_masks:
            preview = ', '.join(missing_masks[:5])
            raise KeyError(f'GMP mask checkpoint is missing {len(missing_masks)} masks, e.g. {preview}')

        sparsity = zero_cnt / max(total_cnt, 1)
        msg = (
            f'[GMP] stage={stage}, method=fixed_mask, mask_ckpt={mask_path}, '
            f'target_modules={len(parameters_to_prune)}, '
            f'mask_sparsity={sparsity:.4f}'
        )
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
        return len(parameters_to_prune)

    if scope == 'global':
        method_obj = prune.L1Unstructured if method == 'l1_unstructured' else prune.RandomUnstructured
        try:
            # Newer PyTorch APIs
            prune.global_unstructured(
                parameters=parameters_to_prune,
                pruning_method=method_obj,
                amount=amount
            )
        except TypeError:
            # Older PyTorch APIs
            prune.global_unstructured(
                parameters_to_prune=parameters_to_prune,
                pruning_method=method_obj,
                amount=amount
            )
    else:
        for module, pname in parameters_to_prune:
            if method == 'l1_unstructured':
                prune.l1_unstructured(module, name=pname, amount=amount)
            else:
                prune.random_unstructured(module, name=pname, amount=amount)

    zero_cnt = 0
    total_cnt = 0
    for module, pname in parameters_to_prune:
        mask = getattr(module, f'{pname}_mask')
        total_cnt += int(mask.numel())
        zero_cnt += int((mask == 0).sum().item())
    sparsity = zero_cnt / max(total_cnt, 1)

    msg = (
        f'[GMP] stage={stage}, method={method}, scope={scope}, amount={amount}, '
        f'target_modules={len(parameters_to_prune)}, '
        f'mask_sparsity={sparsity:.4f}'
    )
    if logger is not None:
        logger.info(msg)
        if bool(gmp_cfg.get('VERBOSE', True)):
            for n in target_names:
                logger.info(f'[GMP] target: {n}.{param_name}')
    else:
        print(msg)
        if bool(gmp_cfg.get('VERBOSE', True)):
            for n in target_names:
                print(f'[GMP] target: {n}.{param_name}')

    return len(parameters_to_prune)
