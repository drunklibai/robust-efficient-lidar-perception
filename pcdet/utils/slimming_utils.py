import torch
import torch.nn as nn


def get_core_model(model):
    return model.module if hasattr(model, 'module') else model


def slimming_l1_loss(model, module_types=('BatchNorm1d', 'BatchNorm2d'),
                     include_keywords=None, exclude_keywords=None):
    include_keywords = include_keywords or []
    exclude_keywords = exclude_keywords or []

    core_model = get_core_model(model)
    loss = None
    cnt = 0
    for name, module in core_model.named_modules():
        if module.__class__.__name__ not in module_types:
            continue
        if include_keywords and not any(k in name for k in include_keywords):
            continue
        if exclude_keywords and any(k in name for k in exclude_keywords):
            continue
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        cur = module.weight.abs().sum()
        loss = cur if loss is None else loss + cur
        cnt += int(module.weight.numel())

    if loss is None:
        dev = next(core_model.parameters()).device
        loss = torch.zeros((), device=dev)
    return loss, cnt
