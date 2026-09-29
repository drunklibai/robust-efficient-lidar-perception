from pathlib import Path

import yaml
from easydict import EasyDict


def _to_easydict(obj):
    if isinstance(obj, dict):
        return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_easydict(v) for v in obj]
    return obj


def _deep_update(dst, src):
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key, None), (dict, EasyDict)):
            _deep_update(dst[key], val)
        else:
            dst[key] = _to_easydict(val)


def apply_sampling_policy_from_cfg(cfg, logger=None):
    policy_path = cfg.MODEL.get('SAMPLING_POLICY_PATH', None)
    if policy_path is None or str(policy_path).strip() == '':
        return False

    policy_path = Path(policy_path)
    if not policy_path.exists():
        candidate = cfg.ROOT_DIR / policy_path
        if candidate.exists():
            policy_path = candidate
        else:
            raise FileNotFoundError(f'Sampling policy file not found: {policy_path}')

    with open(policy_path, 'r') as f:
        policy_cfg = yaml.safe_load(f) or {}

    has_any_override = False
    if 'MODEL' in policy_cfg:
        model_override = policy_cfg.get('MODEL', {})
        if not isinstance(model_override, dict):
            raise ValueError(f'Invalid MODEL policy format in {policy_path}')
        _deep_update(cfg.MODEL, model_override)
        has_any_override = True

    if 'DATA_CONFIG' in policy_cfg:
        data_override = policy_cfg.get('DATA_CONFIG', {})
        if not isinstance(data_override, dict):
            raise ValueError(f'Invalid DATA_CONFIG policy format in {policy_path}')
        _deep_update(cfg.DATA_CONFIG, data_override)
        has_any_override = True

    if not has_any_override:
        # Backward-compatible: treat top-level keys as MODEL overrides.
        if not isinstance(policy_cfg, dict):
            raise ValueError(f'Invalid sampling policy format in {policy_path}')
        _deep_update(cfg.MODEL, policy_cfg)

    msg = f'[SamplingPolicy] applied from: {policy_path}'
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
    return True
