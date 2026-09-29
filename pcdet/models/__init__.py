from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector
from ..utils.slimming_utils import slimming_l1_loss

try:
    import kornia
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key == 'camera_imgs':
            batch_dict[key] = val.cuda()
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        core_model = model.module if hasattr(model, 'module') else model
        slimming_cfg = core_model.model_cfg.get('SLIMMING', None)
        if slimming_cfg is not None and bool(slimming_cfg.get('ENABLED', False)):
            lam = float(slimming_cfg.get('LAMBDA', 1e-5))
            module_types = tuple(slimming_cfg.get('MODULE_TYPES', ['BatchNorm1d', 'BatchNorm2d']))
            include_keywords = slimming_cfg.get('INCLUDE_KEYWORDS', [])
            exclude_keywords = slimming_cfg.get('EXCLUDE_KEYWORDS', [])
            slim_l1, bn_cnt = slimming_l1_loss(
                model,
                module_types=module_types,
                include_keywords=include_keywords,
                exclude_keywords=exclude_keywords
            )
            loss = loss + lam * slim_l1
            tb_dict['slimming_l1'] = float(slim_l1.item())
            tb_dict['slimming_lambda'] = lam
            tb_dict['slimming_bn_params'] = int(bn_cnt)

        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
