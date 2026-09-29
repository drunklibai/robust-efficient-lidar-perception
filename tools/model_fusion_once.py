import argparse
import copy
import importlib
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

import _init_path  # noqa: F401
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.ops.iou3d_nms import iou3d_nms_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Majority-vote box fusion for ONCE result.pkl files')
    parser.add_argument('--inputs', nargs='+', default=None, help='Input result.pkl files from different models')
    parser.add_argument('--load_fused', type=str, default=None, help='Evaluate an existing fused result.pkl without fusing inputs')
    parser.add_argument('--model_names', nargs='+', default=None, help='Optional model names, same length as inputs')
    parser.add_argument('--output', type=str, default=None, help='Output fused result.pkl')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='BEV IoU threshold for grouping boxes')
    parser.add_argument('--vote_threshold', type=float, default=0.5, help='Fraction of models required to keep a group')
    parser.add_argument('--nms_threshold', type=float, default=0.1, help='Final class-wise BEV NMS threshold')
    parser.add_argument('--score_threshold', type=float, default=0.0, help='Drop input boxes below this score')
    parser.add_argument('--mode', choices=['weighted_average', 'average', 'best'], default='weighted_average')
    parser.add_argument(
        '--iou_backend', choices=['cpu', 'cuda', 'axis_aligned'], default='cpu',
        help='BEV IoU backend. Use cpu by default to avoid incompatible CUDA extension builds.'
    )
    parser.add_argument('--cfg_file', type=str, default=None, help='ONCE model cfg used to build dataset for evaluation')
    parser.add_argument('--eval_split', type=str, default='val', help='ONCE split for evaluation, e.g. train, train_cor, val, val_cor')
    parser.add_argument('--output_dir', type=str, default='output/once_fusion', help='Directory for fusion evaluation outputs')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers when building ONCE dataset')
    parser.add_argument('--save_eval', action='store_true', default=False, help='Run ONCE AP evaluation and save eval files')
    args = parser.parse_args()

    if args.load_fused is None and not args.inputs:
        parser.error('Either --inputs or --load_fused must be specified')
    if args.load_fused is None and args.output is None:
        parser.error('--output is required when fusing --inputs')
    if args.save_eval and args.cfg_file is None:
        parser.error('--cfg_file is required when --save_eval is set')
    return args


def load_detections(paths):
    detections = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, 'rb') as f:
            detections.append(pickle.load(f))

    lengths = [len(x) for x in detections]
    if len(set(lengths)) != 1:
        raise ValueError('Input result.pkl files have different lengths: %s' % lengths)
    return detections


def to_numpy(x, dtype=None):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def normalize_detection(det, score_threshold=0.0):
    boxes = to_numpy(det.get('boxes_lidar', det.get('boxes_3d', None)), dtype=np.float32)
    scores = to_numpy(det.get('score', None), dtype=np.float32)
    names = to_numpy(det.get('name', None))

    if boxes is None or scores is None or names is None:
        return {
            'boxes': np.zeros((0, 7), dtype=np.float32),
            'scores': np.zeros((0,), dtype=np.float32),
            'names': np.zeros((0,), dtype=object),
        }

    if boxes.ndim == 1 and boxes.shape[0] == 7:
        boxes = boxes.reshape(1, 7)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        boxes = np.zeros((0, 7), dtype=np.float32)

    scores = scores.reshape(-1)
    names = names.reshape(-1)
    n = min(len(boxes), len(scores), len(names))
    boxes, scores, names = boxes[:n], scores[:n], names[:n]

    def decode_name(name):
        if isinstance(name, bytes):
            return name.decode('utf-8', errors='ignore')
        return str(name)

    names = np.array([decode_name(x) for x in names], dtype=object)
    valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & ((boxes[:, 3:6] > 1e-6).all(axis=1))
    if score_threshold > 0:
        valid &= scores >= score_threshold

    return {
        'boxes': boxes[valid].astype(np.float32),
        'scores': scores[valid].astype(np.float32),
        'names': names[valid],
    }


def boxes_iou_bev_axis_aligned_np(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    xa_min = boxes_a[:, 0] - boxes_a[:, 3] / 2
    xa_max = boxes_a[:, 0] + boxes_a[:, 3] / 2
    ya_min = boxes_a[:, 1] - boxes_a[:, 4] / 2
    ya_max = boxes_a[:, 1] + boxes_a[:, 4] / 2

    xb_min = boxes_b[:, 0] - boxes_b[:, 3] / 2
    xb_max = boxes_b[:, 0] + boxes_b[:, 3] / 2
    yb_min = boxes_b[:, 1] - boxes_b[:, 4] / 2
    yb_max = boxes_b[:, 1] + boxes_b[:, 4] / 2

    inter_x = np.maximum(0.0, np.minimum(xa_max[:, None], xb_max[None, :]) - np.maximum(xa_min[:, None], xb_min[None, :]))
    inter_y = np.maximum(0.0, np.minimum(ya_max[:, None], yb_max[None, :]) - np.maximum(ya_min[:, None], yb_min[None, :]))
    inter = inter_x * inter_y
    area_a = np.maximum((xa_max - xa_min) * (ya_max - ya_min), 1e-6)
    area_b = np.maximum((xb_max - xb_min) * (yb_max - yb_min), 1e-6)
    return (inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)).astype(np.float32)


def boxes_iou_bev_np(boxes_a, boxes_b, backend='cpu'):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    if backend == 'axis_aligned':
        return boxes_iou_bev_axis_aligned_np(boxes_a, boxes_b)

    try:
        if backend == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is not available')
            ta = torch.from_numpy(boxes_a.astype(np.float32)).cuda()
            tb = torch.from_numpy(boxes_b.astype(np.float32)).cuda()
            with torch.no_grad():
                iou = iou3d_nms_utils.boxes_iou_bev(ta, tb)
            iou = iou.cpu().numpy()
        else:
            ta = torch.from_numpy(boxes_a.astype(np.float32))
            tb = torch.from_numpy(boxes_b.astype(np.float32))
            with torch.no_grad():
                iou = iou3d_nms_utils.boxes_bev_iou_cpu(ta, tb)
            iou = iou.numpy() if torch.is_tensor(iou) else iou
    except Exception as err:
        raise RuntimeError('Rotated BEV IoU failed; rebuild OpenPCDet ops for this environment') from err

    return np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)


def group_boxes(class_boxes_list, iou_threshold, iou_backend='cpu'):
    all_boxes = []
    all_sources = []
    all_indices = []
    for source_idx, boxes in enumerate(class_boxes_list):
        for box_idx, box in enumerate(boxes):
            all_boxes.append(box)
            all_sources.append(source_idx)
            all_indices.append(box_idx)

    if len(all_boxes) == 0:
        return []

    all_boxes = np.asarray(all_boxes, dtype=np.float32)
    all_sources = np.asarray(all_sources, dtype=np.int64)
    all_indices = np.asarray(all_indices, dtype=np.int64)
    ious = boxes_iou_bev_np(all_boxes, all_boxes, backend=iou_backend)

    order = np.arange(len(all_boxes))
    used = np.zeros(len(all_boxes), dtype=bool)
    groups = []

    for idx in order:
        if used[idx]:
            continue

        candidate = np.where((ious[idx] >= iou_threshold) & (~used))[0]
        selected = []
        selected_sources = set()

        # Keep at most one box from each model in a group.
        for cand in candidate:
            src = int(all_sources[cand])
            if src in selected_sources:
                continue
            selected.append(cand)
            selected_sources.add(src)

        if not selected:
            selected = [idx]
            selected_sources = {int(all_sources[idx])}

        used[selected] = True
        selected = np.asarray(selected, dtype=np.int64)
        groups.append({
            'boxes': all_boxes[selected],
            'sources': all_sources[selected],
            'indices': all_indices[selected],
        })

    return groups


def fuse_group(group_boxes_arr, group_scores, mode):
    if mode == 'best':
        best = int(np.argmax(group_scores))
        return group_boxes_arr[best].copy(), float(group_scores[best])

    if mode == 'average':
        weights = np.ones_like(group_scores, dtype=np.float32) / max(len(group_scores), 1)
    else:
        weights = group_scores.astype(np.float32) + 1e-6
        weights = weights / max(float(weights.sum()), 1e-6)

    centers = group_boxes_arr[:, :3]
    sizes = group_boxes_arr[:, 3:6]
    headings = group_boxes_arr[:, 6]
    center = (centers * weights[:, None]).sum(axis=0)
    size = (sizes * weights[:, None]).sum(axis=0)
    sin_sum = (np.sin(headings) * weights).sum()
    cos_sum = (np.cos(headings) * weights).sum()
    heading = np.arctan2(sin_sum, cos_sum)
    box = np.concatenate([center, size, [heading]]).astype(np.float32)
    return box, float(group_scores.max())


def classwise_nms(boxes, scores, names, nms_threshold, iou_backend='cpu'):
    if len(boxes) == 0:
        return boxes, scores, names

    keep_all = []
    for cls in sorted(set(names.tolist())):
        cls_inds = np.where(names == cls)[0]
        cls_boxes = boxes[cls_inds]
        cls_scores = scores[cls_inds]
        order = np.argsort(-cls_scores)
        cls_keep = []

        while len(order) > 0:
            cur = order[0]
            cls_keep.append(cur)
            if len(order) == 1:
                break
            rest = order[1:]
            ious = boxes_iou_bev_np(cls_boxes[cur:cur + 1], cls_boxes[rest], backend=iou_backend).reshape(-1)
            order = rest[ious <= nms_threshold]

        keep_all.extend(cls_inds[np.asarray(cls_keep, dtype=np.int64)].tolist())

    keep_all = np.asarray(keep_all, dtype=np.int64)
    order = np.argsort(-scores[keep_all])
    keep_all = keep_all[order]
    return boxes[keep_all], scores[keep_all], names[keep_all]


def fuse_frame(frame_dets, num_models, args):
    normalized = [normalize_detection(det, score_threshold=args.score_threshold) for det in frame_dets]
    class_names = sorted(set(name for det in normalized for name in det['names'].tolist()))

    fused_boxes = []
    fused_scores = []
    fused_names = []

    min_votes = int(np.ceil(args.vote_threshold * num_models))
    min_votes = max(1, min_votes)

    for class_name in class_names:
        class_boxes_list = []
        class_scores_list = []
        for det in normalized:
            mask = det['names'] == class_name
            class_boxes_list.append(det['boxes'][mask])
            class_scores_list.append(det['scores'][mask])

        groups = group_boxes(class_boxes_list, args.iou_threshold, args.iou_backend)
        for group in groups:
            unique_sources = set(int(x) for x in group['sources'])
            if len(unique_sources) < min_votes:
                continue

            group_scores = []
            for source_idx, box_idx in zip(group['sources'], group['indices']):
                group_scores.append(class_scores_list[int(source_idx)][int(box_idx)])
            group_scores = np.asarray(group_scores, dtype=np.float32)
            box, score = fuse_group(group['boxes'], group_scores, args.mode)

            fused_boxes.append(box)
            fused_scores.append(score)
            fused_names.append(class_name)

    if len(fused_boxes) == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=object),
        )

    boxes = np.asarray(fused_boxes, dtype=np.float32)
    scores = np.asarray(fused_scores, dtype=np.float32)
    names = np.asarray(fused_names, dtype=object)
    return classwise_nms(boxes, scores, names, args.nms_threshold, args.iou_backend)


def make_output_det(template_det, boxes, scores, names):
    out = {}
    frame_id = template_det.get('frame_id', None)
    if frame_id is not None:
        out['frame_id'] = frame_id

    out['name'] = names
    out['score'] = scores
    out['boxes_lidar'] = boxes
    out['boxes_3d'] = boxes

    # Keep ONCE result.pkl compact. These placeholder fields preserve compatibility
    # with utilities that expect OpenPCDet detection dictionaries.
    n = len(names)
    out['bbox'] = np.zeros((n, 4), dtype=np.float32)
    out['location'] = boxes[:, :3].copy() if n > 0 else np.zeros((0, 3), dtype=np.float32)
    out['dimensions'] = boxes[:, 3:6].copy() if n > 0 else np.zeros((0, 3), dtype=np.float32)
    out['rotation_y'] = boxes[:, 6].copy() if n > 0 else np.zeros((0,), dtype=np.float32)
    return out


def normalize_for_once_eval(det):
    boxes = to_numpy(det.get('boxes_3d', det.get('boxes_lidar', None)), dtype=np.float32)
    scores = to_numpy(det.get('score', None), dtype=np.float32)
    names = to_numpy(det.get('name', None))

    if boxes is None or scores is None or names is None:
        boxes = np.zeros((0, 7), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        names = np.zeros((0,), dtype=object)

    if boxes.ndim == 1 and boxes.shape[0] == 7:
        boxes = boxes.reshape(1, 7)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        boxes = np.zeros((0, 7), dtype=np.float32)

    scores = scores.reshape(-1)
    names = names.reshape(-1)
    n = min(len(boxes), len(scores), len(names))
    boxes = boxes[:n].astype(np.float32)
    scores = scores[:n].astype(np.float32)
    names = np.array([x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in names[:n]], dtype=object)

    eval_det = dict(det)
    eval_det['name'] = names
    eval_det['score'] = scores
    eval_det['boxes_3d'] = boxes
    eval_det['boxes_lidar'] = boxes
    return eval_det


SPLIT_TO_INFO_SPLIT = {
    'train': 'train',
    'train_cor': 'train',
    'val': 'val',
    'val_cor': 'val',
    'test': 'test',
    'test_cor': 'test',
    'train_mixed': 'train',
    'train_mixed_nosel': 'train',
    'train_mixed_sorted_uncertainty': 'train',
    'train_mixed_sorted_uncertainty_fixed': 'train',
    'train_mixed_sorted_safety_aware': 'train',
}


def get_info_split(split):
    if split in SPLIT_TO_INFO_SPLIT:
        return SPLIT_TO_INFO_SPLIT[split]
    if split.endswith('_cor'):
        return split[:-4]
    if split.startswith('train_mixed'):
        return 'train'
    return split


def resolve_once_root(data_path):
    path = Path(data_path)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parents[1] / path,
    ]
    path_str = str(path).replace('\\', '/')
    while path_str.startswith('../'):
        path_str = path_str[3:]
        candidates.append(Path.cwd() / path_str)
        candidates.append(Path(__file__).resolve().parents[1] / path_str)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def parse_once_split(root_path, split):
    split_file = root_path / 'ImageSets' / (split + '.txt')
    default_domain = 'cor' if split.endswith('_cor') else None
    if not split_file.exists() and split.endswith('_cor'):
        base_split_file = root_path / 'ImageSets' / (split[:-4] + '.txt')
        if base_split_file.exists():
            split_file = base_split_file

    if not split_file.exists():
        return None

    split_items = []
    for line in split_file.read_text().splitlines():
        tokens = line.strip().split()
        if len(tokens) == 0:
            continue
        seq_id = tokens[0]
        domain = tokens[1] if len(tokens) > 1 else default_domain
        if domain not in [None, 'ori', 'cor']:
            raise ValueError('Unsupported ONCE domain %s in %s' % (domain, split_file))
        split_items.append((seq_id, domain))
    return split_items


def expand_once_infos_by_split(infos, split_items, split, info_split):
    if split_items is None:
        return infos

    infos_by_seq = {}
    for info in infos:
        infos_by_seq.setdefault(info['sequence_id'], []).append(info)

    expanded_infos = []
    missing_seqs = []
    for seq_id, domain in split_items:
        seq_infos = infos_by_seq.get(seq_id, None)
        if seq_infos is None:
            missing_seqs.append(seq_id)
            continue
        for base_info in seq_infos:
            info = copy.deepcopy(base_info)
            if domain is not None:
                info['domain'] = domain
            expanded_infos.append(info)

    if len(missing_seqs) > 0:
        missing_seqs = sorted(set(missing_seqs))
        raise RuntimeError(
            'ONCE split %s references %d sequences missing from %s infos, examples: %s'
            % (split, len(missing_seqs), info_split, missing_seqs[:5])
        )

    return expanded_infos


def import_once_eval():
    repo_root = Path(__file__).resolve().parents[1]
    package_paths = {
        'pcdet.datasets': repo_root / 'pcdet' / 'datasets',
        'pcdet.datasets.once': repo_root / 'pcdet' / 'datasets' / 'once',
        'pcdet.datasets.once.once_eval': repo_root / 'pcdet' / 'datasets' / 'once' / 'once_eval',
    }
    for name, package_path in package_paths.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(package_path)]
            sys.modules[name] = module
    return importlib.import_module('pcdet.datasets.once.once_eval.evaluation').get_evaluation_results


class OnceEvalDataset:
    def __init__(self, once_infos, dataset_cfg=None):
        self.once_infos = once_infos
        self.dataset_cfg = dataset_cfg

    def evaluation(self, det_annos, class_names, **kwargs):
        get_evaluation_results = import_once_eval()
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.once_infos]

        eval_class_names = class_names
        eval_kwargs = {}
        once_eval_cfg = self.dataset_cfg.get('ONCE_EVAL', None) if self.dataset_cfg is not None else None
        if once_eval_cfg is not None:
            mode = once_eval_cfg.get('MODE', 'official')
            if mode == 'kitti_car':
                eval_class_names = ['Car']
                eval_kwargs.update({
                    'use_superclass': False,
                    'num_pr_points': 40,
                    'ap_name': 'AP_R40'
                })
            elif mode == 'custom':
                eval_classes = once_eval_cfg.get('EVAL_CLASSES', None)
                if eval_classes is not None:
                    eval_class_names = list(eval_classes)
                eval_kwargs.update({
                    'use_superclass': once_eval_cfg.get('USE_SUPERCLASS', True),
                    'num_pr_points': once_eval_cfg.get('NUM_PR_POINTS', 50),
                    'ap_name': once_eval_cfg.get('AP_NAME', None)
                })
            elif mode != 'official':
                raise ValueError('Unsupported ONCE_EVAL.MODE: %s' % mode)

        return get_evaluation_results(eval_gt_annos, eval_det_annos, eval_class_names, **eval_kwargs)


def cfg_from_yaml_file_robust(cfg_file, config):
    cfg_path = Path(cfg_file)
    try:
        return cfg_from_yaml_file(str(cfg_path), config)
    except FileNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    if not cfg_path.is_absolute():
        candidates.extend([
            repo_root / cfg_path,
            repo_root / 'tools' / cfg_path,
        ])
    candidates.append(cfg_path)

    tools_dir = repo_root / 'tools'
    last_error = None
    old_cwd = os.getcwd()
    for candidate in candidates:
        try:
            os.chdir(str(tools_dir))
            return cfg_from_yaml_file(str(candidate), config)
        except FileNotFoundError as err:
            last_error = err
        finally:
            os.chdir(old_cwd)

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(cfg_file)


def build_once_dataset(args):
    cfg_from_yaml_file_robust(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(Path(args.cfg_file).parts[1:-1])
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = args.eval_split

    root_path = resolve_once_root(cfg.DATA_CONFIG.DATA_PATH)
    info_split = get_info_split(args.eval_split)
    if info_split not in cfg.DATA_CONFIG.INFO_PATH:
        raise KeyError('INFO_PATH has no key %s for eval split %s' % (info_split, args.eval_split))

    once_infos = []
    for info_path in cfg.DATA_CONFIG.INFO_PATH[info_split]:
        info_path = root_path / info_path
        if not info_path.exists():
            raise FileNotFoundError(info_path)
        with open(info_path, 'rb') as f:
            once_infos.extend(pickle.load(f))

    once_infos = [info for info in once_infos if 'annos' in info]
    split_items = parse_once_split(root_path, args.eval_split)
    once_infos = expand_once_infos_by_split(once_infos, split_items, args.eval_split, info_split)

    print('Loaded ONCE eval infos: split=%s, info_split=%s, root=%s, samples=%d'
          % (args.eval_split, info_split, root_path, len(once_infos)))
    return OnceEvalDataset(once_infos, cfg.DATA_CONFIG)


def evaluate_fusion_with_once_metrics(detections, dataset, output_dir, eval_split):
    eval_dir = Path(output_dir) / 'eval' / 'fusion' / eval_split / 'default'
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_detections = [normalize_for_once_eval(det) for det in detections]
    if len(eval_detections) != len(dataset.once_infos):
        raise ValueError(
            'Fused detections length (%d) must match ONCE dataset length (%d) for split %s'
            % (len(eval_detections), len(dataset.once_infos), eval_split)
        )

    with open(eval_dir.parent / 'result.pkl', 'wb') as f:
        pickle.dump(eval_detections, f)

    print('\nRunning ONCE evaluation...')
    print('Evaluation split: %s' % eval_split)
    print('Number of ONCE samples: %d' % len(dataset.once_infos))
    print('Number of fused detections: %d' % len(eval_detections))
    if len(eval_detections) > 0:
        print('First detection keys: %s' % list(eval_detections[0].keys()))
        if 'annos' in dataset.once_infos[0]:
            print('First ground truth keys: %s' % list(dataset.once_infos[0]['annos'].keys()))

    ap_result_str, ap_dict = dataset.evaluation(eval_detections, cfg.CLASS_NAMES)

    with open(eval_dir / 'eval_results.txt', 'w') as f:
        f.write(ap_result_str)
    with open(eval_dir / 'eval_results_dict.pkl', 'wb') as f:
        pickle.dump(ap_dict, f)

    print('\nEvaluation Results:')
    print(ap_result_str)
    print('Fusion evaluation saved to %s' % eval_dir)
    return ap_result_str, ap_dict


def summarize(detections, label):
    counts = np.asarray([len(det.get('name', [])) for det in detections], dtype=np.float32)
    print(
        '%s: frames=%d, total_boxes=%d, avg_boxes=%.3f, max_boxes=%d'
        % (label, len(detections), int(counts.sum()), float(counts.mean()) if len(counts) else 0.0,
           int(counts.max()) if len(counts) else 0)
    )


def main():
    args = parse_args()
    if args.load_fused is not None:
        with open(args.load_fused, 'rb') as f:
            fused = pickle.load(f)
        summarize(fused, 'loaded fused (%s)' % args.load_fused)
    else:
        detections = load_detections(args.inputs)
        num_models = len(detections)
        model_names = args.model_names or ['model_%d' % i for i in range(num_models)]
        if len(model_names) != num_models:
            raise ValueError('--model_names length must match --inputs length')

        print('Loaded %d models:' % num_models)
        for name, path, det in zip(model_names, args.inputs, detections):
            summarize(det, '%s (%s)' % (name, path))

        num_frames = len(detections[0])
        fused = []
        for frame_idx in range(num_frames):
            if (frame_idx + 1) % 200 == 0 or frame_idx == 0:
                print('Fusing frame %d/%d' % (frame_idx + 1, num_frames))

            frame_dets = [model_dets[frame_idx] for model_dets in detections]
            boxes, scores, names = fuse_frame(frame_dets, num_models, args)
            fused.append(make_output_det(frame_dets[0], boxes, scores, names))

        summarize(fused, 'fused')

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'wb') as f:
            pickle.dump(fused, f)
        print('Saved fused detections to %s' % output)

    if args.save_eval:
        dataset = build_once_dataset(args)
        evaluate_fusion_with_once_metrics(fused, dataset, args.output_dir, args.eval_split)


if __name__ == '__main__':
    main()
