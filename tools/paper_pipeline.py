"""Reproducible file preparation for the paper. Run from the repository root.

No dataset, prediction cache, or checkpoint is bundled. CUDA imports are lazy so
split preparation, corruption assembly, and metadata validation work on CPU.
"""
import argparse
import copy
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
CORRUPTIONS = ['density_dec', 'density_inc', 'gaussian', 'background', 'rain', 'snow']


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(value, f)


def write_lines(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(x + '\n' for x in lines), encoding='utf-8')


def frame_key(row):
    return (str(row['sequence_id']) + '/' if 'sequence_id' in row else '') + str(row['frame_id'])


def split_line(row, domain):
    prefix = str(row['sequence_id']) + ' ' if 'sequence_id' in row else ''
    return prefix + str(row['frame_id']) + ' ' + domain


def frames(dataset, root, split):
    """Canonical ordering follows the configured OpenPCDet split, not filenames."""
    root = Path(root)
    tokens = [x.split() for x in (root / 'ImageSets' / (split + '.txt')).read_text().splitlines() if x.strip()]
    if dataset == 'kitti':
        result = [{'frame_id': x[0]} for x in tokens]
    else:
        infos = read_pickle(root / ('once_infos_' + split + '.pkl'))
        by_seq = {}
        for info in infos:
            if 'annos' in info:
                row = {'sequence_id': str(info['sequence_id']), 'frame_id': str(info['frame_id'])}
                by_seq.setdefault(row['sequence_id'], []).append(row)
        result = []
        for entry in tokens:
            if len(entry) != 1:
                raise ValueError('Prepare expects the original one-sequence-per-line ONCE split')
            if entry[0] not in by_seq:
                raise ValueError('No annotated infos for sequence ' + entry[0])
            result.extend(by_seq[entry[0]])
    keys = [frame_key(row) for row in result]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('Empty or duplicate frame identities in ' + split)
    return result


def prepare(args):
    root = Path(args.data_root)
    for split in ['train', 'val']:
        rows = frames(args.dataset, root, split)
        for domain in ['ori', 'cor']:
            # ONCE uses train_mixed* to resolve its train info file.
            stem = ('train_mixed_reference_' + domain) if split == 'train' else 'val_' + domain
            write_lines(root / 'ImageSets' / (stem + '.txt'), [split_line(r, domain) for r in rows])
        (root / ('paper_' + split + '_frames.json')).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        if split == 'train':
            mixed = [split_line(row, domain) for row in rows for domain in ['ori', 'cor']]
            write_lines(root / 'ImageSets/train_mixed_nosel.txt', mixed)
    print('Prepared local frame manifests, reference splits, and the complete 1:1 mixture')


def assemble_corruptions(args):
    """Select one cached severity-4 corruption per frame without synthesizing substitutes."""
    root = Path(args.data_root)
    sources = json.loads(Path(args.sources).read_text())
    if set(sources) != set(CORRUPTIONS):
        raise ValueError('Sources must contain exactly: ' + ', '.join(CORRUPTIONS))
    rows = frames(args.dataset, root, args.split)
    rng = np.random.default_rng(args.seed)
    choices = np.resize(np.arange(len(CORRUPTIONS)), len(rows))
    rng.shuffle(choices)
    jobs = []
    for row, choice in zip(rows, choices):
        name = CORRUPTIONS[int(choice)]
        relative = (Path(row['sequence_id']) / 'lidar_roof' if args.dataset == 'once' else Path()) / (row['frame_id'] + '.bin')
        source = Path(sources[name]) / relative
        target = root / ('data_cor' if args.dataset == 'once' else 'training/velodyne_cor') / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size % 16:
            raise ValueError('Expected float32 x,y,z,intensity: ' + str(source))
        if target.exists():
            raise FileExistsError('Refusing to overwrite a cached corruption: ' + str(target))
        jobs.append((row, name, source, target))
    manifest = []
    for row, name, source, target in jobs:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append(dict(row, corruption=name, severity=4, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (root / ('paper_corruptions_' + args.split + '.json')).write_text(
        json.dumps({'seed': args.seed, 'assignment': 'balanced-shuffled', 'frames': manifest}, indent=2), encoding='utf-8')
    print('Assembled', len(jobs), 'cached corruptions; saved assignment and checksums')


def validate_predictions(predictions, rows):
    if len(predictions) != len(rows):
        raise ValueError('Prediction/frame count mismatch')
    for index, (pred, row) in enumerate(zip(predictions, rows)):
        if str(pred.get('frame_id', '')) != row['frame_id']:
            raise ValueError('Prediction frame_id mismatch at index ' + str(index))
        if 'sequence_id' in row and str(pred.get('sequence_id', '')) != row['sequence_id']:
            raise ValueError('Prediction sequence_id mismatch at index ' + str(index))


def predict(args):
    import _init_path  # noqa: F401
    import torch
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    root = Path(args.data_root).resolve()
    rows = frames(args.dataset, root, 'train')
    config = cfg_from_yaml_file(args.cfg_file, EasyDict())
    dc = config.DATA_CONFIG
    dc.DATA_PATH = str(root)
    dc.DATA_SPLIT['test'] = 'train_mixed_reference_' + args.domain
    if args.dataset == 'kitti':
        dc.INFO_PATH['test'] = ['kitti_infos_train.pkl']
        dc.MIXED_DATASET = EasyDict(ENABLED=True)
    logger = common_utils.create_logger()
    common_utils.set_random_seed(args.seed)
    dataset, loader, _ = build_dataloader(dc, config.CLASS_NAMES, args.batch_size, False,
                                         workers=args.workers, logger=logger, training=False)
    if len(dataset) != len(rows):
        raise ValueError('Reference dataset length differs from prepared frame manifest')
    model = build_network(config.MODEL, len(config.CLASS_NAMES), dataset)
    model.load_params_from_file(args.ckpt, logger, to_cpu=True)
    model.cuda().eval()
    result = []
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            pred, _ = model(batch)
            records = dataset.generate_prediction_dicts(batch, pred, config.CLASS_NAMES)
            for record in records:
                row = rows[len(result)]
                if str(record['frame_id']) != row['frame_id']:
                    raise ValueError('Unexpected reference dataloader order')
                record.update(row)
                result.append(record)
    validate_predictions(result, rows)
    write_pickle(args.output, result)


def fuse(args):
    from model_fusion_once import fuse_frame, make_output_det
    rows = frames(args.dataset, args.data_root, 'train')
    predictions = [read_pickle(p) for p in args.inputs]
    if len(predictions) != 7 and not args.allow_single_reference:
        raise ValueError('The main pipeline requires seven references; use --allow_single_reference only for the one-reference ablation')
    if args.allow_single_reference and len(predictions) not in [1, 7]:
        raise ValueError('Expected one or seven references')
    for records in predictions:
        validate_predictions(records, rows)
    options = SimpleNamespace(score_threshold=0., vote_threshold=.5, iou_threshold=.5,
                              nms_threshold=.1, mode=args.mode, iou_backend='cpu')
    fused = []
    for i, row in enumerate(rows):
        boxes, scores, names = fuse_frame([p[i] for p in predictions], len(predictions), options)
        record = make_output_det(row, boxes, scores, names)
        record.update(row)
        fused.append(record)
    write_pickle(args.output, fused)


def normalized(pred):
    boxes = np.asarray(pred.get('boxes_lidar', pred.get('boxes_3d', [])), dtype=np.float32).reshape(-1, 7)
    scores = np.asarray(pred['score'], dtype=np.float32)
    names = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in pred['name']], dtype=str)
    if len(boxes) != len(scores) or len(scores) != len(names):
        raise ValueError('Inconsistent box/score/name counts')
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or (boxes[:, 3:6] <= 0).any():
        raise ValueError('Invalid reference boxes or scores')
    return boxes, scores, names


def sensitivity(clean, corrupted, iou_function):
    a, sa, na = normalized(clean)
    b, sb, nb = normalized(corrupted)
    score_diffs, box_diffs = [], []
    total = 0
    for cls in CLASSES:
        aa, ss = a[na == cls], sa[na == cls]
        bb, tt = b[nb == cls], sb[nb == cls]
        total += max(len(aa), len(bb))
        if not len(aa) or not len(bb):
            continue
        ious = np.clip(iou_function(aa, bb), 0, 1)
        candidates = sorted([(float(ious[i,j]), i, j) for i,j in zip(*np.where(ious >= .5))], reverse=True)
        used_a, used_b = set(), set()
        for iou, i, j in candidates:
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            score_diffs.append(abs(float(ss[i]) - float(tt[j])))
            box_diffs.append(1. - iou)
    missing = 1. - len(score_diffs) / total if total else 0.
    score = float(np.mean(score_diffs)) if score_diffs else 0.
    box = float(np.mean(box_diffs)) if box_diffs else 0.
    return score + box + .5 * missing


def rank(args):
    rows = frames(args.dataset, args.data_root, 'train')
    clean, cor = read_pickle(args.clean), read_pickle(args.corrupted)
    validate_predictions(clean, rows)
    validate_predictions(cor, rows)
    if args.order == 'sensitivity':
        from model_fusion_once import boxes_iou_bev_np
        values = [sensitivity(a, b, boxes_iou_bev_np) for a,b in zip(clean, cor)]
        order = np.argsort(values, kind='stable')
    elif args.order == 'confidence':
        def confidence(pred):
            _, scores, names = normalized(pred)
            parts = [np.mean(np.sort(scores[names == c])[-3:]) for c in CLASSES if np.any(names == c)]
            return float(np.mean(parts)) if parts else 0.
        # Confidence control ranks each clean/corrupted entry separately.
        entries = [(row, dom, confidence(pred)) for row,a,b in zip(rows,clean,cor)
                   for dom,pred in [('ori',a),('cor',b)]]
        entries.sort(key=lambda x: -x[2])
        write_lines(args.output, [split_line(row, dom) for row,dom,_ in entries])
        return
    else:
        order = np.random.default_rng(args.seed).permutation(len(rows))
        values = [0.] * len(rows)
    output = [split_line(rows[int(i)], dom) for i in order for dom in ['ori', 'cor']]
    if args.order == 'random':
        np.random.default_rng(args.seed).shuffle(output)
    write_lines(args.output, output)
    Path(str(args.output) + '.scores.json').write_text(json.dumps([
        dict(row, sensitivity=float(v)) for row,v in zip(rows,values)
    ], indent=2), encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name, function in [('prepare',prepare), ('assemble-corruptions',assemble_corruptions),
                           ('predict',predict), ('fuse',fuse), ('rank',rank)]:
        s = sub.add_parser(name)
        s.set_defaults(function=function)
        s.add_argument('--dataset', choices=['kitti','once'], required=True)
        s.add_argument('--data_root', required=True)
        if name == 'assemble-corruptions':
            s.add_argument('--sources', required=True, help='JSON: six corruption names -> severity-4 cache directories')
            s.add_argument('--split', choices=['train','val'], required=True)
            s.add_argument('--seed', type=int, default=1234)
        elif name == 'predict':
            s.add_argument('--cfg_file', required=True)
            s.add_argument('--ckpt', required=True)
            s.add_argument('--domain', choices=['ori','cor'], required=True)
            s.add_argument('--output', required=True)
            s.add_argument('--seed', type=int, default=1234)
            s.add_argument('--batch_size', type=int, default=1)
            s.add_argument('--workers', type=int, default=4)
        elif name == 'fuse':
            s.add_argument('--inputs', nargs='+', required=True)
            s.add_argument('--output', required=True)
            s.add_argument('--mode', choices=['weighted_average','average','best'], default='weighted_average')
            s.add_argument('--allow_single_reference', action='store_true')
        elif name == 'rank':
            s.add_argument('--clean', required=True)
            s.add_argument('--corrupted', required=True)
            s.add_argument('--output', required=True)
            s.add_argument('--order', choices=['sensitivity','confidence','random'], default='sensitivity')
            s.add_argument('--seed', type=int, default=1234)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    args.function(args)
