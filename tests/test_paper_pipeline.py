import importlib.util
import ast
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_pipeline', ROOT / 'tools/paper_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def pred(boxes=(), scores=(), names=()):
    return dict(boxes_lidar=np.asarray(boxes).reshape(-1,7), score=np.asarray(scores), name=np.asarray(names))


def test_empty_reference_pair_has_zero_sensitivity():
    assert pipeline.sensitivity(pred(), pred(), lambda a,b: None) == 0


def test_unmatched_and_matched_sensitivity():
    a = pred([[0,0,0,1,1,1,0]], [.8], ['Car'])
    b = pred([[0,0,0,1,1,1,0]], [.6], ['Car'])
    assert pipeline.sensitivity(a, pred(), lambda a,b: None) == .5
    assert pipeline.sensitivity(a,b,lambda a,b: np.array([[.75]])) == pytest.approx(.45)


def test_matching_is_one_to_one_and_class_specific():
    a = pred([[0,0,0,1,1,1,0]]*2, [.8,.8], ['Car','Car'])
    b = pred([[0,0,0,1,1,1,0]], [.8], ['Car'])
    assert pipeline.sensitivity(a,b,lambda a,b: np.ones((2,1))) == .25
    b['name'] = np.array(['Pedestrian'])
    assert pipeline.sensitivity(a,b,lambda a,b: None) == .5


def test_prediction_alignment_rejects_wrong_order_and_sequence():
    rows = [{'sequence_id':'a','frame_id':'1'}, {'sequence_id':'b','frame_id':'2'}]
    with pytest.raises(ValueError):
        pipeline.validate_predictions(list(reversed(rows)), rows)
    with pytest.raises(ValueError):
        pipeline.validate_predictions([{'frame_id':'1'}, {'frame_id':'2'}], rows)


def setup_kitti(root):
    (root / 'ImageSets').mkdir()
    (root / 'ImageSets/train.txt').write_text('000001\n000002\n')
    (root / 'ImageSets/val.txt').write_text('000003\n')


def test_prepare_keeps_all_pairs_without_touching_original_split(tmp_path):
    setup_kitti(tmp_path)
    pipeline.prepare(SimpleNamespace(dataset='kitti',data_root=tmp_path))
    lines = (tmp_path/'ImageSets/train_mixed_nosel.txt').read_text().splitlines()
    assert lines == ['000001 ori','000001 cor','000002 ori','000002 cor']
    assert (tmp_path/'ImageSets/train.txt').read_text() == '000001\n000002\n'
    assert (tmp_path/'ImageSets/val_cor.txt').read_text() == '000003 cor\n'


def test_once_order_follows_sequence_split_and_filters_unlabelled(tmp_path):
    (tmp_path/'ImageSets').mkdir()
    (tmp_path/'ImageSets/train.txt').write_text('b\na\n')
    infos = [{'sequence_id':'a','frame_id':'1','annos':{}},
             {'sequence_id':'b','frame_id':'2','annos':{}},
             {'sequence_id':'b','frame_id':'3'}]
    with (tmp_path/'once_infos_train.pkl').open('wb') as f:
        pickle.dump(infos,f)
    assert [r['frame_id'] for r in pipeline.frames('once',tmp_path,'train')] == ['2','1']


def test_corruption_cache_is_fixed_and_never_overwritten(tmp_path):
    setup_kitti(tmp_path)
    sources = {}
    for name in pipeline.CORRUPTIONS:
        source = tmp_path / name
        source.mkdir()
        for frame in ['000001','000002']:
            np.zeros((2,4),np.float32).tofile(source/(frame+'.bin'))
        sources[name] = str(source)
    path = tmp_path/'sources.json'
    path.write_text(json.dumps(sources))
    args = SimpleNamespace(dataset='kitti',data_root=tmp_path,sources=path,split='train',seed=1234)
    pipeline.assemble_corruptions(args)
    saved = json.loads((tmp_path/'paper_corruptions_train.json').read_text())
    assert len(saved['frames']) == 2
    assert all(len(row['sha256']) == 64 for row in saved['frames'])
    with pytest.raises(FileExistsError):
        pipeline.assemble_corruptions(args)


def test_paper_configs_have_consistent_protocol():
    for dataset in ['kitti','once']:
        for model in ['pointrcnn','second']:
            cfg = yaml.safe_load((ROOT/f'tools/cfgs/paper/{dataset}/{model}_ours.yaml').read_text())
            data = cfg['DATA_CONFIG']
            assert data['SHUFFLE_TRAIN'] is False
            assert data['DATA_SPLIT']['train'] == 'train_mixed_sorted_uncertainty'
            assert 'DISTILL' not in data
            assert cfg['MODEL']['SAMPLING_POLICY_PATH'] == f'output/policies/{dataset}_{model}.yaml'
            if dataset == 'once':
                assert data['ONCE_EVAL']['NUM_PR_POINTS'] == 40
                assert data['ONCE_EVAL']['EVAL_CLASSES'] == pipeline.CLASSES


@pytest.mark.parametrize('dataset', ['kitti', 'once'])
@pytest.mark.parametrize('model', ['pointrcnn', 'second'])
def test_policy_generation_runs_on_cpu_and_preserves_compact_widths(tmp_path, dataset, model):
    (tmp_path/'ImageSets').mkdir()
    if dataset == 'kitti':
        point_dir = tmp_path/'training/velodyne_ori'
        split = '000001 ori\n'
    else:
        point_dir = tmp_path/'data/sequence/lidar_roof'
        split = 'sequence 000001 ori\n'
    point_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    points = rng.uniform([1,-2,-1,0], [10,2,0,1], size=(64,4)).astype(np.float32)
    points.tofile(point_dir/'000001.bin')
    (tmp_path/'ImageSets/train_mixed_reference_ori.txt').write_text(split)
    variant = 'original' if model == 'pointrcnn' else 'ours'
    cfg_path = ROOT/f'tools/cfgs/paper/{dataset}/{model}_{variant}.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    script = 'search_sampling_policy.py' if dataset == 'kitti' else 'search_sampling_policy_once.py'
    output = tmp_path/'policy.yaml'
    subprocess.run([sys.executable, str(ROOT/'tools'/script), '--cfg_file',str(cfg_path),
                    '--data_path',str(tmp_path), '--split','train_mixed_reference_ori',
                    '--num_frames','0','--num_query','4','--num_ref','8','--output',str(output)],
                   cwd=ROOT, capture_output=True, text=True, check=True)
    policy = yaml.safe_load(output.read_text())
    if model == 'pointrcnn':
        assert policy['MODEL']['ROI_HEAD']['SA_CONFIG']['NPOINTS'][-1] == -1
        original = cfg['MODEL']['BACKBONE_3D']['SA_CONFIG']['NPOINTS']
        reduced = policy['MODEL']['BACKBONE_3D']['SA_CONFIG']['NPOINTS']
        assert all(0 < new <= old for old,new in zip(original,reduced))
    else:
        assert policy['MODEL']['BACKBONE_2D']['NUM_FILTERS'] == cfg['MODEL']['BACKBONE_2D']['NUM_FILTERS']
        old = next(p for p in cfg['DATA_CONFIG']['DATA_PROCESSOR'] if p['NAME']=='transform_points_to_voxels')
        new = next(p for p in policy['DATA_CONFIG']['DATA_PROCESSOR'] if p['NAME']=='transform_points_to_voxels')
        assert new['MAX_NUMBER_OF_VOXELS']['test'] < old['MAX_NUMBER_OF_VOXELS']['test']


def test_internal_module_dependencies_are_in_release():
    missing = []
    for source in (ROOT/'pcdet').rglob('*.py'):
        for node in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
            if not isinstance(node, ast.ImportFrom) or not node.level or not node.module:
                continue
            parent = source.parent
            for _ in range(node.level - 1):
                parent = parent.parent
            target = parent.joinpath(*node.module.split('.'))
            if not target.with_suffix('.py').exists() and not (target/'__init__.py').exists():
                missing.append((str(source.relative_to(ROOT)), node.module))
    assert not missing, missing


def test_all_declared_cuda_extension_sources_are_present():
    tree = ast.parse((ROOT/'setup.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'make_cuda_ext':
            args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            folder = ROOT.joinpath(*args['module'].split('.'))
            assert all((folder/s).is_file() for s in args['sources'])
