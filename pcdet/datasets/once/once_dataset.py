import copy
import pickle
import numpy as np

from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path

from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils
from .once_toolkits import Octopus

class ONCEDataset(DatasetTemplate):
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
        'raw_small': 'raw_small',
        'raw_medium': 'raw_medium',
        'raw_large': 'raw_large',
    }

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = dataset_cfg.DATA_SPLIT['train'] if training else dataset_cfg.DATA_SPLIT['test']
        assert self.get_info_split(self.split) in self.dataset_cfg.INFO_PATH

        self.split_items, self.sample_seq_list, self.sequence_domain_map = self.parse_split_file(self.split)
        self.cam_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
        self.cam_tags = ['top', 'top2', 'left_back', 'left_front', 'right_front', 'right_back', 'back']
        self.toolkits = Octopus(self.root_path)

        self.once_infos = []
        self.include_once_data(self.split)

    def parse_split_file(self, split):
        split_dir = self.root_path / 'ImageSets' / (split + '.txt')
        default_domain = 'cor' if split.endswith('_cor') else None
        if not split_dir.exists() and split.endswith('_cor'):
            base_split_dir = self.root_path / 'ImageSets' / (split[:-4] + '.txt')
            if base_split_dir.exists():
                split_dir = base_split_dir

        if not split_dir.exists():
            return None, None, {}

        split_items = []
        sample_seq_set = set()
        sequence_domain_map = {}

        for line in open(split_dir).readlines():
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue

            seq_id = tokens[0]
            frame_id = None
            domain = default_domain

            if len(tokens) == 2:
                if tokens[1] in ['ori', 'cor']:
                    domain = tokens[1]
                else:
                    frame_id = tokens[1]
            elif len(tokens) >= 3:
                frame_id = tokens[1]
                domain = tokens[2]

            if domain not in [None, 'ori', 'cor']:
                raise ValueError('Unsupported ONCE domain %s in %s' % (domain, split_dir))
            split_items.append((seq_id, frame_id, domain))
            sample_seq_set.add(seq_id)
            if domain is not None and frame_id is None:
                sequence_domain_map[seq_id] = domain

        return split_items, sorted(sample_seq_set), sequence_domain_map

    def get_info_split(self, split):
        if split in self.SPLIT_TO_INFO_SPLIT:
            return self.SPLIT_TO_INFO_SPLIT[split]
        if split.endswith('_cor'):
            return split[:-4]
        if split.startswith('train_mixed'):
            return 'train'
        return split

    def expand_infos_by_split(self, infos):
        if self.split_items is None:
            return infos

        infos_by_seq = {}
        infos_by_seq_frame = {}
        for info in infos:
            infos_by_seq.setdefault(info['sequence_id'], []).append(info)
            infos_by_seq_frame[(str(info['sequence_id']), str(info['frame_id']))] = info

        expanded_infos = []
        missing_seqs = []
        missing_frames = []
        for seq_id, frame_id, domain in self.split_items:
            if frame_id is not None:
                base_info = infos_by_seq_frame.get((str(seq_id), str(frame_id)), None)
                if base_info is None:
                    missing_frames.append((seq_id, frame_id))
                    continue

                info = copy.deepcopy(base_info)
                if domain is not None:
                    info['domain'] = domain
                expanded_infos.append(info)
                continue

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
                % (self.split, len(missing_seqs), self.get_info_split(self.split), missing_seqs[:5])
            )
        if len(missing_frames) > 0:
            raise RuntimeError(
                'ONCE split %s references %d frames missing from %s infos, examples: %s'
                % (self.split, len(missing_frames), self.get_info_split(self.split), missing_frames[:5])
            )

        return expanded_infos

    def include_once_data(self, split):
        if self.logger is not None:
            self.logger.info('Loading ONCE dataset')
        once_infos = []

        info_split = self.get_info_split(split)
        for info_path in self.dataset_cfg.INFO_PATH[info_split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                once_infos.extend(infos)

        def check_annos(info):
            return 'annos' in info

        if self.split != 'raw':
            once_infos = list(filter(check_annos,once_infos))

        once_infos = self.expand_infos_by_split(once_infos)
        self.once_infos.extend(once_infos)

        if self.logger is not None:
            domain_counts = {}
            for info in once_infos:
                domain = info.get('domain', 'ori')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            self.logger.info('Total samples for ONCE dataset: %d' % (len(once_infos)))
            self.logger.info('ONCE split=%s, info_split=%s, domain_counts=%s'
                             % (self.split, self.get_info_split(self.split), domain_counts))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split

        self.split_items, self.sample_seq_list, self.sequence_domain_map = self.parse_split_file(self.split)

    def get_lidar(self, sequence_id, frame_id, use_corrupted=False):
        if not use_corrupted:
            return self.toolkits.load_point_cloud(sequence_id, frame_id)

        bin_path = self.root_path / 'data_cor' / sequence_id / 'lidar_roof' / ('%s.bin' % frame_id)
        assert bin_path.exists(), 'Corrupted lidar file %s not found' % bin_path
        return np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)

    def get_image(self, sequence_id, frame_id, cam_name):
        return self.toolkits.load_image(sequence_id, frame_id, cam_name)

    def project_lidar_to_image(self, sequence_id, frame_id):
        return self.toolkits.project_lidar_to_image(sequence_id, frame_id)

    def point_painting(self, points, info):
        semseg_dir = './' # add your own seg directory
        used_classes = [0,1,2,3,4,5]
        num_classes = len(used_classes)
        frame_id = str(info['frame_id'])
        seq_id = str(info['sequence_id'])
        painted = np.zeros((points.shape[0], num_classes)) # classes + bg
        for cam_name in self.cam_names:
            img_path = Path(semseg_dir) / Path(seq_id) / Path(cam_name) / Path(frame_id+'_label.png')
            calib_info = info['calib'][cam_name]
            cam_2_velo = calib_info['cam_to_velo']
            cam_intri = np.hstack([calib_info['cam_intrinsic'], np.zeros((3, 1), dtype=np.float32)])
            point_xyz = points[:, :3]
            points_homo = np.hstack(
                [point_xyz, np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
            points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
            mask = points_lidar[:, 2] > 0
            points_lidar = points_lidar[mask]
            points_img = np.dot(points_lidar, cam_intri.T)
            points_img = points_img / points_img[:, [2]]
            uv = points_img[:, [0,1]]
            #depth = points_img[:, [2]]
            seg_map = np.array(Image.open(img_path)) # (H, W)
            H, W = seg_map.shape
            seg_feats = np.zeros((H*W, num_classes))
            seg_map = seg_map.reshape(-1)
            for cls_i in used_classes:
                seg_feats[seg_map==cls_i, cls_i] = 1
            seg_feats = seg_feats.reshape(H, W, num_classes).transpose(2, 0, 1)
            uv[:, 0] = (uv[:, 0] - W / 2) / (W / 2)
            uv[:, 1] = (uv[:, 1] - H / 2) / (H / 2)
            uv_tensor = torch.from_numpy(uv).unsqueeze(0).unsqueeze(0)  # [1,1,N,2]
            seg_feats = torch.from_numpy(seg_feats).unsqueeze(0) # [1,C,H,W]
            proj_scores = F.grid_sample(seg_feats, uv_tensor, mode='bilinear', padding_mode='zeros')  # [1, C, 1, N]
            proj_scores = proj_scores.squeeze(0).squeeze(1).transpose(0, 1).contiguous() # [N, C]
            painted[mask] = proj_scores.numpy()
        return np.concatenate([points, painted], axis=1)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.once_infos) * self.total_epochs

        return len(self.once_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.once_infos)

        info = copy.deepcopy(self.once_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        domain = info.get('domain', self.sequence_domain_map.get(seq_id, None))
        use_corrupted = domain == 'cor'
        points = self.get_lidar(seq_id, frame_id, use_corrupted=use_corrupted)

        if self.dataset_cfg.get('POINT_PAINTING', False):
            points = self.point_painting(points, info)

        input_dict = {
            'points': points,
            'frame_id': frame_id,
            'use_corrupted': use_corrupted,
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        data_dict['use_corrupted'] = use_corrupted
        return data_dict

    def get_infos(self, num_workers=4, sample_seq_list=None):
        import concurrent.futures as futures
        import json
        root_path = self.root_path
        cam_names = self.cam_names

        """
        # dataset json format
        {
            'meta_info':
            'calib': {
                'cam01': {
                    'cam_to_velo': list
                    'cam_intrinsic': list
                    'distortion': list
                }
                ...
            }
            'frames': [
                {
                    'frame_id': timestamp,
                    'annos': {
                        'names': list
                        'boxes_3d': list of list
                        'boxes_2d': {
                            'cam01': list of list
                            ...
                        }
                    }
                    'pose': list
                },
                ...
            ]
        }
        # open pcdet format
        {
            'meta_info':
            'sequence_id': seq_idx
            'frame_id': timestamp
            'timestamp': timestamp
            'lidar': path
            'cam01': path
            ...
            'calib': {
                'cam01': {
                    'cam_to_velo': np.array
                    'cam_intrinsic': np.array
                    'distortion': np.array
                }
                ...
            }
            'pose': np.array
            'annos': {
                'name': np.array
                'boxes_3d': np.array
                'boxes_2d': {
                    'cam01': np.array
                    ....
                }
            }
        }
        """
        def process_single_sequence(seq_idx):
            print('%s seq_idx: %s' % (self.split, seq_idx))
            seq_infos = []
            seq_path = Path(root_path) / 'data' / seq_idx
            json_path = seq_path / ('%s.json' % seq_idx)
            with open(json_path, 'r') as f:
                info_this_seq = json.load(f)
            meta_info = info_this_seq['meta_info']
            calib = info_this_seq['calib']
            for f_idx, frame in enumerate(info_this_seq['frames']):
                frame_id = frame['frame_id']
                if f_idx == 0:
                    prev_id = None
                else:
                    prev_id = info_this_seq['frames'][f_idx-1]['frame_id']
                if f_idx == len(info_this_seq['frames'])-1:
                    next_id = None
                else:
                    next_id = info_this_seq['frames'][f_idx+1]['frame_id']
                pc_path = str(seq_path / 'lidar_roof' / ('%s.bin' % frame_id))
                pose = np.array(frame['pose'])
                frame_dict = {
                    'sequence_id': seq_idx,
                    'frame_id': frame_id,
                    'timestamp': int(frame_id),
                    'prev_id': prev_id,
                    'next_id': next_id,
                    'meta_info': meta_info,
                    'lidar': pc_path,
                    'pose': pose
                }
                calib_dict = {}
                for cam_name in cam_names:
                    cam_path = str(seq_path / cam_name / ('%s.jpg' % frame_id))
                    frame_dict.update({cam_name: cam_path})
                    calib_dict[cam_name] = {}
                    calib_dict[cam_name]['cam_to_velo'] = np.array(calib[cam_name]['cam_to_velo'])
                    calib_dict[cam_name]['cam_intrinsic'] = np.array(calib[cam_name]['cam_intrinsic'])
                    calib_dict[cam_name]['distortion'] = np.array(calib[cam_name]['distortion'])
                frame_dict.update({'calib': calib_dict})

                if 'annos' in frame:
                    annos = frame['annos']
                    boxes_3d = np.array(annos['boxes_3d'])
                    if boxes_3d.shape[0] == 0:
                        print(frame_id)
                        continue
                    boxes_2d_dict = {}
                    for cam_name in cam_names:
                        boxes_2d_dict[cam_name] = np.array(annos['boxes_2d'][cam_name])
                    annos_dict = {
                        'name': np.array(annos['names']),
                        'boxes_3d': boxes_3d,
                        'boxes_2d': boxes_2d_dict
                    }

                    points = self.get_lidar(seq_idx, frame_id)
                    corners_lidar = box_utils.boxes_to_corners_3d(np.array(annos['boxes_3d']))
                    num_gt = boxes_3d.shape[0]
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_gt):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annos_dict['num_points_in_gt'] = num_points_in_gt

                    frame_dict.update({'annos': annos_dict})
                seq_infos.append(frame_dict)
            return seq_infos

        sample_seq_list = sample_seq_list if sample_seq_list is not None else self.sample_seq_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_sequence, sample_seq_list)
        all_infos = []
        for info in infos:
            all_infos.extend(info)
        return all_infos

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('once_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            if 'annos' not in infos[k]:
                continue
            print('gt_database sample: %d' % (k + 1))
            info = infos[k]
            frame_id = info['frame_id']
            seq_id = info['sequence_id']
            points = self.get_lidar(seq_id, frame_id)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['boxes_3d']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (frame_id, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_3d': np.zeros((num_samples, 7))
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_3d'] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                raise NotImplementedError
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        from .once_eval.evaluation import get_evaluation_results

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.once_infos]

        once_eval_cfg = self.dataset_cfg.get('ONCE_EVAL', None)
        eval_class_names = class_names
        eval_kwargs = {}
        if once_eval_cfg is not None:
            mode = once_eval_cfg.get('MODE', 'official')
            if mode == 'both':
                kitti_result_str, kitti_dict = get_evaluation_results(
                    eval_gt_annos, eval_det_annos, ['Car'],
                    use_superclass=False,
                    num_pr_points=40,
                    ap_name='AP_R40'
                )
                official_result_str, official_dict = get_evaluation_results(
                    eval_gt_annos, eval_det_annos, class_names
                )

                ap_result_str = (
                    '\n================ KITTI-style Car AP_R40 ================\n'
                    + kitti_result_str
                    + '\n================ Official ONCE AP@50 ================\n'
                    + official_result_str
                )
                ap_dict = {}
                ap_dict.update({'KITTI_' + key: val for key, val in kitti_dict.items()})
                ap_dict.update({'ONCE_' + key: val for key, val in official_dict.items()})
                return ap_result_str, ap_dict
            elif mode == 'kitti_car':
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

        ap_result_str, ap_dict = get_evaluation_results(
            eval_gt_annos, eval_det_annos, eval_class_names, **eval_kwargs
        )

        return ap_result_str, ap_dict

def create_once_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    """
    生成ONCE数据集的信息文件

    遍历指定的数据划分（训练、验证、测试），收集每条样本的基础信息，
    并将结果保存为pickle文件。同时为训练集创建地面真值数据库，用于数据增强。

    Args:
        dataset_cfg: 数据集配置对象
        class_names: 类别名称列表
        data_path: 数据根目录路径
        save_path: 信息文件保存路径
        workers: 并行处理的工作线程数，默认值为4

    Returns:
        无返回值，结果直接保存至指定路径
    """
    dataset = ONCEDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    splits = ['train', 'val', 'test']
    ignore = []

    print('---------------Start to generate data infos---------------')
    for split in splits:
        if split in ignore:
            continue

        filename = 'once_infos_%s.pkl' % split
        filename = save_path / Path(filename)
        dataset.set_split(split)
        once_infos = dataset.get_infos(num_workers=workers)
        with open(filename, 'wb') as f:
            pickle.dump(once_infos, f)
        print('ONCE info %s file is saved to %s' % (split, filename))

    train_filename = save_path / 'once_infos_train.pkl'
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split('train')
    dataset.create_groundtruth_database(train_filename, split='train')
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--runs_on', type=str, default='server', help='')
    args = parser.parse_args()

    if args.func == 'create_once_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))


        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        once_data_path = ROOT_DIR / 'data' / 'once'
        once_save_path = ROOT_DIR / 'data' / 'once'

        if args.runs_on == 'cloud':
            once_data_path = Path('/cache/once/')
            once_save_path = Path('/cache/once/')
            dataset_cfg.DATA_PATH = dataset_cfg.CLOUD_DATA_PATH

        create_once_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Bus', 'Truck', 'Pedestrian', 'Bicycle'],
            data_path=once_data_path,
            save_path=once_save_path
        )
