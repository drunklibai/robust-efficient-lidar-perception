import argparse
import _init_path
import torch
import numpy as np

from pathlib import Path
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils.gmp_pruning_utils import apply_gmp_pruning_from_cfg
from pcdet.utils.imp_pruning_utils import apply_imp_pruning_from_cfg
from pcdet.utils import common_utils
from pcdet.utils.sampling_policy_utils import apply_sampling_policy_from_cfg


def parse_config():
    parser = argparse.ArgumentParser(description='PCDet inference speed benchmark')
    parser.add_argument('--cfg_file', type=str, required=True, help='model config file')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to test')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (recommend = 1)')
    parser.add_argument('--num_iters', type=int, default=200, help='number of benchmark iterations')
    parser.add_argument('--warmup_iters', type=int, default=20, help='warmup iterations')
    parser.add_argument('--repeat', type=int, default=1, help='repeat the same benchmark protocol multiple times')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def _to_cuda_batch(batch):
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.bool_):
                batch[k] = torch.from_numpy(v).float().cuda(non_blocking=True)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.cuda(non_blocking=True)
        elif isinstance(v, list):
            processed = []
            for item in v:
                if isinstance(item, np.ndarray) and (np.issubdtype(item.dtype, np.number) or np.issubdtype(item.dtype, np.bool_)):
                    processed.append(torch.from_numpy(item).float().cuda(non_blocking=True))
                elif isinstance(item, torch.Tensor):
                    processed.append(item.cuda(non_blocking=True))
                else:
                    processed.append(item)
            batch[k] = processed
    return batch


def main():
    args, cfg = parse_config()

    logger = common_utils.create_logger()
    logger.info('========== Speed Benchmark ==========')
    logger.info(f'Config: {args.cfg_file}')
    logger.info(f'Checkpoint: {args.ckpt}')
    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Warmup iterations per repeat: {args.warmup_iters}')
    logger.info(f'Benchmark iterations per repeat: {args.num_iters}')
    logger.info(f'Repeat: {args.repeat}')
    apply_sampling_policy_from_cfg(cfg, logger=logger)

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set
    )
    apply_gmp_pruning_from_cfg(cfg, model, logger=logger, stage='test')
    apply_imp_pruning_from_cfg(cfg, model, logger=logger, stage='test')
    model.load_params_from_file(args.ckpt, logger, to_cpu=False)
    model.cuda()
    model.eval()

    logger.info('Model built successfully')

    def make_loader_iter():
        return iter(test_loader)

    def next_batch_cuda(loader_iter):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = make_loader_iter()
            batch = next(loader_iter)
        return _to_cuda_batch(batch), loader_iter

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    repeat_summaries = []
    all_times = []
    global_peak_mem = 0.0

    for repeat_idx in range(args.repeat):
        loader_iter = make_loader_iter()

        with torch.no_grad():
            for _ in range(args.warmup_iters):
                batch, loader_iter = next_batch_cuda(loader_iter)
                model(batch)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        times = []
        with torch.no_grad():
            for _ in range(args.num_iters):
                batch, loader_iter = next_batch_cuda(loader_iter)
                starter.record()
                model(batch)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))

        times = np.array(times, dtype=np.float64)
        if len(times) == 0:
            logger.error('No timing data collected!')
            return

        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        global_peak_mem = max(global_peak_mem, peak_mem)
        total_time_s = float(times.sum() / 1000.0)
        mean_latency = float(times.mean())
        median_latency = float(np.median(times))
        p90_latency = float(np.percentile(times, 90))
        p95_latency = float(np.percentile(times, 95))
        fps = 1000.0 / mean_latency

        repeat_summaries.append({
            'total_time_s': total_time_s,
            'mean': mean_latency,
            'median': median_latency,
            'p90': p90_latency,
            'p95': p95_latency,
            'fps': fps,
            'peak_mem': peak_mem
        })
        all_times.append(times)

        logger.info(
            f'Run {repeat_idx + 1}/{args.repeat}: '
            f'total={total_time_s:.3f}s, '
            f'mean={mean_latency:.2f}ms, '
            f'median={median_latency:.2f}ms, '
            f'p90={p90_latency:.2f}ms, '
            f'p95={p95_latency:.2f}ms, '
            f'fps={fps:.2f}, '
            f'peak_mem={peak_mem:.1f}MB'
        )

    all_times = np.concatenate(all_times, axis=0)
    run_means = np.array([x['mean'] for x in repeat_summaries], dtype=np.float64)
    run_totals = np.array([x['total_time_s'] for x in repeat_summaries], dtype=np.float64)

    mean_latency = float(all_times.mean())
    std_latency = float(run_means.std())
    frame_std_latency = float(all_times.std())
    median_latency = float(np.median(all_times))
    p90_latency = float(np.percentile(all_times, 90))
    p95_latency = float(np.percentile(all_times, 95))
    fps = 1000.0 / mean_latency

    logger.info('========== Benchmark Result ==========')
    logger.info(f'Repeats: {args.repeat}')
    logger.info(f'Frames per repeat: {args.num_iters}')
    logger.info(f'Total measured frames: {len(all_times)}')
    logger.info(f'Latency mean: {mean_latency:.2f} ms')
    logger.info(f'Latency run-mean std: {std_latency:.2f} ms')
    logger.info(f'Latency frame std: {frame_std_latency:.2f} ms')
    logger.info(f'Latency median: {median_latency:.2f} ms')
    logger.info(f'Latency p90: {p90_latency:.2f} ms')
    logger.info(f'Latency p95: {p95_latency:.2f} ms')
    logger.info(f'FPS: {fps:.2f}')
    logger.info(f'Total time per {args.num_iters} frames: {run_totals.mean():.3f} ± {run_totals.std():.3f} s')
    logger.info(f'Peak GPU Memory: {global_peak_mem:.1f} MB')
    logger.info('=====================================')


if __name__ == '__main__':
    main()
