"""Export seven reference detectors on both domains, then fuse and rank."""
import argparse
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['kitti', 'once'], required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--references', required=True, help='YAML list of name/config/checkpoint entries')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--fusion_mode', choices=['weighted_average', 'average', 'best'], default='weighted_average')
    args = p.parse_args()
    refs = yaml.safe_load(Path(args.references).read_text())
    if not isinstance(refs, list) or len(refs) != 7:
        raise ValueError('Provide exactly seven reference entries')
    names = [r['name'] for r in refs]
    if len(set(names)) != 7 or any(Path(n).name != n or n in ['.', '..'] for n in names):
        raise ValueError('Reference names must be unique plain directory names')
    for ref in refs:
        for key in ['config', 'checkpoint']:
            if not Path(ref[key]).is_file():
                raise FileNotFoundError(ref[key])
    common = ['--dataset', args.dataset, '--data_root', args.data_root]
    base = [sys.executable, 'tools/paper_pipeline.py']
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for domain in ['ori', 'cor']:
        outputs = []
        for ref in refs:
            output = dest / ref['name'] / (domain + '.pkl')
            outputs.append(str(output))
            subprocess.run(base + ['predict'] + common + [
                '--cfg_file', ref['config'], '--ckpt', ref['checkpoint'], '--domain', domain,
                '--seed', str(args.seed), '--output', str(output)], check=True)
        subprocess.run(base + ['fuse'] + common + ['--inputs'] + outputs + [
            '--mode', args.fusion_mode, '--output', str(dest / ('fused_' + domain + '.pkl'))], check=True)
    subprocess.run(base + ['rank'] + common + [
        '--clean', str(dest / 'fused_ori.pkl'), '--corrupted', str(dest / 'fused_cor.pkl'),
        '--output', str(Path(args.data_root) / 'ImageSets/train_mixed_sorted_uncertainty.txt')], check=True)


if __name__ == '__main__':
    main()
