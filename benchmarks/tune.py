"""Run hardware tuning candidates sequentially, preserving every measurement."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=1000000)
    parser.add_argument('--features', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('benchmark-results/tuning'))
    parser.add_argument('--tiles', type=int, nargs='+', default=[1024, 2048, 4096])
    parser.add_argument('--groups', type=int, nargs='+', default=[1, 2, 4])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'python-package')}
    candidates = []
    # These runs deliberately share neither GPU time nor CPU build work.
    for tile in args.tiles:
        for group in args.groups:
            destination = args.output / f'rows-{tile}-features-{group}.json'
            command = [sys.executable, str(ROOT / 'benchmarks/run.py'),
                       '--rows', str(args.rows), '--features', str(args.features),
                       '--rounds', str(args.rounds), '--repeats', str(args.repeats),
                       '--histogram-rows', str(tile), '--features-per-group', str(group),
                       '--output', str(destination)]
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
            result = json.loads(destination.read_text())
            entry = {'rows_per_tile': tile, 'features_per_group': group,
                     'median_train_seconds': result['median_train_seconds'],
                     'report': str(destination)}
            candidates.append(entry)
            print(json.dumps(entry), flush=True)
    report = {'candidates': candidates, 'best_measured': min(candidates, key=lambda c: c['median_train_seconds']),
              'note': 'A workload-specific measurement, not a universal setting or peak-performance claim.'}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
