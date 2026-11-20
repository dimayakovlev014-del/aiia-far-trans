"""Run the unchanged course starter (four provided baselines) and record wall time."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_STARTER = HERE.parent / 'vendor' / 'RecSys-LLMs' / 'project' / 'far-trans' / 'starter.py'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs' / 'baseline')
    parser.add_argument('--starter', type=Path, default=DEFAULT_STARTER)
    args = parser.parse_args(argv)
    command = [sys.executable, str(args.starter), 'run', '--data-dir', str(args.data_dir),
               '--output-dir', str(args.output_dir)]
    started = time.perf_counter()
    subprocess.run(command, check=True, env=dict(os.environ, PYTHONHASHSEED='0'))
    wall = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'run_summary.json').write_text(json.dumps(
        {'command': command, 'pythonhashseed': '0', 'wall_seconds_all_four_baselines_one_run': wall},
        indent=2) + '\n', encoding='utf-8')
    print(f'starter baselines: wall time {wall:.1f} s for the whole run (all four together)')


if __name__ == '__main__':
    main()
