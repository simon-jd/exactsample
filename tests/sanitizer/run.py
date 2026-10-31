"""Run all four real Compute Sanitizer tools, returning 77 if unavailable."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/sanitizer'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    executable = shutil.which('compute-sanitizer')
    try:
        import torch
        available = torch.cuda.is_available()
    except ImportError:
        available = False
    if executable is None or not available:
        report = {'status': 'skipped', 'reason': 'compute-sanitizer or CUDA runtime unavailable',
                  'tools_run': [], 'qualification': False}
        (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return 77
    # EXACTSAMPLE_SANITIZER skips the full-vocabulary tests, which take hours under memcheck.
    environment = dict(os.environ, EXACTSAMPLE_REQUIRE_CUDA='1', EXACTSAMPLE_SANITIZER='1')
    results = []
    for tool in ['memcheck', 'racecheck', 'initcheck', 'synccheck']:
        command = [executable, '--tool', tool, '--error-exitcode', '1',
                   '--log-file', str(args.output_dir / f'{tool}.log'),
                   sys.executable, '-m', 'unittest', 'tests.test_cuda', 'tests.test_fused', '-v']
        completed = subprocess.run(command, env=environment, capture_output=True, text=True)
        (args.output_dir / f'{tool}-test.log').write_text(completed.stdout + completed.stderr)
        shown = [os.path.relpath(c, ROOT) if os.path.isabs(c) and c.startswith(str(ROOT)) else c for c in command]
        results.append({'tool': tool, 'command': shown, 'returncode': completed.returncode})
    report = {'status': 'failed' if any(r['returncode'] for r in results) else 'passed',
              'tools_run': results, 'qualification': True}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
