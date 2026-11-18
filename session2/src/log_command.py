"""Run a command with untouched combined output, exit code and external elapsed time."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cwd', default='.')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    root = Path(__file__).resolve().parents[2]
    folder = root / 'session2/evidence/commands'
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / (a.name + '.log')
    if log.exists():
        raise FileExistsError(log)
    start = time.perf_counter()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log.open('w') as f:
        result = subprocess.run(command, cwd=a.cwd, stdout=f, stderr=subprocess.STDOUT)
    record = dict(command=command, cwd=str(Path(a.cwd).resolve()), started_utc=stamp,
                  elapsed_seconds=time.perf_counter()-start, exit_code=result.returncode,
                  raw_log=str(log.relative_to(root)))
    log.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    print(log.read_text()[-6000:], flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
