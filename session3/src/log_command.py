"""Derived command evidence, never a replacement for a native session export."""
import argparse
import datetime
import json
import subprocess
import time
from common import ROOT, S3, dump, source_hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    folder = S3/'evidence/commands'
    folder.mkdir(parents=True, exist_ok=True)
    log = folder/(a.name+'.log')
    if log.exists():
        raise FileExistsError(log)
    start = time.perf_counter()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    hashes = source_hashes()
    with log.open('w') as f:
        result = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    record = dict(command=command, cwd=str(ROOT), started_utc=stamp,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_sha256_at_start=hashes, elapsed_seconds=time.perf_counter()-start,
                  exit_code=result.returncode, raw_log=str(log.relative_to(ROOT)))
    dump(log.with_suffix('.json'), record)
    print(json.dumps({k:v for k,v in record.items() if k!='source_sha256_at_start'}), flush=True)
    print(log.read_text()[-2500:], flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
