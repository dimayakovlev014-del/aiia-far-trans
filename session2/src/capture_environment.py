"""Save local hardware/runtime facts without scanning secrets or user documents."""
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from experiment import ROOT,dump


def main():
    hardware=subprocess.run(['sysctl','hw.memsize','hw.model','machdep.cpu.brand_string'],capture_output=True,text=True)
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    (ROOT/'session2/configs/requirements-inference.lock').write_text(''.join(f'{n}=={v}\n' for n,v in sorted(packages.items(),key=lambda x:x[0].lower()) if n not in ('pip','setuptools','uv')))
    dump(ROOT/'session2/evidence/environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
         machine=platform.machine(),hardware=hardware.stdout,hardware_exit=hardware.returncode,packages=packages,
         disk_free_bytes=shutil.disk_usage(ROOT).free,
         baseline_python='CPython 3.9.6 Apple CommandLineTools isolated session2/.venv',
         baseline_hash_seed='0 for canonical baselines and freeze; initial unfixed run preserved separately',
         cached_models_before_download='Checked HF, llama.cpp, LM Studio, Ollama caches: absent',
         spend='No paid services or cloud inference; local time/electricity not monetized'))
    print((ROOT/'session2/evidence/environment.json').read_text())


if __name__=='__main__':
    main()
