"""Install the verified official CPython 3.9 / Mac arm64 Metal wheel locally."""
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import zipfile

from prepare_data import ROOT,sha


def main():
    if sys.version_info[:2]!=(3,9) or platform.system()!='Darwin' or platform.machine()!='arm64':
        raise RuntimeError('This pinned wheel is for CPython 3.9 on macOS arm64; choose an official compatible wheel for other platforms')
    config=json.loads((ROOT/'session2/configs/runtime.json').read_text())
    wheel=ROOT/'session2'/config['wheel']
    if not wheel.exists():
        subprocess.run(['curl','-4','--http1.1','--tls-max','1.2','-fL','--retry','3','--connect-timeout','15',
                        '--max-time','300','--output',str(wheel),config['official_url']],check=True)
    if sha(wheel)!=config['sha256']:
        raise ValueError('Official wheel digest mismatch; do not install')
    with zipfile.ZipFile(wheel) as z:
        if z.testzip() is not None:
            raise ValueError('Wheel ZIP integrity check failed; do not install')
    subprocess.run([sys.executable,'-m','pip','install',str(wheel),
                    '-r',str(ROOT/'session2/configs/requirements-inference.lock')],check=True)


if __name__=='__main__':
    main()
