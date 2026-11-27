"""Run/resume smoke, wait for its runtime review, then run/resume 100 users."""
import json,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(here/'experiment.py'),'--phase','smoke'],check=True)
print('Smoke complete: 3 users. Waiting for runtime review before 100 users.',flush=True)
deadline=time.monotonic()+1800
protocol=json.loads((here/'manifest.json').read_text())['protocol_hash']
while not (here/'full.authorized').exists():
    if time.monotonic()>deadline:raise TimeoutError('Runtime review pending; smoke scores saved. Resume this same command later.')
    time.sleep(2)
assert (here/'full.authorized').read_text().strip()==protocol
subprocess.run([sys.executable,str(here/'experiment.py'),'--phase','full'],check=True)
