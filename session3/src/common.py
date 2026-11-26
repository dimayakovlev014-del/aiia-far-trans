"""HW9 paths; reuse the pinned HW8 worker, parser and metric without editing them."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
S3 = ROOT / 'session3'
sys.path.insert(0, str(ROOT / 'session2/src'))
from experiment import digest, dump, jsonl, parse_and_fallback
from prepare_data import sha
from evaluate import ndcg, aggregate
from run_inference import Worker


def read(path):
    import json
    return json.loads(path.read_text())


def source_hashes():
    paths = list((S3/'src').glob('*.py')) + list((S3/'reference').glob('*.json'))
    paths += [ROOT/'session2/src'/n for n in ('experiment.py', 'inference_worker.py',
                                             'run_inference.py', 'evaluate.py')]
    paths += list((ROOT/'session2/configs').glob('*.json'))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}
