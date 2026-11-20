"""Contract tests with a FAKE generator (no model). These are not experiment results.

Usage: python test_pipeline.py --data-dir /path/to/FAR-Trans
Runs the baselines, both ranker variants with a fake model on 100 users, and the
evaluation checks, all in a temporary folder.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate  # noqa: E402
import masked_ranker  # noqa: E402
import run_baselines  # noqa: E402


class FakeLocalGenerator:
    """Returns the last ten candidates reversed; every 7th call is invalid text."""
    calls = 0

    def __init__(self, *args, **kwargs):
        self.metadata = {'backend': 'FAKE_TEST_NOT_A_MODEL', 'device': 'none'}

    def generate(self, prompt, max_new_tokens=512):
        FakeLocalGenerator.calls += 1
        payload = json.loads(prompt.split('\n', 1)[1])
        ids = [a['ISIN'] for a in payload['candidate_assets']]
        text = 'not json' if FakeLocalGenerator.calls % 7 == 0 else json.dumps(ids[-10:][::-1])
        return {'text': text, 'input_tokens': len(prompt) // 4, 'output_tokens': 10, 'seconds': 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'outputs'
        run_baselines.main(['--data-dir', str(args.data_dir), '--output-dir', str(out / 'baseline')])
        for mask, folder in (('off', 'llmrank'), ('on', 'llmrank_masked')):
            masked_ranker.load_course_modules(masked_ranker.DEFAULT_LLM_DIR)
            import runtime
            runtime.LocalGenerator = FakeLocalGenerator
            for name in ('ranker', 'data_adapter'):
                sys.modules.pop(name, None)  # fresh, unpatched course modules per variant
            FakeLocalGenerator.calls = 0
            masked_ranker.main(['--data-dir', str(args.data_dir), '--output-dir', str(out / folder),
                                '--model', 'fake', '--mask-names', mask, '--on-invalid', 'fallback'])
            sys.modules['runtime'].LocalGenerator = FakeLocalGenerator
        result = evaluate.main(['--data-dir', str(args.data_dir), '--outputs', str(out),
                                '--results-md', str(Path(tmp) / 'RESULTS.md')])
        assert not result['checks_failed'], result['checks_failed']
        a = result['analysis']
        # Identical prompts are served from the course cache, so the fake sees <=100 calls.
        assert 80 <= a['llmrank']['valid_users'] < 100, a['llmrank']
        assert result['rows'][-1]['generation']['records'] == 100
        print('cache hits:', result['rows'][-1]['generation']['cache_hits'])
        assert a['masked_vs_unmasked']['identical_top10_lists'] == 100
        print('\nALL CONTRACT CHECKS PASSED (fake generator, not a model result)')


if __name__ == '__main__':
    main()
