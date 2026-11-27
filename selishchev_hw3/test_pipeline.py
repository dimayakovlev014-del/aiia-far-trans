"""Contract test of run_ranker.py + evaluate.py with FAKE generators (no model; not results).

python test_pipeline.py --data-dir /path/to/FAR-Trans
Builds four fake runs (free/constrained x names/masked) on the 100 users in a temp
folder and requires evaluate.py to report zero failed checks.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'selishchev_hw2'))
sys.path.insert(0, str(HERE))  # HW3 modules take priority


class FakeBackend:
    def __init__(self, constrained):
        self.constrained = constrained
        self.calls = 0
        self.metadata = {'backend': 'FAKE_TEST_NOT_A_MODEL', 'device': 'none',
                         'decoding': 'greedy_grammar_constrained' if constrained else 'free'}
        self.load_seconds = self.vocab_seconds = 0.0

    def generate(self, prompt, max_new_tokens=512):
        self.calls += 1
        ids = [a['ISIN'] for a in json.loads(prompt.split('\n', 1)[1])['candidate_assets']]
        if not self.constrained and self.calls % 7 == 0:
            return {'text': 'not json', 'input_tokens': 1, 'output_tokens': 1, 'seconds': 0.0}
        out = {'text': json.dumps(ids[5:15][::-1]), 'input_tokens': 1, 'output_tokens': 1, 'seconds': 0.0}
        if self.constrained:
            out.update(constraint_steps=30, constraint_interventions=self.calls % 3, constraint_intervention_steps=[])
        return out


def fresh_course_modules():
    for name in ('ranker', 'data_adapter'):
        sys.modules.pop(name, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    a = ap.parse_args()
    import masked_ranker
    import run_baselines
    import run_ranker
    import evaluate
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        run_baselines.main(['--data-dir', str(a.data_dir), '--output-dir', str(out / 'baseline')])
        common = ['--data-dir', str(a.data_dir), '--model', 'fake', '--on-invalid', 'fallback']
        for mask, folder in (('off', 'llmrank'), ('on', 'llmrank_masked')):
            fresh_course_modules()
            masked_ranker.load_course_modules(masked_ranker.DEFAULT_LLM_DIR)
            import runtime
            runtime.LocalGenerator = lambda *x, **k: FakeBackend(False)
            fresh_course_modules()
            masked_ranker.main(common + ['--mask-names', mask, '--output-dir', str(out / folder)])
        for mask, folder in (('off', 'llmrank_constrained'), ('on', 'llmrank_constrained_masked')):
            fresh_course_modules()
            run_ranker.main(common + ['--mask-names', mask, '--output-dir', str(out / folder)],
                            generator_factory=lambda: FakeBackend(True))
        spec = out / 'runs.json'
        spec.write_text(json.dumps([[m, str(out / m), m] for m in
                                    ('llmrank', 'llmrank_masked', 'llmrank_constrained', 'llmrank_constrained_masked')]))
        result = evaluate.main(['--data-dir', str(a.data_dir), '--outputs', str(out), '--runs-json', str(spec),
                                '--results-md', str(out / 'RESULTS.md')])
        assert not result['checks_failed'], result['checks_failed']
        an = result['analysis']
        assert an['llmrank_constrained']['valid_users'] == 100
        assert an['llmrank_constrained_masked_vs_llmrank_constrained']['identical_top10'] == 100
        assert an['llmrank_constrained']['users_with_interventions'] > 0
    print('\nALL PIPELINE CONTRACT CHECKS PASSED (fake generators, not model results)')


if __name__ == '__main__':
    main()
