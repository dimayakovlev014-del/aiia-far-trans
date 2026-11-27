"""Run the course LLM reranker with grammar-constrained decoding (HW3).

--mask-names off : main method  -> method name llmrank_constrained
--mask-names on  : ablation     -> method name llmrank_constrained_masked
The prompt is built by the unchanged course ranker.py; masking reuses the HW2
wrapper (selishchev_hw2/masked_ranker.py). The free-decoding control runs are the
HW2 outputs in ../selishchev_hw2/outputs/{llmrank,llmrank_masked}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LLM_DIR = ROOT / 'vendor' / 'RecSys-LLMs' / 'project' / 'far-trans-llm'
sys.path.insert(0, str(ROOT / 'selishchev_hw2'))
sys.path.insert(0, str(HERE))  # HW3 modules take priority


def main(argv=None, generator_factory=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    import masked_ranker  # HW2 code: course-module loading, masking and renaming helpers
    ranker, data_adapter = masked_ranker.load_course_modules(LLM_DIR)
    parser = masked_ranker.build_parser(data_adapter)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='float32')
    parser.add_argument('--decoding', choices=['constrained', 'free'], default='constrained')
    args = parser.parse_args(argv)
    masked = args.mask_names == 'on'
    if args.decoding == 'free':  # the provided course reranker, run on the same device/dtype as a control
        method = 'llmrank_free_masked' if masked else 'llmrank_free'
    else:
        method = 'llmrank_constrained_masked' if masked else 'llmrank_constrained'
    if masked:
        masked_ranker.install_mask(ranker)
    masked_ranker.install_renaming(data_adapter, method, {
        'asset_name_masking': masked, 'mask_value': masked_ranker.MASK if masked else None,
        'masked_fields': ['assetName'] if masked else [], 'decoding_change': args.decoding})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / 'generations.jsonl'
    first_line = len(log_path.read_text(encoding='utf-8').splitlines()) if log_path.exists() else 0

    started = time.perf_counter()
    bundle = data_adapter.load(args)
    data_seconds = time.perf_counter() - started
    prep = {'data_and_candidates_seconds': data_seconds}
    generator = None
    if not args.prepare_only:
        from runtime import LoggedGenerator
        if generator_factory is None:
            from constrained import make_generator_class
            backend = make_generator_class()(args.model, args.revision, args.device, args.allow_download,
                                             dtype=args.dtype, constrained=args.decoding == 'constrained')
        else:
            backend = generator_factory()
        prep.update(model_load_seconds=getattr(backend, 'load_seconds', None),
                    constraint_vocabulary_seconds=getattr(backend, 'vocab_seconds', None))
        generator = LoggedGenerator(backend, args.output_dir)
    started = time.perf_counter()
    result = ranker.run(args, bundle, generator=generator)
    ranking_seconds = time.perf_counter() - started

    gen = masked_ranker.summarize_generations(log_path, first_line)
    rows = [json.loads(l) for l in log_path.read_text(encoding='utf-8').splitlines()[first_line:] if l.strip()] \
        if log_path.exists() else []
    if rows:
        inter = [r['response'].get('constraint_interventions', 0) for r in rows]
        gen.update(users_with_interventions=sum(1 for x in inter if x),
                   interventions_total=sum(inter),
                   constraint_steps_total=sum(r['response'].get('constraint_steps', 0) for r in rows))
    summary = {'method': method, 'asset_name_masking': masked, 'prepare_only': bool(args.prepare_only),
               'preparation_seconds': prep,
               'recommendation_generation_seconds': ranking_seconds,
               'command': [sys.executable, Path(__file__).name] + argv,
               'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
               'hardware': masked_ranker.hardware_info(),
               'generations_this_run': gen, 'ranker_result': result}
    (args.output_dir / 'run_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{method}: preparation {prep}, generation pass {ranking_seconds:.1f} s; '
          f'summary in {args.output_dir / "run_summary.json"}')
    return summary


if __name__ == '__main__':
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED='0'))
    main()
