"""HW2 change: asset-name masking in the course LLMRank-style reranker.

Session 1 ablation (a): does the LLM's reordering of fixed candidates depend on
the real asset names (which can activate pre-trained knowledge about the asset),
or only on the structured fields and the purchase history?

The single change is in the prompt: the value of ``assetName`` is replaced by the
constant ``MASKED`` for every asset (purchase history and all 50 candidates).
Everything else is the unchanged course code: same users, cutoff, 50 CBF
candidates in the same order, same instruction text, model, revision, greedy
decoding, output limit, parser and fallback rule. ISINs stay visible because they
are the required output IDs; aliasing them would also change the output format.

Run ``--mask-names off`` to get the unmasked control (identical to ranker.py)
under the same hardware and settings.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LLM_DIR = HERE.parent / 'vendor' / 'RecSys-LLMs' / 'project' / 'far-trans-llm'
MASK = 'MASKED'


def load_course_modules(llm_dir):
    llm_dir = Path(llm_dir).resolve()
    if not (llm_dir / 'ranker.py').exists():
        raise SystemExit(f'ranker.py not found in {llm_dir}; expected vendor/ at the team repo root')
    sys.path.insert(0, str(llm_dir))
    import data_adapter
    import ranker
    return ranker, data_adapter


def install_mask(ranker):
    """Replace assetName by MASK in every asset record that enters the prompt."""
    original = ranker.asset_record

    def masked_asset_record(isin, assets, cutoff):
        record = original(isin, assets, cutoff)
        record['assetName'] = MASK
        return record

    ranker.asset_record = masked_asset_record
    return original


def install_renaming(data_adapter, method_name, extra_metadata):
    """Rename the 'llmrank' column and add masking metadata to run_manifest.json."""
    original = data_adapter.save_results

    def save_results(args, bundle, predictions, failures, method, runtime_metadata):
        renamed = {uid: {(method_name if key == 'llmrank' else key): ranking
                         for key, ranking in by_method.items()}
                   for uid, by_method in predictions.items()}
        return original(args, bundle, renamed, failures, method_name,
                        dict(runtime_metadata, **extra_metadata))

    data_adapter.save_results = save_results
    return original


def hardware_info():
    info = {'platform': platform.platform(), 'machine': platform.machine(),
            'processor': platform.processor(), 'python': sys.version.split()[0]}
    if sys.platform == 'darwin':
        for key, name in (('cpu', 'machdep.cpu.brand_string'), ('memory_bytes', 'hw.memsize')):
            try:
                info[key] = subprocess.run(['sysctl', '-n', name], capture_output=True,
                                           text=True, check=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                pass
    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        info['cuda_device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        mps = getattr(torch.backends, 'mps', None)
        info['mps_available'] = bool(mps and mps.is_available())
    except ImportError:
        info['torch'] = None
    return info


def summarize_generations(log_path, first_line):
    """Token and time totals for the generation records appended by this run."""
    if not log_path.exists():
        return {}
    lines = log_path.read_text(encoding='utf-8').splitlines()[first_line:]
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        return {}
    inputs = [r['response']['input_tokens'] for r in rows]
    outputs = [r['response']['output_tokens'] for r in rows]
    seconds = [r['response']['seconds'] for r in rows if not r['cache_hit']]
    return {'records': len(rows), 'cache_hits': sum(r['cache_hit'] for r in rows),
            'input_tokens_mean': sum(inputs) / len(inputs), 'input_tokens_max': max(inputs),
            'output_tokens_mean': sum(outputs) / len(outputs),
            'generation_seconds_new_calls': sum(seconds)}


def build_parser(data_adapter):
    parser = data_adapter.arguments('Course LLMRank adaptation with optional asset-name masking')
    for action in parser._actions:
        if action.dest == 'device':
            # The course runtime passes the device string to torch unchanged;
            # 'mps' (Apple Silicon) keeps the runtime's float32 non-CUDA dtype.
            action.choices = ['auto', 'cpu', 'cuda', 'mps']
    parser.add_argument('--llm-dir', type=Path, default=DEFAULT_LLM_DIR,
                        help='Path to project/far-trans-llm of the course repository')
    parser.add_argument('--mask-names', choices=['on', 'off'], required=True)
    parser.add_argument('--shuffle-seed', type=int)
    parser.add_argument('--history-events', type=int, default=15)
    parser.add_argument('--prepare-only', action='store_true')
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--llm-dir', type=Path, default=DEFAULT_LLM_DIR)
    known, _ = pre.parse_known_args(argv)
    ranker, data_adapter = load_course_modules(known.llm_dir)
    args = build_parser(data_adapter).parse_args(argv)
    masked = args.mask_names == 'on'
    method_name = 'llmrank_masked' if masked else 'llmrank'
    if masked:
        install_mask(ranker)
    install_renaming(data_adapter, method_name,
                     {'asset_name_masking': masked, 'mask_value': MASK if masked else None,
                      'masked_fields': ['assetName'] if masked else []})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / 'generations.jsonl'
    first_line = len(log_path.read_text(encoding='utf-8').splitlines()) if log_path.exists() else 0
    started = time.perf_counter()
    result = ranker.run(args, data_adapter.load(args))
    wall = time.perf_counter() - started
    summary = {'method': method_name, 'asset_name_masking': masked,
               'prepare_only': bool(args.prepare_only),
               'wall_seconds_including_data_and_model_load': wall,
               'command': [sys.executable, Path(__file__).name] + argv,
               'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
               'hardware': hardware_info(),
               'generations_this_run': summarize_generations(log_path, first_line),
               'ranker_result': result}
    (args.output_dir / 'run_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{method_name}: wall time {wall:.1f} s; summary in {args.output_dir / "run_summary.json"}')
    return summary


def ensure_hash_seed():
    # Same canonical setting as the team's session2 runs: teacher code iterates
    # Python sets, so near-tie order can depend on the process hash seed.
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execve(sys.executable, [sys.executable] + sys.argv,
                  dict(os.environ, PYTHONHASHSEED='0'))


if __name__ == '__main__':
    ensure_hash_seed()
    main()
