"""Cross-check the runs and build the HW2 results table.

1. Re-scores each LLM run through the unchanged starter (``--responses``), so the
   LLM methods get a ``protocol.json`` with the same protocol_hash as the baselines.
2. Checks: same users, data hashes, baseline scores, identical 50 candidates in the
   same order, masked prompts differ from unmasked prompts only in assetName, no date
   on or after the cutoff in any prompt.
3. Rebuilds the truth labels from transactions.csv independently of the starter.
4. Writes outputs/evaluation/results.json (full, user-level; not for commit) and
   RESULTS.md (aggregate only; safe to commit).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_STARTER = HERE.parent / 'vendor' / 'RecSys-LLMs' / 'project' / 'far-trans' / 'starter.py'
BASELINES = ['popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell']
DATE = re.compile(r'\d{4}-\d{2}-\d{2}')


def jload(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jlines(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def predictions(path):
    result = defaultdict(dict)
    with open(path, newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            result[row['method']].setdefault(row['user_id'], []).append((int(row['rank']), row['ISIN']))
    return {m: {u: [i for _, i in sorted(v)] for u, v in d.items()} for m, d in result.items()}


def ndcg10(ranking, truth):
    if not truth:
        return 0.0
    gain = sum(1 / math.log2(i + 2) for i, isin in enumerate(ranking[:10]) if isin in truth)
    return gain / sum(1 / math.log2(i + 2) for i in range(min(10, len(truth))))


def rebuild_truth(data_dir, users, eligible, cutoff, end):
    truth = defaultdict(set)
    users = set(users)
    with open(Path(data_dir) / 'transactions.csv', newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if (row['customerID'] in users and row['transactionType'] == 'Buy'
                    and int(row['transactionID']) >= 0 and row['ISIN'] in eligible
                    and cutoff <= row['timestamp'][:10] < end):
                truth[row['customerID']].add(row['ISIN'])
    return truth


def payload_of(prompt):
    instruction, body = prompt.split('\n', 1)
    return instruction, json.loads(body)


def check(condition, message, failures):
    if not condition:
        failures.append(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, default=HERE / 'outputs')
    parser.add_argument('--starter', type=Path, default=DEFAULT_STARTER)
    parser.add_argument('--results-md', type=Path, default=HERE / 'RESULTS.md')
    args = parser.parse_args(argv)
    out = args.outputs
    base_dir, eval_dir = out / 'baseline', out / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    protocol = jload(base_dir / 'protocol.json')
    base_metrics = jload(base_dir / 'metrics.json')
    contexts = {c['user_id']: c for c in jlines(base_dir / 'contexts.jsonl')}
    users, cutoff, end = protocol['users'], protocol['cutoff'], protocol['end_exclusive']
    candidates = {u: [a['id'] for a in contexts[u]['candidate_assets']] for u in users}
    truth = rebuild_truth(args.data_dir, users, set(protocol['eligible_assets']), cutoff, end)
    nonempty = [u for u in users if truth.get(u)]

    runs = {}
    for name, folder in (('llmrank', 'llmrank'), ('llmrank_masked', 'llmrank_masked')):
        d = out / folder
        if not (d / 'metrics.json').exists():
            print(f'WARNING: {d} has no completed run; {name} reported as incomplete')
            continue
        manifest = jload(d / 'run_manifest.json')
        finals = {r['user_id']: r for r in jlines(d / 'final_responses.jsonl')}
        preds = predictions(d / 'predictions.csv')
        gens = jlines(d / 'generations.jsonl')[-len(users):]
        runs[name] = {'dir': d, 'metrics': jload(d / 'metrics.json'), 'manifest': manifest,
                      'finals': finals, 'preds': preds, 'gens': gens,
                      'summary': jload(d / 'run_summary.json') if (d / 'run_summary.json').exists() else {}}
        # 1. Same cohort and data as the starter run.
        check(manifest['users'] == users, f'{name}: user list differs from protocol.json', failures)
        check(manifest['cutoff'] == cutoff and manifest['end_exclusive'] == end,
              f'{name}: cutoff/window differs', failures)
        check(manifest['data_sha256'] == protocol['source_sha256'], f'{name}: data hashes differ', failures)
        check(manifest['candidate_count'] == protocol['candidate_count'], f'{name}: candidate count differs', failures)
        for b in BASELINES:
            check(abs(runs[name]['metrics'][b]['ndcg_at_10'] - base_metrics['methods'][b]['ndcg_at_10']) < 1e-12,
                  f'{name}: baseline {b} differs from starter', failures)
        check(list(finals) == users, f'{name}: final_responses not one per user in order', failures)
        # 2. Prompts: same candidates in the same order; no future dates.
        for uid, record in zip(users, gens):
            instruction, payload = payload_of(record['prompt'])
            check([a['ISIN'] for a in payload['candidate_assets']] == candidates[uid],
                  f'{name}/{uid}: prompt candidates differ from starter contexts', failures)
            dates = [x for x in DATE.findall(json.dumps(payload['profile_as_of_cutoff'])
                                             + json.dumps(payload['buy_history_counts'])
                                             + json.dumps(payload['recent_buy_events_oldest_to_newest'])
                                             + json.dumps(payload['candidate_assets']))]
            check(all(x < cutoff for x in dates), f'{name}/{uid}: date on/after cutoff in prompt', failures)
            check(record['response']['text'] == finals[uid]['raw_response'],
                  f'{name}/{uid}: logged generation does not match final response', failures)
        # 3. Re-score through the unchanged starter under the same protocol hash.
        responses = eval_dir / f'{name}_responses.jsonl'
        with responses.open('w', encoding='utf-8') as stream:
            for uid in users:
                stream.write(json.dumps({'user_id': uid, 'stage': 'ranker',
                                         'protocol_hash': protocol['protocol_hash'],
                                         'ranked_ids': preds[name][uid]}) + '\n')
        starter_out = eval_dir / f'starter_{name}'
        subprocess.run([sys.executable, str(args.starter), 'run', '--data-dir', str(args.data_dir),
                        '--output-dir', str(starter_out), '--stage', 'ranker',
                        '--responses', str(responses), '--on-invalid', 'error'],
                       check=True, env=dict(os.environ, PYTHONHASHSEED='0'), stdout=subprocess.DEVNULL)
        starter_metrics = jload(starter_out / 'metrics.json')
        check(jload(starter_out / 'protocol.json')['protocol_hash'] == protocol['protocol_hash'],
              f'{name}: starter re-score protocol hash differs', failures)
        check(abs(starter_metrics['methods']['custom']['ndcg_at_10'] - runs[name]['metrics'][name]['ndcg_at_10']) < 1e-12,
              f'{name}: starter re-score differs from ranker score', failures)
        runs[name]['starter_custom'] = starter_metrics['methods']['custom']

    if {'llmrank', 'llmrank_masked'} <= set(runs):
        for uid, a, b in zip(users, runs['llmrank']['gens'], runs['llmrank_masked']['gens']):
            ia, pa = payload_of(a['prompt'])
            ib, pb = payload_of(b['prompt'])
            for section in ('buy_history_counts', 'candidate_assets'):
                for row in pa[section]:
                    row['assetName'] = 'MASKED'
            check(ia == ib and pa == pb, f'{uid}: masked prompt differs beyond assetName', failures)
            check(all(r['assetName'] == 'MASKED' for r in pb['candidate_assets'] + pb['buy_history_counts']),
                  f'{uid}: unmasked name in masked prompt', failures)

    # Independent nDCG from rebuilt truth must equal the reported per-user values.
    score = {}
    all_preds = predictions(base_dir / 'predictions.csv')
    first = next(iter(runs.values()), None)
    if first:
        all_preds['candidate_order_first10'] = first['preds']['candidate_order_first10']
    for name, run in runs.items():
        all_preds[name] = run['preds'][name]
        reported = {(r['user_id'], r['method']): r for r in jload(run['dir'] / 'per_user.json')}
        for uid in users:
            r = reported[(uid, name)]
            check(r['relevant_count'] == len(truth.get(uid, ())), f'{uid}: truth count differs', failures)
            check(abs(r['ndcg_at_10'] - ndcg10(all_preds[name][uid], truth.get(uid, set()))) < 1e-12,
                  f'{name}/{uid}: independent nDCG differs', failures)
    for method, by_user in all_preds.items():
        score[method] = {u: ndcg10(by_user[u], truth.get(u, set())) for u in users}

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    rows = []
    timing = jload(base_dir / 'run_summary.json') if (base_dir / 'run_summary.json').exists() else {}
    for method in BASELINES + ['candidate_order_first10'] + list(runs):
        if method not in score:
            continue
        run = runs.get(method)
        rows.append({
            'method': method, 'users': len(users), 'nonempty_users': len(nonempty),
            'ndcg_all': mean(score[method][u] for u in users),
            'ndcg_nonempty': mean(score[method][u] for u in nonempty),
            'failed_cases': run['manifest']['runtime']['raw_invalid_count'] if run else 0,
            'parse_repairs': run['manifest']['runtime']['parse_repair_count'] if run else 0,
            'wall_seconds': (run['summary'].get('wall_seconds_including_data_and_model_load') if run
                             else None if method == 'candidate_order_first10'
                             else timing.get('wall_seconds_all_four_baselines_one_run')),
            'generation': run['summary'].get('generations_this_run') if run else None,
        })

    analysis = {}
    for name, run in runs.items():
        valid = [u for u in users if run['finals'][u]['error'] is None]
        ranked = all_preds[name]
        analysis[name] = {
            'valid_users': len(valid), 'valid_nonempty_users': sum(1 for u in valid if u in nonempty),
            'valid_only_ndcg_all': mean(score[name][u] for u in valid),
            'valid_only_candidate_order_ndcg_all': mean(score['candidate_order_first10'][u] for u in valid),
            'valid_only_ndcg_nonempty': mean(score[name][u] for u in valid if u in nonempty),
            'valid_only_candidate_order_ndcg_nonempty': mean(score['candidate_order_first10'][u] for u in valid if u in nonempty),
            'users_ranking_differs_from_candidate_order': sum(ranked[u] != candidates[u][:10] for u in users),
            'mean_overlap_with_candidate_first10': mean(len(set(ranked[u]) & set(candidates[u][:10])) for u in users),
            'error_types': sorted({(run['finals'][u]['error'] or '').split(':')[0] for u in users} - {''}),
        }
    if {'llmrank', 'llmrank_masked'} <= set(runs):
        a, b = all_preds['llmrank'], all_preds['llmrank_masked']
        analysis['masked_vs_unmasked'] = {
            'identical_top10_lists': sum(a[u] == b[u] for u in users),
            'mean_top10_overlap': mean(len(set(a[u]) & set(b[u])) for u in users),
            'nonempty_users_better_masked': sum(score['llmrank_masked'][u] > score['llmrank'][u] for u in nonempty),
            'nonempty_users_worse_masked': sum(score['llmrank_masked'][u] < score['llmrank'][u] for u in nonempty),
        }
    ceiling = [u for u in nonempty if truth[u] & set(candidates[u])]
    analysis['candidate_ceiling'] = {
        'nonempty_users_with_a_future_buy_among_50_candidates': len(ceiling),
        'best_possible_rerank_ndcg_all': mean(
            ndcg10([i for i in candidates[u] if i in truth.get(u, ())] + [i for i in candidates[u] if i not in truth.get(u, ())],
                   truth.get(u, set())) for u in users)}

    per_user = {u: {m: score[m][u] for m in score} | {'relevant': len(truth[u]),
                'truth_in_candidates': len(truth[u] & set(candidates[u]))} for u in nonempty}
    result = {'protocol_hash': protocol['protocol_hash'], 'cutoff': cutoff, 'end_exclusive': end,
              'checks_failed': failures, 'rows': rows, 'analysis': analysis,
              'per_nonempty_user': per_user,
              'runtime_metadata': {n: r['manifest']['runtime'] for n, r in runs.items()},
              'hardware': {n: r['summary'].get('hardware') for n, r in runs.items()}}
    (eval_dir / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def f4(x):
        return 'n/a' if x is None else f'{x:.4f}'

    lines = ['# HW2 results (aggregate only)', '',
             f'Protocol hash `{protocol["protocol_hash"]}`, cutoff {cutoff}, window to {end} (exclusive).', '']
    for name, run in runs.items():
        meta, hw = run['manifest']['runtime'], run['summary'].get('hardware') or {}
        memory = hw.get('memory_bytes')
        memory = f"{int(memory) / 2**30:.0f} GB" if memory and str(memory).isdigit() else 'n/a'
        lines.append(f"- {name}: {meta.get('model')} @ {str(meta.get('resolved_revision') or meta.get('requested_revision'))[:12]}, "
                     f"device {meta.get('device')}, dtype {meta.get('dtype')}, torch {hw.get('torch')}, "
                     f"{hw.get('cpu', hw.get('machine', 'n/a'))}, RAM {memory}")
    lines += ['',
             '| Method | Users | Users with future purchases | nDCG@10 all | nDCG@10 nonempty | Runtime, s | Failed cases |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        wall = 'n/a' if r['wall_seconds'] is None else f"{r['wall_seconds']:.1f}"
        if r['method'] == 'candidate_order_first10':
            wall = 'n/a (no model; listed by the LLM runs)'
        elif r['method'] in BASELINES:
            wall += ' (whole run, 4 baselines)'
        lines.append(f"| {r['method']} | {r['users']} | {r['nonempty_users']} | {f4(r['ndcg_all'])} | "
                     f"{f4(r['ndcg_nonempty'])} | {wall} | {r['failed_cases']} |")
    lines += ['', '```json', json.dumps(analysis, indent=2), '```', '',
              f'Checks failed: {len(failures)}']
    lines += [f'- {m}' for m in failures[:50]]
    args.results_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return result


if __name__ == '__main__':
    main()
