"""HW3 evaluation: checks, main comparison, ablation and failure analysis.

Reads the starter baseline run (outputs/baseline), the two HW3 constrained runs
(outputs/llmrank_constrained, outputs/llmrank_constrained_masked) and the HW2
free-decoding runs (../selishchev_hw2/outputs/llmrank, .../llmrank_masked).
Writes outputs/evaluation/results.json (user-level, local only) and RESULTS.md
(aggregate only, safe to commit).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STARTER = ROOT / 'vendor' / 'RecSys-LLMs' / 'project' / 'far-trans' / 'starter.py'
BASELINES = ['popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell']
RUNS = [  # (method, folder, role)
    ('llmrank', ROOT / 'selishchev_hw2' / 'outputs' / 'llmrank', 'provided LLM ranker, free decoding (HW2 run)'),
    ('llmrank_masked', ROOT / 'selishchev_hw2' / 'outputs' / 'llmrank_masked', 'free decoding, names masked (HW2 run)'),
    ('llmrank_free', HERE / 'outputs' / 'llmrank_free', 'provided LLM ranker, free decoding, same device/dtype as HW3 runs'),
    ('llmrank_constrained', HERE / 'outputs' / 'llmrank_constrained', 'main: constrained decoding'),
    ('llmrank_constrained_masked', HERE / 'outputs' / 'llmrank_constrained_masked', 'ablation: constrained, names masked'),
]
DATE = re.compile(r'\d{4}-\d{2}-\d{2}')


def jload(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def jlines(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def predictions(path):
    out = defaultdict(dict)
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[row['method']].setdefault(row['user_id'], []).append((int(row['rank']), row['ISIN']))
    return {m: {u: [i for _, i in sorted(v)] for u, v in d.items()} for m, d in out.items()}


def ndcg10(ranking, truth):
    if not truth:
        return 0.0
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(ranking[:10]) if x in truth)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(10, len(truth))))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def bootstrap_ci(diffs, reps=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return [means[int(0.025 * reps)], means[int(0.975 * reps) - 1]]


def payload_of(prompt):
    instruction, body = prompt.split('\n', 1)
    return instruction, json.loads(body)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--results-md', type=Path, default=HERE / 'RESULTS.md')
    ap.add_argument('--outputs', type=Path, default=HERE / 'outputs')
    ap.add_argument('--runs-json', type=Path, help='override run folders (tests)')
    args = ap.parse_args(argv)
    runs_spec = RUNS if not args.runs_json else [tuple(x) for x in jload(args.runs_json)]
    base_dir, eval_dir = args.outputs / 'baseline', args.outputs / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)

    protocol = jload(base_dir / 'protocol.json')
    base_metrics = jload(base_dir / 'metrics.json')['methods']
    contexts = {c['user_id']: c for c in jlines(base_dir / 'contexts.jsonl')}
    users, cutoff, end = protocol['users'], protocol['cutoff'], protocol['end_exclusive']
    cands = {u: [a['id'] for a in contexts[u]['candidate_assets']] for u in users}
    eligible = set(protocol['eligible_assets'])
    truth, hist = defaultdict(set), defaultdict(set)
    uset = set(users)
    with open(args.data_dir / 'transactions.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['customerID'] in uset and r['transactionType'] == 'Buy' and int(r['transactionID']) >= 0:
                d = r['timestamp'][:10]
                if d < cutoff:
                    hist[r['customerID']].add(r['ISIN'])
                elif d < end and r['ISIN'] in eligible:
                    truth[r['customerID']].add(r['ISIN'])
    nonempty = [u for u in users if truth[u]]

    runs, prompts = {}, {}
    for method, folder, role in runs_spec:
        d = Path(folder)
        if not (d / 'metrics.json').exists():
            print(f'WARNING: {d} not complete; {method} reported as incomplete')
            continue
        man = jload(d / 'run_manifest.json')
        finals = {r['user_id']: r for r in jlines(d / 'final_responses.jsonl')}
        gens = jlines(d / 'generations.jsonl')[-len(users):]
        preds = predictions(d / 'predictions.csv')
        summ = jload(d / 'run_summary.json') if (d / 'run_summary.json').exists() else {}
        runs[method] = dict(dir=d, role=role, man=man, finals=finals, gens=gens, preds=preds, summ=summ,
                            metrics=jload(d / 'metrics.json'))
        check(man['users'] == users, f'{method}: users differ from protocol.json')
        check((man['cutoff'], man['end_exclusive']) == (cutoff, end), f'{method}: window differs')
        check(man['data_sha256'] == protocol['source_sha256'], f'{method}: data hashes differ')
        check(man['candidate_count'] == protocol['candidate_count'], f'{method}: candidate count differs')
        for b in BASELINES:
            check(abs(runs[method]['metrics'][b]['ndcg_at_10'] - base_metrics[b]['ndcg_at_10']) < 1e-12,
                  f'{method}: baseline {b} differs')
        check(list(finals) == users, f'{method}: final responses not one per user in order')
        prompts[method] = {}
        for uid, rec in zip(users, gens):
            prompts[method][uid] = rec['prompt']
            _, pay = payload_of(rec['prompt'])
            check([a['ISIN'] for a in pay['candidate_assets']] == cands[uid], f'{method}/{uid}: candidates differ')
            dates = DATE.findall(json.dumps([pay['profile_as_of_cutoff'], pay['buy_history_counts'],
                                             pay['recent_buy_events_oldest_to_newest'], pay['candidate_assets']]))
            check(all(x < cutoff for x in dates), f'{method}/{uid}: date on/after cutoff in prompt data')
            check(rec['response']['text'] == finals[uid]['raw_response'], f'{method}/{uid}: log != final response')
        # re-score through the unchanged starter under the same protocol hash
        resp = eval_dir / f'{method}_responses.jsonl'
        with resp.open('w', encoding='utf-8') as s:
            for uid in users:
                s.write(json.dumps({'user_id': uid, 'stage': 'ranker', 'protocol_hash': protocol['protocol_hash'],
                                    'ranked_ids': preds[method][uid]}) + '\n')
        out = eval_dir / f'starter_{method}'
        subprocess.run([sys.executable, str(STARTER), 'run', '--data-dir', str(args.data_dir), '--output-dir', str(out),
                        '--stage', 'ranker', '--responses', str(resp), '--on-invalid', 'error'],
                       check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, PYTHONHASHSEED='0'))
        sm = jload(out / 'metrics.json')['methods']['custom']
        check(jload(out / 'protocol.json')['protocol_hash'] == protocol['protocol_hash'], f'{method}: protocol hash differs')
        check(abs(sm['ndcg_at_10'] - runs[method]['metrics'][method]['ndcg_at_10']) < 1e-12, f'{method}: starter re-score differs')

    # the constrained runs must use exactly the prompts of the matching free runs
    for a, b in (('llmrank', 'llmrank_constrained'), ('llmrank_masked', 'llmrank_constrained_masked'),
                 ('llmrank_free', 'llmrank_constrained')):
        if a in prompts and b in prompts:
            check(all(prompts[a][u] == prompts[b][u] for u in users), f'{b}: prompts differ from {a}')
    for a, b in (('llmrank_constrained', 'llmrank_constrained_masked'),):
        if a in prompts and b in prompts:
            for u in users:
                ia, pa = payload_of(prompts[a][u])
                ib, pb = payload_of(prompts[b][u])
                for sec in ('buy_history_counts', 'candidate_assets'):
                    for row in pa[sec]:
                        row['assetName'] = 'MASKED'
                check(ia == ib and pa == pb, f'{u}: masked prompt differs beyond assetName')

    allp = predictions(base_dir / 'predictions.csv')
    allp['candidate_order_first10'] = {u: cands[u][:10] for u in users}
    for m, r in runs.items():
        allp[m] = r['preds'][m]
        check(r['preds']['candidate_order_first10'] == allp['candidate_order_first10'], f'{m}: candidate order differs')
        rep = {(x['user_id'], x['method']): x for x in jload(r['dir'] / 'per_user.json')}
        for u in users:
            check(rep[(u, m)]['relevant_count'] == len(truth[u]), f'{u}: truth count differs')
            check(abs(rep[(u, m)]['ndcg_at_10'] - ndcg10(allp[m][u], truth[u])) < 1e-12, f'{m}/{u}: nDCG differs')
    score = {m: {u: ndcg10(by[u], truth[u]) for u in users} for m, by in allp.items()}

    rows = []
    base_time = jload(base_dir / 'run_summary.json').get('wall_seconds_all_four_baselines_one_run') \
        if (base_dir / 'run_summary.json').exists() else None
    for m in BASELINES + ['candidate_order_first10'] + list(runs):
        r = runs.get(m)
        rt = r['man']['runtime'] if r else {}
        s = r['summ'] if r else {}
        rows.append({'method': m, 'role': r['role'] if r else ('control: initial candidate order' if m.startswith('cand') else 'provided baseline'),
                     'users': len(users), 'nonempty_users': len(nonempty),
                     'ndcg_all': mean(score[m].values()), 'ndcg_nonempty': mean(score[m][u] for u in nonempty),
                     'invalid': rt.get('raw_invalid_count', 0), 'repairs': rt.get('parse_repair_count', 0),
                     'fallbacks': sum(1 for u in users if r and r['finals'][u]['fallback']) if r else 0,
                     'generation_seconds': (s.get('generations_this_run') or {}).get('generation_seconds_new_calls'),
                     'run_seconds': s.get('recommendation_generation_seconds', s.get('wall_seconds_including_data_and_model_load')),
                     'preparation': s.get('preparation_seconds'),
                     'baseline_run_seconds': base_time if m in BASELINES else None})

    def paired(a, b, pool):
        diffs = [score[a][u] - score[b][u] for u in pool]
        return {'mean_diff': mean(diffs), 'ci95': bootstrap_ci(diffs) if diffs else None,
                'wins': sum(d > 0 for d in diffs), 'ties': sum(d == 0 for d in diffs), 'losses': sum(d < 0 for d in diffs)}

    analysis = {}
    for m, r in runs.items():
        valid = [u for u in users if r['finals'][u]['error'] is None]
        inter = [r['gens'][i]['response'].get('constraint_interventions') for i in range(len(users))]
        top = allp[m]
        analysis[m] = {
            'valid_users': len(valid), 'valid_nonempty': sum(u in nonempty for u in valid),
            'error_types': dict(Counter(r['finals'][u]['error'] for u in users if r['finals'][u]['error'])),
            'users_with_interventions': sum(1 for x in inter if x) if inter[0] is not None else None,
            'interventions_total': sum(x or 0 for x in inter) if inter[0] is not None else None,
            'top10_equals_candidate_order': sum(top[u] == cands[u][:10] for u in users),
            'valid_top10_equals_candidate_order': sum(top[u] == cands[u][:10] for u in valid),
            'mean_overlap_with_candidate_top10_valid': mean(len(set(top[u]) & set(cands[u][:10])) for u in valid),
            'mean_previously_bought_in_top10': mean(len(set(top[u]) & hist[u]) for u in users),
            'vs_candidate_order_all': paired(m, 'candidate_order_first10', users),
            'vs_candidate_order_nonempty': paired(m, 'candidate_order_first10', nonempty),
        }
    analysis['candidate_order_first10'] = {'mean_previously_bought_in_top10': mean(len(set(cands[u][:10]) & hist[u]) for u in users)}
    for a, b in (('llmrank_constrained', 'llmrank'), ('llmrank_constrained', 'llmrank_free'),
                 ('llmrank_free', 'llmrank'), ('llmrank_constrained_masked', 'llmrank_constrained'),
                 ('llmrank_constrained_masked', 'llmrank_masked')):
        if a in runs and b in runs:
            analysis[f'{a}_vs_{b}'] = dict(paired(a, b, users),
                                           nonempty=paired(a, b, nonempty),
                                           identical_top10=sum(allp[a][u] == allp[b][u] for u in users),
                                           mean_top10_overlap=mean(len(set(allp[a][u]) & set(allp[b][u])) for u in users))
    for b in BASELINES:
        if 'llmrank_constrained' in runs:
            analysis[f'llmrank_constrained_vs_{b}'] = paired('llmrank_constrained', b, users)
    ceiling = [u for u in nonempty if truth[u] & set(cands[u])]
    analysis['coverage'] = {
        'nonempty_users': len(nonempty),
        'nonempty_with_truth_in_50_candidates': len(ceiling),
        'nonempty_with_truth_in_candidate_top10': sum(1 for u in nonempty if truth[u] & set(cands[u][:10])),
        'future_purchases': sum(len(truth[u]) for u in nonempty),
        'future_purchases_previously_bought': sum(len(truth[u] & hist[u]) for u in nonempty),
        'perfect_rerank_ndcg_all': mean(ndcg10(sorted(cands[u], key=lambda i: i not in truth[u]), truth[u]) for u in users)}
    per_user = {}
    for u in nonempty:
        row = {'relevant': len(truth[u]), 'in_candidates': sorted(truth[u] & set(cands[u])),
               'candidate_positions': [cands[u].index(i) + 1 for i in cands[u] if i in truth[u]]}
        for m in allp:
            row[m] = {'ndcg': score[m][u], 'hit_ranks': [k + 1 for k, i in enumerate(allp[m][u]) if i in truth[u]]}
            if m in runs:
                row[m]['valid'] = runs[m]['finals'][u]['error'] is None
        per_user[u] = row
    result = {'protocol_hash': protocol['protocol_hash'], 'cutoff': cutoff, 'end_exclusive': end,
              'checks_failed': fails, 'rows': rows, 'analysis': analysis, 'per_nonempty_user': per_user,
              'runtime_metadata': {m: r['man']['runtime'] for m, r in runs.items()},
              'run_summaries': {m: {k: r['summ'].get(k) for k in ('preparation_seconds', 'recommendation_generation_seconds',
                                                                   'wall_seconds_including_data_and_model_load',
                                                                   'generations_this_run', 'hardware', 'command')}
                                for m, r in runs.items()}}
    (eval_dir / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    f4 = lambda x: 'n/a' if x is None else f'{x:.4f}'
    lines = ['# HW3 development results (aggregate only; not official benchmark scores)', '',
             f'Protocol hash `{protocol["protocol_hash"]}`; 100-user teaching sample; cutoff {cutoff}; window to {end} (exclusive).', '',
             '| Method | Role | Users | Nonempty | nDCG@10 all | nDCG@10 nonempty | Invalid | Repairs | Fallbacks | Generation s |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        g = r['generation_seconds'] if r['generation_seconds'] is not None else r['baseline_run_seconds']
        lines.append(f"| {r['method']} | {r['role']} | {r['users']} | {r['nonempty_users']} | {f4(r['ndcg_all'])} | "
                     f"{f4(r['ndcg_nonempty'])} | {r['invalid']} | {r['repairs']} | {r['fallbacks']} | "
                     f"{'n/a' if g is None else f'{g:.1f}'} |")
    for m, r in runs.items():
        rt = r['man']['runtime']
        lines.append(f"- {m}: {rt.get('model')} @ {str(rt.get('resolved_revision'))[:12]}, device {rt.get('device')}, "
                     f"dtype {rt.get('dtype')}, decoding {rt.get('decoding', 'greedy_free')}")
    lines += ['', '```json', json.dumps(analysis, indent=2), '```', '', f'Checks failed: {len(fails)}']
    lines += [f'- {x}' for x in fails[:50]]
    args.results_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return result


if __name__ == '__main__':
    main()
