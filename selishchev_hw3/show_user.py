"""Print the raw material for a by-hand check of ONE user (no nDCG is computed here).

Usage: python show_user.py --data-dir /path/to/FAR-Trans --method llmrank_masked [--user ID]
Without --user, lists the users who have future purchases so you can pick one.
"""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=Path, required=True)
parser.add_argument('--method', default='llmrank_constrained')
parser.add_argument('--user')
args = parser.parse_args()
run = HERE / 'outputs' / args.method
if not run.exists():  # HW2 free-decoding runs
    run = HERE.parent / 'selishchev_hw2' / 'outputs' / args.method
per_user = json.loads((run / 'per_user.json').read_text())
if not args.user:
    print('Users with future purchases (relevant_count > 0):')
    for r in per_user:
        if r['method'] == args.method and r['relevant_count']:
            print(f"  {r['user_id']}  relevant_count={r['relevant_count']}")
    raise SystemExit
print(f'1) Top-10 of {args.method} for {args.user} (outputs/{args.method}/predictions.csv):')
with open(run / 'predictions.csv', newline='') as f:
    for row in csv.DictReader(f):
        if row['method'] == args.method and row['user_id'] == args.user:
            print(f"   rank {row['rank']:>2}  {row['ISIN']}")
print('\n1b) Initial candidate order, first 10 (the control), for comparison:')
with open(run / 'predictions.csv', newline='') as f:
    for row in csv.DictReader(f):
        if row['method'] == 'candidate_order_first10' and row['user_id'] == args.user:
            print(f"   rank {row['rank']:>2}  {row['ISIN']}")
print('\n2) Raw rows of transactions.csv for this user dated 2021-06-01 .. 2021-06-30:')
with open(args.data_dir / 'transactions.csv', newline='') as f:
    for line in f:
        if args.user in line and ',2021-06-' in line:
            print('   ' + line.rstrip())
print('\n3) Dates that appear in the prompt sent to the model for this user:')
users = json.loads((run / 'run_manifest.json').read_text())['users']
gens = [json.loads(l) for l in (run / 'generations.jsonl').read_text().splitlines() if l.strip()][-len(users):]
prompt = gens[users.index(args.user)]['prompt']
payload = json.loads(prompt.split('\n', 1)[1])
dates = sorted({e['date'] for e in payload['recent_buy_events_oldest_to_newest']})
print('   recent Buy dates:', ', '.join(dates) or 'none')
print('   profile timestamp:', payload['profile_as_of_cutoff'].get('timestamp'))
print('   asset names in prompt:', sorted({a['assetName'] for a in payload['candidate_assets']})[:3], '...')
print('\n4) Value reported by the pipeline (compare with your own calculation):')
for r in per_user:
    if r['user_id'] == args.user and r['method'] in (args.method, 'candidate_order_first10'):
        print(f"   {r['method']}: relevant_count={r['relevant_count']} nDCG@10={r['ndcg_at_10']:.4f}")
