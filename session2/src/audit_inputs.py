"""Independently rescan only needed pre-cutoff transactions and audit frozen prompts."""
import csv
from collections import Counter,defaultdict
from datetime import date
import json

from experiment import ROOT,dump,jsonl
from prepare_data import sha


def main():
    frozen=ROOT/'session2/outputs/frozen'
    requests=jsonl(frozen/'behavioral_requests.jsonl')
    protocol=json.loads((frozen/'behavioral_protocol.json').read_text())
    universe=set(protocol['eligible_assets'])
    selected=set(protocol['users'])
    counts=defaultdict(Counter)
    latest=defaultdict(dict)
    seen=set()
    max_date=''
    buyers=defaultdict(set)
    for row in csv.DictReader((ROOT/'session2/data/transactions.csv').open(encoding='utf-8-sig')):
        key=tuple(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        if int(row['transactionID'])<0 or row['ISIN'] not in universe or row['transactionType']!='Buy' or row['timestamp'][:10]>=protocol['cutoff']:
            continue
        uid,i,t=row['customerID'],row['ISIN'],row['timestamp'][:10]
        buyers[i].add(uid)
        if uid in selected:
            counts[uid][i]+=1
            latest[uid][i]=max(t,latest[uid].get(i,''))
            max_date=max(max_date,t)
    for r in requests:
        uid=r['user_id']
        p=r['payload']
        reverse=r['alias_to_id']
        assert len(p['candidates'])==len(set(r['candidates']))==50
        assert [reverse[c[0]] for c in p['candidates']]==r['candidates']
        assert {reverse[a]:n for a,n,age in p['history']}==dict(counts[uid])
        assert p['total_buy_events']==sum(counts[uid].values())
        assert p['distinct_purchased_assets']==len(counts[uid])
        for alias,score,rank,n,age,pop in p['candidates']:
            i=reverse[alias]
            assert n==counts[uid][i]
            assert pop==len(buyers[i])
            expected=(date.fromisoformat(protocol['cutoff'])-date.fromisoformat(latest[uid][i])).days if i in latest[uid] else None
            assert age==expected
        assert uid not in r['prompt']
        for i in reverse.values():
            assert i not in r['prompt']
        assert not any(k in r['prompt'] for k in ('assetName','riskLevel','investmentCapacity','customerID','"truth"'))
    snapshot=json.loads((ROOT/'session2/reference/teacher_snapshot.json').read_text())
    for f in snapshot['files']:
        assert sha(ROOT/'vendor/RecSys-LLMs'/f['path'])==f['sha256']
    dump(ROOT/'session2/evidence/input_audit.json',dict(performed_by='Codex AI',personal_confirmation='PENDING',
         users=100,candidates_per_user=50,history_counts_independently_rescanned=True,
         recency_and_popularity_independently_rescanned=True,strict_cutoff=protocol['cutoff'],max_selected_buy_date=max_date,
         customer_and_original_asset_identifiers_absent=True,semantic_fields_absent=True,
         teacher_source_hashes_verified=len(snapshot['files']),future_truth_path='outputs/frozen/evaluation.json only; not supplied to model worker'))
    print((ROOT/'session2/evidence/input_audit.json').read_text())


if __name__=='__main__':
    main()
