"""Independent saved-score audit. No torch import or inference."""
import csv,hashlib,importlib.util,json,math
from pathlib import Path

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    result=HERE/'results.json';r=json.loads(result.read_text());m=r['manifest'];hw2=HERE.parent/'semushkina_hw2'
    assert r['status']=='complete'
    contract={k:v for k,v in m.items() if k not in ('protocol_hash','preparation_seconds','backend')}
    calculated=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert calculated==m['protocol_hash']==r['protocol_hash']
    assert all(sha(hw2/name)==h for name,h in m['hw2_unchanged_hashes'].items())
    assert all(sha(HERE/name)==h for name,h in m['source_sha256'].items())
    old=json.loads((hw2/'results.json').read_text());frozen={x['user_id']:x for x in old['frozen']}
    requests={x['key']:x for x in map(json.loads,(HERE/'requests.jsonl').read_text().splitlines())}
    batches=list(map(json.loads,(HERE/'scores.jsonl').read_text().splitlines()));scores={}
    for batch in batches:
        assert batch['protocol_hash']==r['protocol_hash']
        for x in batch['scores']:
            assert x['key'] not in scores
            assert x['prompt_sha256']==requests[x['key']]['prompt_sha256']
            assert x['score']==x['logit_yes']-x['logit_no'] and math.isfinite(x['score'])
            scores[x['key']]=x
    assert len(scores)==2000
    for u in m['common']['users']:
        for k in range(10):
            bp=requests[f'{u}:{k}:behavioral']['prompt'];ep=requests[f'{u}:{k}:profile_enriched']['prompt']
            def split(text):
                before,rest=text.split('<Query>: ',1);query,doc=rest.split('\n<Document>: ',1)
                return before,json.loads(query),doc
            b,bq,bd=split(bp);e,eq,ed=split(ep)
            assert b==e and bd==ed
            assert {k:v for k,v in eq.items() if k not in ('segment','risk_tolerance','investment_capacity_band')}==bq
    spec=importlib.util.spec_from_file_location('teacher_eval',HERE.parent/'vendor/RecSys-LLMs/project/far-trans/starter.py')
    st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)
    for arm in ('behavioral','profile_enriched'):
        rows=[]
        for u in m['common']['users']:
            initial=frozen[u]['candidates'][:10]
            order=sorted(range(10),key=lambda k:(-scores[f'{u}:{k}:{arm}']['score'],k))
            prediction=[initial[k] for k in order];assert prediction==r['predictions'][arm][u]
            truth=set(frozen[u]['truth']);rows.append(dict(relevant_count=len(truth),ndcg_at_10=st.ndcg10(prediction,truth)))
        metric=st.aggregate(rows)
        for key,value in r['metrics'][arm].items():assert metric[key]==value
    check=r['personal_verification'];truth=set();seen=set();eligible=set(m['common']['eligible_assets'])
    with (hw2/'.local/data/transactions.csv').open() as f:
        for x in csv.DictReader(f):
            key=tuple(sorted(x.items()))
            if key in seen:continue
            seen.add(key)
            if x['customerID']==check['user_id'] and int(x['transactionID'])>=0 and x['transactionType']=='Buy' and x['ISIN'] in eligible and '2021-06-01'<=x['timestamp'][:10]<'2021-07-01':truth.add(x['ISIN'])
    assert truth==set(check['truth'])
    positives=sum(bool(set(x['candidates'][:10])&set(x['truth'])) for x in old['frozen'] if x['truth'])
    r['independent_verification']=dict(passed=True,score_records=2000,unique_users=100,
        teacher_evaluator_recomputed=True,only_three_profile_fields_differ=True,hw2_unchanged=True,
        personal_truth_checked_against_transactions=True,nonempty_users_with_retrieved_positive=positives,
        verification_code_sha256=sha(Path(__file__)))
    result.write_text(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    print('Verified 100 users, 2000 saved scores, matched ablation, protocol hash, teacher metrics and unchanged HW2.')

if __name__=='__main__':verify()
