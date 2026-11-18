"""Freeze the unchanged teaching retrieval; prompts and truth are separate artifacts."""
import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from prepare_data import ROOT, sha
TEACHER = ROOT/'vendor/RecSys-LLMs/project'
sys.path.insert(0, str(TEACHER/'far-trans-llm'))
import data_adapter
import ranker as teacher_ranker


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def behavioral_prompt(context, history, events, scores, pop):
    """No truth, profiles, asset records, names or raw IDs can enter the payload."""
    cutoff = context['cutoff']
    candidates = teacher_ranker.candidate_ids(context)
    if len(candidates) != 50:
        raise ValueError('Full 50-candidate pool required')
    historical = [(stamp, isin) for stamp, isin, kind in events
                  if kind == 'Buy' and stamp[:10] < cutoff]
    counts = {}
    latest = {}
    for stamp, isin in historical:
        counts[isin] = counts.get(isin, 0)+1
        latest[isin] = max(latest.get(isin, ''), stamp)
    if counts != dict(history):
        raise ValueError('Raw retained event counts differ from teacher history')
    ids = sorted(set(candidates)|set(counts), key=lambda i: (digest(['behavioral-v1',i]),i))
    forward = {isin:f'A{index:03d}' for index,isin in enumerate(ids,1)}
    reverse = {alias:isin for isin,alias in forward.items()}
    age = lambda stamp: (date.fromisoformat(cutoff)-date.fromisoformat(stamp[:10])).days
    payload = dict(cutoff_exclusive=cutoff, repeat_purchases_allowed=True,
                   distinct_purchased_assets=len(counts), total_buy_events=sum(counts.values()),
                   history_columns=['alias','buy_count','last_buy_age_days'],
                   history=[[forward[i],counts[i],age(latest[i])] for i in ids if i in counts],
                   recent_15_buys_oldest_first=[[forward[i],age(t)] for t,i in sorted(historical)[-15:]],
                   candidate_columns=['alias','cbf_score','original_rank','buy_count','last_buy_age_days','distinct_buyers'],
                   candidates=[[forward[i],float(f'{scores[i]:.4g}'),r,counts.get(i,0),
                                age(latest[i]) if i in latest else None,pop[i]]
                               for r,i in enumerate(candidates,1)])
    prompt = (ROOT/'session2/prompts/behavioral.txt').read_text().strip()+'\n'+json.dumps(payload,separators=(',',':'))
    return prompt,reverse,payload


def parse_and_fallback(raw, allowed, reverse=None, status='generated'):
    """Failure types stay visible even though every user receives a valid list."""
    result = dict(status=status, valid=False, parse_repaired=False, fallback=True, error=None)
    try:
        if status != 'generated':
            raise RuntimeError(status)
        ranking,repaired = teacher_ranker.parse_ranking(raw,set(reverse) if reverse else set(allowed))
        if reverse:
            ranking = [reverse[a] for a in ranking]
        if len(set(ranking)) != 10 or not set(ranking)<=set(allowed):
            raise ValueError('IDs outside candidate pool')
        result.update(valid=True,parse_repaired=repaired,fallback=False,ranking=ranking)
    except (ValueError,TypeError,KeyError,RuntimeError) as exc:
        result.update(error=f'{type(exc).__name__}: {exc}',ranking=allowed[:10])
    return result


def freeze():
    folder=ROOT/'session2/outputs/frozen'
    if folder.exists():
        raise FileExistsError('Frozen input already exists; choose a new experiment instead of overwriting')
    a=SimpleNamespace(max_users=100,horizon_days=30,candidate_count=50,cutoff='2021-06-01',
                      starter_dir=TEACHER/'far-trans',data_dir=ROOT/'session2/data')
    st,assets,history,truth,contexts,baselines,events,pop,manifest=data_adapter.load(a)
    baseline_protocol=json.loads((ROOT/'session2/outputs/baselines_seed0/protocol.json').read_text())
    assert manifest['users']==baseline_protocol['users']
    assert manifest['data_sha256']==baseline_protocol['source_sha256']
    assert sorted(assets)==baseline_protocol['eligible_assets']
    original={}
    with (ROOT/'session2/outputs/baselines_seed0/predictions.csv').open() as f:
        for row in csv.DictReader(f):
            original.setdefault(row['user_id'],{}).setdefault(row['method'],[]).append(row['ISIN'])
    assert original==baselines
    folder.mkdir(parents=True)
    features={i:st.tokens(asset) for i,asset in assets.items()}
    requests={'behavioral':[],'semantic':[]}
    evaluation={}
    for uid in manifest['users']:
        context=contexts[uid]
        candidates=teacher_ranker.candidate_ids(context)
        scores=st.content_scores(history[uid],features)
        if not any(scores.values()):
            scores=pop
        prompt,reverse,payload=behavioral_prompt(context,history[uid],events.get(uid,()),scores,pop)
        requests['behavioral'].append(dict(user_id=uid,prompt=prompt,alias_to_id=reverse,candidates=candidates,
                                           payload=payload,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        semantic=teacher_ranker.build_prompt(context,assets,history[uid],events.get(uid,()),15,None)
        requests['semantic'].append(dict(user_id=uid,prompt=semantic,candidates=candidates,
                                         prompt_sha256=hashlib.sha256(semantic.encode()).hexdigest()))
        evaluation[uid]=dict(truth=sorted(truth.get(uid,set())&set(assets)),
                             distinct_history=len(history[uid]),baselines=baselines[uid])
    for arm,records in requests.items():
        (folder/f'{arm}_requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    dump(folder/'evaluation.json',evaluation)
    for arm in requests:
        protocol=dict(baseline_protocol,arm=arm,candidate_source='Unchanged teacher CBF Top-50',
                      candidates_sha256=digest([(r['user_id'],r['candidates']) for r in requests[arm]]),
                      common_protocol_hash=baseline_protocol['protocol_hash'],
                      teacher_commit=json.loads((ROOT/'session2/reference/teacher_snapshot.json').read_text())['commit'])
        protocol.pop('protocol_hash')
        protocol['protocol_hash']=digest(protocol)
        dump(folder/f'{arm}_protocol.json',protocol)
    dump(folder/'manifest.json',dict(files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
                                     users=100,nonempty=sum(bool(v['truth']) for v in evaluation.values()),
                                     subgroup_1_to_3=sum(1<=v['distinct_history']<=3 for v in evaluation.values()),
                                     selection='teacher SHA256 42 ordering; no truth-based filtering'))
    print((folder/'manifest.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['freeze'])
    freeze()
