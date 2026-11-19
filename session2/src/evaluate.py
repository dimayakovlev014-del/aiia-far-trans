"""Evaluate saved outputs independently, retaining empty truth and every fallback."""
import csv
import json
import math
from pathlib import Path
import statistics

from experiment import ROOT,dump,digest,jsonl,parse_and_fallback
from prepare_data import sha


def ndcg(ranking,truth):
    ideal=sum(1/math.log2(rank+1) for rank in range(1,min(10,len(truth))+1))
    dcg=sum(1/math.log2(rank+1) for rank,isin in enumerate(ranking[:10],1) if isin in truth)
    return dcg/ideal if ideal else 0.0


def aggregate(records):
    nonempty=[r for r in records if r['relevant_count']]
    return dict(users=len(records),nonempty_truth_users=len(nonempty),
                ndcg_at_10=sum(r['ndcg_at_10'] for r in records)/len(records) if records else None,
                ndcg_at_10_nonempty=sum(r['ndcg_at_10'] for r in nonempty)/len(nonempty) if nonempty else None)


def main():
    frozen=ROOT/'session2/outputs/frozen'
    folder=ROOT/'session2/outputs/comparison'
    if folder.exists():
        raise FileExistsError('Preserve comparison artifacts; choose a new directory for corrections')
    evaluation=json.loads((frozen/'evaluation.json').read_text())
    behavioral=jsonl(frozen/'behavioral_requests.jsonl')
    semantic=jsonl(frozen/'semantic_requests.jsonl')
    users=list(evaluation)
    assert users==[r['user_id'] for r in behavioral]==[r['user_id'] for r in semantic]
    assert all(b['candidates']==s['candidates'] and len(b['candidates'])==50 for b,s in zip(behavioral,semantic))
    requests={r['user_id']:r for r in behavioral}
    predictions={u:dict(evaluation[u]['baselines'],candidate_order=requests[u]['candidates'][:10]) for u in users}
    responses={}
    coverage={}
    protocol_alignment={}
    baseline=json.loads((ROOT/'session2/outputs/baselines_seed0/protocol.json').read_text())
    for name,expected in baseline['source_sha256'].items():
        assert sha(ROOT/'session2/data'/name)==expected
    snapshot=json.loads((ROOT/'session2/reference/teacher_snapshot.json').read_text())
    for entry in snapshot['files']:
        assert sha(ROOT/'vendor/RecSys-LLMs'/entry['path'])==entry['sha256']
    for arm in ('behavioral','semantic'):
        path=ROOT/f'session2/outputs/{arm}_full/responses.jsonl'
        if not path.exists():
            if arm=='behavioral':
                raise RuntimeError('Required 100-user model run missing')
            continue
        rows=jsonl(path)
        assert [r['user_id'] for r in rows]==users
        protocol=json.loads(path.with_name('protocol.json').read_text())
        assert protocol['inference_model']==json.loads((ROOT/'session2/configs/model.json').read_text())
        assert protocol['decoding']==json.loads((ROOT/'session2/configs/decoding.json').read_text())
        for k,v in baseline.items():
            if k!='protocol_hash':
                assert protocol[k]==v,(arm,k)
        saved_hash=protocol.pop('protocol_hash')
        assert digest(protocol)==saved_hash
        frozen_protocol=json.loads((frozen/f'{arm}_protocol.json').read_text())
        summary=json.loads(path.with_name('summary.json').read_text())
        assert summary['protocol_hash']==frozen_protocol['protocol_hash']
        protocol_alignment[arm]=dict(run_protocol_hash=saved_hash,
             frozen_input_protocol_hash=frozen_protocol['protocol_hash'],
             common_baseline_protocol_hash=baseline['protocol_hash'],
             raw_summary_protocol_hash_meaning='Frozen input protocol, not the model/backend-extended run protocol',
             common_contract_matches=True,model_and_decoding_match_summary=protocol['inference_model']==summary['model'] and protocol['decoding']==summary['decoding'])
        assert protocol_alignment[arm]['model_and_decoding_match_summary']
        for r,request in zip(rows,behavioral if arm=='behavioral' else semantic):
            reconstructed=parse_and_fallback(r['response'].get('text'),request['candidates'],request.get('alias_to_id'),r['response']['status'])
            for field in ('valid','fallback','ranking','parse_repaired','error','status'):
                assert reconstructed[field]==r[field],(arm,field)
            predictions[r['user_id']][arm]=r['ranking']
        responses[arm]=rows
        unique_responses={r['cache_key']:r['response'] for r in rows}
        coverage[arm]=dict(users=len(rows),generated=sum(r['status']=='generated' for r in rows),
                           valid=sum(r['valid'] for r in rows),fallbacks=sum(r['fallback'] for r in rows),
                           parse_repairs=sum(r['parse_repaired'] for r in rows),
                           cache_hits=sum(r['cache_hit'] for r in rows),
                           failure_types={k:sum((r['status'] if r['status']!='generated' else 'invalid_response')==k and not r['valid'] for r in rows)
                                          for k in sorted({r['status'] if r['status']!='generated' else 'invalid_response' for r in rows if not r['valid']})},
                           validation_reasons={reason:sum(r['error']==reason for r in rows)
                                               for reason in sorted({r['error'] for r in rows if r['error']})},
                           finish_reasons={reason:sum(r['response'].get('finish_reason','no_generation')==reason for r in rows)
                                           for reason in sorted({r['response'].get('finish_reason','no_generation') for r in rows})},
                           unique_model_requests=len(unique_responses),
                           generation_seconds=sum(r.get('inference_seconds',0) for r in unique_responses.values()),
                           generation_seconds_user_sum=sum(r['response'].get('inference_seconds',0) for r in rows),
                           peak_worker_memory_bytes=max(r['response'].get('peak_process_memory_bytes',0) for r in rows),
                           median_generation_seconds=statistics.median(r['response'].get('inference_seconds',0) for r in rows),
                           median_served_request_seconds=statistics.median(r['request_wall_seconds'] for r in rows))
    methods=list(predictions[users[0]])
    per_user=[]
    for uid in users:
        truth=set(evaluation[uid]['truth'])
        for method,ranking in predictions[uid].items():
            assert len(ranking)==len(set(ranking))==10
            if method in ('behavioral','semantic','candidate_order'):
                assert set(ranking)<=set(requests[uid]['candidates'])
            per_user.append(dict(user_id=uid,method=method,relevant_count=len(truth),
                                 ndcg_at_10=ndcg(ranking,truth),short_history=1<=evaluation[uid]['distinct_history']<=3))
    metrics={m:aggregate([r for r in per_user if r['method']==m]) for m in methods}
    subgroup={m:aggregate([r for r in per_user if r['method']==m and r['short_history']]) for m in methods}
    for arm,rows in responses.items():
        valid={r['user_id'] for r in rows if r['valid']}
        coverage[arm]['valid_only_diagnostic']=aggregate([r for r in per_user if r['method']==arm and r['user_id'] in valid])
        coverage[arm]['changed_top10_users']=sum(r['ranking']!=requests[r['user_id']]['candidates'][:10] for r in rows)
        if arm=='behavioral':
            ascending=0
            for r in rows:
                if r['valid']:
                    forward={i:a for a,i in requests[r['user_id']]['alias_to_id'].items()}
                    aliases=[forward[i] for i in r['ranking']]
                    ascending+=aliases==sorted(aliases)
            coverage[arm]['ascending_alias_order_valid_users']=ascending
    reference=json.loads((ROOT/'session2/outputs/baselines_seed0/metrics.json').read_text())['methods']
    for m in reference:
        for key in ('ndcg_at_10','ndcg_at_10_nonempty'):
            assert math.isclose(reference[m][key],metrics[m][key],abs_tol=1e-14)
    assert all(predictions[u]['cbf']==predictions[u]['candidate_order'] for u in users)
    truth_count=sum(len(evaluation[u]['truth']) for u in users)
    hits=sum(len(set(requests[u]['candidates'])&set(evaluation[u]['truth'])) for u in users)
    recalls=[len(set(requests[u]['candidates'])&set(evaluation[u]['truth']))/len(evaluation[u]['truth']) for u in users if evaluation[u]['truth']]
    oracle=[]
    for u in users:
        truth=set(evaluation[u]['truth'])
        available=[i for i in requests[u]['candidates'] if i in truth]
        other=[i for i in requests[u]['candidates'] if i not in truth]
        oracle.append(dict(relevant_count=len(truth),ndcg_at_10=ndcg((available+other)[:10],truth)))
    retrieval=dict(total_positives=truth_count,retrieved_positives=hits,missed_positives=truth_count-hits,
                    recall50_micro=hits/truth_count,recall50_nonempty_macro=sum(recalls)/len(recalls),
                    nonempty_users_with_retrieved_positive=sum(bool(set(requests[u]['candidates'])&set(evaluation[u]['truth'])) for u in users),
                    candidate_oracle_ndcg10=aggregate(oracle))
    folder.mkdir(parents=True)
    dump(folder/'metrics.json',metrics)
    dump(folder/'subgroup.json',subgroup)
    dump(folder/'coverage.json',coverage)
    dump(folder/'retrieval.json',retrieval)
    dump(folder/'per_user.json',per_user)
    dump(folder/'predictions.json',predictions)
    dump(folder/'checks.json',dict(cohort_equality=True,candidate_equality=True,baseline_recalculation=True,
                                   candidate_order_equals_cbf=True,protocol_common_contract=True,
                                   full_protocol_hashes_verified=True,raw_response_reparse=True,
                                   teacher_hashes_intact=True,users=100,nonempty=11,
                                   source_files={p.name:sha(p) for p in frozen.iterdir() if p.is_file()}))
    dump(folder/'protocol_alignment.json',protocol_alignment)
    # A nonempty, nonzero saved example; selection is diagnostic, never tuning.
    valid_users={r['user_id'] for r in responses['behavioral'] if r['valid']}
    positive=[r for r in per_user if r['method']=='behavioral' and r['relevant_count'] and r['ndcg_at_10']>0]
    example=next((r for r in positive if r['user_id'] in valid_users),
                 positive[0] if positive else next(r for r in per_user if r['method']=='behavioral' and r['relevant_count']))
    uid=example['user_id']
    example_response=next(r for r in responses['behavioral'] if r['user_id']==uid)
    ranking=predictions[uid]['behavioral']
    truth=set(evaluation[uid]['truth'])
    relevant_ranks=[r for r,i in enumerate(ranking,1) if i in truth]
    dcg=sum(1/math.log2(r+1) for r in relevant_ranks)
    idcg=sum(1/math.log2(r+1) for r in range(1,min(10,len(truth))+1))
    line=users.index(uid)+1
    text=f'''# Session 2 personal check — PENDING

This calculation was performed by Codex (AI), not personally by Dmitry. Dmitry's confirmation remains PENDING. No Session 1 date check substitutes for this result check.

Saved user: `{uid}` (JSONL line {line}).

Response status: `{example_response['status']}`; valid model ranking: `{example_response['valid']}`; original-order fallback: `{example_response['fallback']}`.

- Input + alias map: `session2/outputs/frozen/behavioral_requests.jsonl`, line {line}.
- Raw generation, status and mapped ranking: `session2/outputs/behavioral_full/responses.jsonl`, line {line}.
- Truth: `session2/outputs/frozen/evaluation.json`, key `{uid}`, field `truth`.
- Independent score: `session2/outputs/comparison/per_user.json`, user `{uid}`, method `behavioral`.

Future relevant IDs: `{sorted(truth)}`.

| Rank | Predicted original ID | Relevant | Discounted gain |
|---|---|---|---|
'''
    for r,i in enumerate(ranking,1):
        text+=f'| {r} | {i} | {int(i in truth)} | {(1/math.log2(r+1) if i in truth else 0):.10f} |\n'
    text+=f'''\nRelevant ranks: {relevant_ranks}. DCG@10 = {dcg:.10f}. IDCG@10 uses all {len(truth)} future eligible positives, including any outside the candidate pool: {idcg:.10f}. nDCG@10 = DCG/IDCG = {example['ndcg_at_10']:.10f} ({example['ndcg_at_10']:.4f}).

Dmitry should inspect those saved files, verify the relevant ranks and the displayed division, and state whether this is the check he personally performed. No confirmation is recorded yet.
'''
    (ROOT/'session2/evidence/personal_check.md').write_text(text)
    dump(ROOT/'session2/evidence/personal_check.json',dict(status='PENDING',performed_by='Codex AI',user_id=uid,
                                                         jsonl_line=line,ranking=ranking,truth=sorted(truth),
                                                         model_valid=example_response['valid'],fallback=example_response['fallback'],
                                                         relevant_ranks=relevant_ranks,dcg=dcg,idcg=idcg,ndcg=example['ndcg_at_10']))
    failures=[(arm,r) for arm,rows in responses.items() for r in rows if r['fallback']]
    if failures:
        arm,r=failures[0]
        dump(ROOT/'session2/evidence/inspected_failure.json',dict(arm=arm,record=r,
              source=f'session2/outputs/{arm}_full/responses.jsonl',
              observation='AI inspected saved raw output and validation reason; original CBF first ten retained'))
    print(json.dumps(dict(metrics=metrics,subgroup=subgroup,coverage=coverage,retrieval=retrieval),indent=2))


if __name__=='__main__':
    main()
