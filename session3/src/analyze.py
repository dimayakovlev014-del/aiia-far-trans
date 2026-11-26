"""Audit all comparisons; retain raw precision and personal-check status PENDING."""
from collections import Counter
import csv
import importlib.util
import json
import math
import statistics
from common import ROOT, S3, aggregate, digest, dump, jsonl, ndcg, parse_and_fallback, read, sha


def independent(ranking, truth):
    # Separate numerator/denominator implementation, also compared with teacher/HW8.
    gains = [int(asset in truth)/math.log(position+2,2) for position,asset in enumerate(ranking[:10])]
    ideal_gains = [1/math.log(position+2,2) for position in range(min(10,len(truth)))]
    numerator, denominator = math.fsum(gains), math.fsum(ideal_gains)
    return dict(dcg=numerator, idcg=denominator, ndcg_at_10=numerator/denominator if denominator else 0.0,
                relevant_ranks=[position+1 for position,asset in enumerate(ranking[:10]) if asset in truth])


def metrics_for(users, predictions, evaluation, method):
    return aggregate([dict(relevant_count=len(evaluation[u]['truth']),
                           ndcg_at_10=ndcg(predictions[u][method],set(evaluation[u]['truth']))) for u in users])


def pair(users, predictions, evaluation, left, right):
    # Positive delta means right improves on left. Ordered Top-10 is the full output.
    rows = []
    for u in users:
        a, b = predictions[u][left], predictions[u][right]
        x, y = ndcg(a,set(evaluation[u]['truth'])), ndcg(b,set(evaluation[u]['truth']))
        rows.append(dict(user_id=u,delta=y-x,left_ndcg=x,right_ndcg=y,changed_order=a!=b,
                         same_set=set(a)==set(b),jaccard=len(set(a)&set(b))/len(set(a)|set(b))))
    return dict(users=len(rows),nonempty_truth_users=sum(bool(evaluation[u]['truth']) for u in users),
                mean_paired_delta=math.fsum(r['delta'] for r in rows)/len(rows) if rows else None,
                improved=sum(r['delta']>0 for r in rows),worsened=sum(r['delta']<0 for r in rows),
                tied=sum(r['delta']==0 for r in rows),changed_top10_order=sum(r['changed_order'] for r in rows),
                exact_top10_order=sum(not r['changed_order'] for r in rows),
                exact_top10_set=sum(r['same_set'] for r in rows),
                mean_top10_jaccard=statistics.mean(r['jaccard'] for r in rows) if rows else None), rows


def main():
    out = S3/'outputs/analysis'
    if out.exists():
        raise FileExistsError(out)
    frozen = S3/'outputs/frozen'
    manifest = read(frozen/'manifest.json')
    for name, expected in manifest['files'].items():
        assert sha(frozen/name) == expected
    evaluation = read(frozen/'evaluation.json')
    users = list(evaluation)
    requests = {arm:jsonl(frozen/(arm+'_requests.jsonl')) for arm in ('main','ablation')}
    assert all([r['user_id'] for r in rows] == users for rows in requests.values())
    assert all(m['candidates']==a['candidates'] for m,a in zip(requests['main'],requests['ablation']))
    baseline_path = ROOT/'session2/outputs/baselines_seed0'
    baseline = read(baseline_path/'protocol.json')
    assert baseline['users'] == users
    assert digest({k:v for k,v in baseline.items() if k!='protocol_hash'}) == baseline['protocol_hash']
    original = {}
    for row in csv.DictReader((baseline_path/'predictions.csv').open()):
        original.setdefault(row['user_id'],{}).setdefault(row['method'],[]).append(row['ISIN'])
    assert original == {u:evaluation[u]['baselines'] for u in users}
    predictions = {u:dict(original[u],candidate_order=requests['main'][i]['candidates'][:10]) for i,u in enumerate(users)}
    assert all(predictions[u]['cbf']==predictions[u]['candidate_order'] for u in users)
    model = read(ROOT/'session2/configs/model.json')
    config = read(ROOT/'session2/configs/decoding.json')
    responses, summaries, protocols = {}, {}, {}
    proof_files = [baseline_path/'predictions.csv',baseline_path/'protocol.json',baseline_path/'metrics.json']
    for arm in ('main','ablation','semantic'):
        folder = ROOT/'session2/outputs/semantic_full' if arm=='semantic' else S3/'outputs'/(arm+'_full')
        rows = jsonl(folder/'responses.jsonl')
        assert [r['user_id'] for r in rows] == users
        assert len(rows)==100
        protocol = read(folder/'protocol.json')
        summary = read(folder/'summary.json')
        assert digest({k:v for k,v in protocol.items() if k!='protocol_hash'}) == protocol['protocol_hash']
        protocol_model = protocol['inference_model'] if arm=='semantic' else protocol['model']
        assert protocol_model == summary['model'] == model
        assert protocol['decoding'] == summary['decoding'] == config
        if arm=='semantic':
            reqs = jsonl(ROOT/'session2/outputs/frozen/semantic_requests.jsonl')
            assert [r['user_id'] for r in reqs] == users
            assert all(r['candidates']==m['candidates'] for r,m in zip(reqs,requests['main']))
            for k,v in baseline.items():
                if k!='protocol_hash':
                    assert protocol[k] == v, k
            old_command = read(ROOT/'session2/evidence/commands/semantic_full.json')
            for name in ('inference_worker.py','experiment.py','run_inference.py'):
                key='session2/src/'+name
                assert sha(ROOT/key)==old_command['source_sha256_at_start'][key]
            for r,req in zip(rows,reqs):
                descriptor=dict(model=model,decoding=config,
                    backend={k:v for k,v in protocol['backend'].items() if k!='load_seconds'},
                    prompt=req['prompt'],worker_source_sha256=sha(ROOT/'session2/src/inference_worker.py'))
                assert r['cache_key']==digest(descriptor)
            proof_files += [ROOT/'session2/outputs/frozen/semantic_requests.jsonl',
                            ROOT/'session2/outputs/frozen/semantic_protocol.json',
                            ROOT/'session2/evidence/commands/semantic_full.json']
        else:
            reqs = requests[arm]
            assert summary['completed'] and summary['actual_generation_calls']==100 and summary['cache_hits']==0
            assert protocol['common_protocol_hash']==read(frozen/'common_protocol.json')['protocol_hash']
            assert protocol['request_file_sha256']==sha(frozen/(arm+'_requests.jsonl'))
            assert summary['run_protocol_hash']==protocol['protocol_hash']
            assert protocol['input_protocol_hash']==read(frozen/(arm+'_protocol.json'))['protocol_hash']
            for key in ('session2/src/inference_worker.py','session2/src/experiment.py','session2/src/run_inference.py'):
                assert protocol['source_sha256'][key]==sha(ROOT/key)
        for r,req in zip(rows,reqs):
            assert r['response']['status']=='generated'
            assert r['response']['user_prompt_sha256']==req['prompt_sha256']
            rebuilt=parse_and_fallback(r['response'].get('text'),req['candidates'],req.get('alias_to_id'),r['response']['status'])
            assert all(r[k]==rebuilt[k] for k in ('status','valid','parse_repaired','fallback','error','ranking'))
            predictions[r['user_id']][arm]=r['ranking']
        if arm=='semantic':
            main_backend=read(S3/'outputs/main_full/backend.json')
            assert {k:v for k,v in protocol['backend'].items() if k!='load_seconds'}=={k:v for k,v in main_backend.items() if k!='load_seconds'}
        proof_files += [folder/'responses.jsonl',folder/'protocol.json',folder/'summary.json',folder/'backend.json']
        responses[arm], summaries[arm], protocols[arm] = rows, summary, protocol
    spec=importlib.util.spec_from_file_location('teacher_metric',ROOT/'vendor/RecSys-LLMs/project/far-trans/starter.py')
    teacher=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher)
    per_user=[]
    methods=list(predictions[users[0]])
    for index,u in enumerate(users):
        truth=set(evaluation[u]['truth'])
        for method in methods:
            ranking=predictions[u][method]
            assert len(ranking)==len(set(ranking))==10
            assert set(ranking)<=set(baseline['eligible_assets'])
            if method in ('main','ablation','semantic','candidate_order'):
                assert set(ranking)<=set(requests['main'][index]['candidates'])
            parts=independent(ranking,truth)
            value=ndcg(ranking,truth)
            assert math.isclose(parts['ndcg_at_10'],value,abs_tol=1e-14)
            assert math.isclose(teacher.ndcg10(ranking,truth),value,abs_tol=1e-14)
            per_user.append(dict(user_id=u,cohort_line=index+1,method=method,relevant_count=len(truth),
                                 hw8_ndcg=value,**parts))
    metrics={m:metrics_for(users,predictions,evaluation,m) for m in methods}
    reference=read(baseline_path/'metrics.json')['methods']
    for method in reference:
        assert all(math.isclose(metrics[method][key],reference[method][key],abs_tol=1e-14)
                   for key in ('ndcg_at_10','ndcg_at_10_nonempty'))
    short=[u for u in users if 1<=evaluation[u]['distinct_history']<=3]
    subgroup={m:metrics_for(short,predictions,evaluation,m) for m in methods}
    paired_main, rows_main=pair(users,predictions,evaluation,'candidate_order','main')
    paired_alias, rows_alias=pair(users,predictions,evaluation,'main','ablation')
    dual=[u for index,u in enumerate(users) if responses['main'][index]['valid'] and responses['ablation'][index]['valid']]
    paired_dual, rows_dual=pair(dual,predictions,evaluation,'main','ablation')
    paired_dual['metrics']={m:metrics_for(dual,predictions,evaluation,m) for m in ('candidate_order','main','ablation')}
    coverage={}
    for arm,rows in responses.items():
        valid=[r['user_id'] for r in rows if r['valid']]
        coverage[arm]=dict(users=100,generated=sum(r['status']=='generated' for r in rows),
            valid=len(valid),fallbacks=sum(r['fallback'] for r in rows),parser_repairs=sum(r['parse_repaired'] for r in rows),
            validation_reasons=dict(Counter(r['error'] for r in rows if r['error'])),
            failure_types=dict(Counter(r['status'] if r['status']!='generated' else 'invalid_response' for r in rows if not r['valid'])),
            finish_reasons=dict(Counter(r['response']['finish_reason'] for r in rows)),
            valid_only_diagnostic={m:metrics_for(valid,predictions,evaluation,m) for m in ('candidate_order',arm)},
            changed_from_candidate_order=sum(predictions[u][arm]!=predictions[u]['candidate_order'] for u in users))
        if arm!='semantic':
            ascending=0
            for r,req in zip(rows,requests[arm]):
                if r['valid']:
                    forward={isin:alias for alias,isin in req['alias_to_id'].items()}
                    aliases=[forward[i] for i in r['ranking']]
                    ascending+=aliases==sorted(aliases)
            coverage[arm]['ascending_alias_order_valid']=ascending
    status_cross=Counter(('valid' if a['valid'] else 'fallback')+'/'+('valid' if b['valid'] else 'fallback')
                         for a,b in zip(responses['main'],responses['ablation']))
    truth_count=sum(len(evaluation[u]['truth']) for u in users)
    hits=[len(set(req['candidates'])&set(evaluation[u]['truth'])) for req,u in zip(requests['main'],users)]
    nonempty=[i for i,u in enumerate(users) if evaluation[u]['truth']]
    retrieval=dict(total_positives=truth_count,retrieved_positives=sum(hits),missed_positives=truth_count-sum(hits),
        recall50_micro=sum(hits)/truth_count,recall50_nonempty_macro=sum(hits[i]/len(evaluation[users[i]]['truth']) for i in nonempty)/len(nonempty),
        nonempty_users_with_retrieved_positive=sum(bool(hits[i]) for i in nonempty),
        short_history_users=len(short),short_history_nonempty=sum(bool(evaluation[u]['truth']) for u in short),
        short_history_retrieved_positives=sum(hits[users.index(u)] for u in short))
    oracle=[]
    for req,u in zip(requests['main'],users):
        truth=set(evaluation[u]['truth'])
        ranking=sorted(req['candidates'],key=lambda asset:asset not in truth)[:10]
        oracle.append(dict(relevant_count=len(truth),ndcg_at_10=ndcg(ranking,truth)))
    retrieval['candidate_oracle_ndcg10']=aggregate(oracle)
    fit=read(S3/'outputs/preflight_metal/prompt_fit.json')
    token_deltas=fit['pairwise_token_count_delta']
    token_summary=dict(main_min=min(fit['counts']['main']),main_max=max(fit['counts']['main']),
        ablation_min=min(fit['counts']['ablation']),ablation_max=max(fit['counts']['ablation']),
        paired_count_differences=sum(bool(v) for v in token_deltas),truncations=0,context=config['n_ctx'],max_output=config['max_tokens'],
        input_token_id_hashes_differ=sum(a['response']['input_token_ids_sha256']!=b['response']['input_token_ids_sha256']
            for a,b in zip(responses['main'],responses['ablation'])))
    runtime={}
    for arm in ('main','ablation'):
        pilot=read(S3/'outputs'/(arm+'_pilot')/'summary.json')
        full=summaries[arm]
        external={phase:read(S3/'evidence/commands'/(arm+'_'+phase+'.json'))['elapsed_seconds'] for phase in ('pilot','full')}
        runtime[arm]=dict(actual_calls_pilot=3,actual_calls_full=100,cache_hits=0,
            full_worker_load_seconds=protocols[arm]['backend']['load_seconds'],
            full_worker_startup_wall_seconds=full['worker_startup_wall_seconds'],
            full_generation_seconds=full['generation_seconds'],pilot_generation_seconds=pilot['generation_seconds'],
            external_seconds=external,external_pilot_plus_full_seconds=sum(external.values()),
            peak_process_memory_bytes=max(r['response'].get('peak_process_memory_bytes',0) for r in responses[arm]))
    old_summary=read(ROOT/'session2/configs/experiment_summary.json')
    runtime['reused_hw8_controls']=dict(classical_four_shared_external_seconds=old_summary['baseline_four_methods_shared_seconds'],
        freezing_shared_seconds=old_summary['input_freezing_shared_seconds'],semantic_external=old_summary['runtime_external_seconds']['semantic'],
        semantic_unique_generation_seconds=old_summary['coverage']['semantic']['generation_seconds'],
        semantic_full_cache_hits=summaries['semantic']['cache_hits'],semantic_full_unique_requests=old_summary['coverage']['semantic']['unique_model_requests'],
        note='Prior HW8 timings, not new HW9 computation; cache-origin duration is not newly generated time')
    runtime['hw9_preprocessing_seconds']=manifest['preprocessing_seconds']
    runtime['hw9_preflight_external_seconds']=read(S3/'evidence/commands/preflight_metal.json')['elapsed_seconds']
    runtime['new_inference_calls_total']=206
    runtime['hw9_bounded_pipeline_external_seconds']=sum(runtime[a]['external_pilot_plus_full_seconds'] for a in ('main','ablation'))+runtime['hw9_preflight_external_seconds']+read(S3/'evidence/commands/prepare.json')['elapsed_seconds']
    runtime['scope']='Sum of separate successful prepare, preflight, pilot and full commands; excludes idle coding/review, sandbox failure, downloads, analysis and rendering. No training; existing weights reused.'
    # Deterministic cases in cohort order. No tuning or additional experiments.
    pools={
        'first_invalid':[i for i in range(100) if not responses['main'][i]['valid'] or not responses['ablation'][i]['valid']],
        'first_dual_valid_changed':[i for i,u in enumerate(users) if u in dual and predictions[u]['main']!=predictions[u]['ablation']],
        'first_comparable_no_effect':[i for i,u in enumerate(users) if u in dual and predictions[u]['main']==predictions[u]['ablation']],
        'first_nonempty_ndcg_difference':[i for i,u in enumerate(users) if evaluation[u]['truth'] and ndcg(predictions[u]['main'],set(evaluation[u]['truth']))!=ndcg(predictions[u]['ablation'],set(evaluation[u]['truth']))]
    }
    cases={}
    for label,indices in pools.items():
        if not indices:
            cases[label]=dict(class_count=0,selected=None)
            continue
        i=indices[0];u=users[i]
        truth=set(evaluation[u]['truth'])
        cases[label]=dict(class_count=len(indices),cohort_line=i+1,user_id=u,truth=sorted(truth),
            retrieved_truth=sorted(truth&set(requests['main'][i]['candidates'])),
            main=responses['main'][i],ablation=responses['ablation'][i],
            components={m:independent(predictions[u][m],truth) for m in ('candidate_order','main','ablation')})
    out.mkdir(parents=True)
    for name,value in [('metrics',metrics),('subgroup',subgroup),('coverage',coverage),('retrieval',retrieval),
                       ('per_user_components',per_user),('predictions',predictions),('paired_main',rows_main),
                       ('paired_alias',rows_alias),('paired_dual_valid',rows_dual),('cases',cases)]:
        dump(out/(name+'.json'),value)
    proof=dict(status='PASS',provenance='Local HW8 outputs, never teacher-published scores',
        common_contract_matches=True,semantic_candidate_order_matches=True,model_decoding_backend_matches=True,
        exact_semantic_cache_descriptors_match=True,raw_responses_reparsed=True,
        classical_predictions_match_frozen=True,baseline_aggregates_recalculated=True,
        sources={str(p.relative_to(ROOT)):sha(p) for p in proof_files})
    dump(out/'control_reuse_proof.json',proof)
    dump(out/'checks.json',dict(all_100_users_aligned=True,all_top10_legal=True,raw_parse_reproduced=True,
        common_and_run_protocol_hashes_verified=True,independent_dcg_idcg_and_teacher_metric_match=True,
        frozen_truth_and_candidate_order_equal=True,future_truth_excluded_from_prompt=True,
        personal_verification='PENDING; AI calculations do not count as personal verification'))
    summary=dict(scope='HW9 development adaptation; same development cohort as HW8',metrics=metrics,subgroup=subgroup,
        paired_main_vs_cbf=paired_main,paired_alias_vs_main=paired_alias,dual_valid_diagnostic=paired_dual,
        validity_cross_table=dict(status_cross),coverage=coverage,retrieval=retrieval,tokens=token_summary,runtime=runtime,
        deterministic_case_counts={k:len(v) for k,v in pools.items()},
        common_protocol_hash=read(frozen/'common_protocol.json')['protocol_hash'],
        common_protocol_file_sha256=sha(frozen/'common_protocol.json'),
        run_protocol_hashes={a:p['protocol_hash'] for a,p in protocols.items()},
        personal_verification='PENDING')
    dump(S3/'configs/results.json',summary)
    # Prefer a new nonempty difference, excluding HW8 line 30 when possible.
    preferred=pools['first_nonempty_ndcg_difference']
    candidates=[i for i in preferred if i+1!=30] or preferred
    if not candidates:
        candidates=[i for i,u in enumerate(users) if evaluation[u]['truth'] and i+1!=30]
    i=candidates[0];u=users[i];truth=set(evaluation[u]['truth'])
    details={m:independent(predictions[u][m],truth) for m in ('main','ablation')}
    worksheet=dict(status='PENDING',AI_calculation_performed=True,personal_confirmation_received=False,
        cohort_line=i+1,user_id=u,truth=sorted(truth),rankings={m:predictions[u][m] for m in details},
        components=details,selection='first nonempty alias-vs-main nDCG difference, excluding HW8 line 30 if possible',
        response_files={m:str((S3/'outputs'/(m+'_full')/'responses.jsonl').relative_to(ROOT)) for m in details})
    dump(S3/'evidence/personal_check.json',worksheet)
    text=f'''# HW9 — личная проверка nDCG@10: PENDING

Этот расчёт выполнил Codex (AI). Личная проверка Дмитрия пока не подтверждена. Проверка HW8 строки 30 её не заменяет.

Выбран первый nonempty случай с изменением nDCG между руками, кроме строки 30 HW8 при наличии альтернативы. Здесь строка **{i+1}**, локальный ключ пользователя `{u}`.

Сверьте строку {i+1} в `session3/outputs/main_full/responses.jsonl` и `session3/outputs/ablation_full/responses.jsonl`: поля `ranking`, `valid`, `fallback`, `error` и сырой `response.text`. Словари алиасов: та же строка в `session3/outputs/frozen/main_requests.jsonl` и `ablation_requests.jsonl`. Убедитесь, что оба декодированных списка относятся к одним 50 исходным кандидатам; при `fallback=true` используется исходный CBF Top-10.

Будущие покупки в [2021-06-01, 2021-07-01), после правил teacher loader: `{sorted(truth)}`. Сверьте `truth` для ключа `{u}` в `session3/outputs/frozen/evaluation.json` (копия HW8, SHA совпал). Это distinct eligible Buy assets; повторные покупки допустимы. При желании исходные строки находятся в `session2/data/transactions.csv`, фильтр customerID + transactionType=Buy + данное окно + transactionID>=0.

| Ранг | Main ISIN | Relevant | Ablation ISIN | Relevant |
|---|---|---|---|---|
'''
    for rank,(a,b) in enumerate(zip(predictions[u]['main'],predictions[u]['ablation']),1):
        text+=f'| {rank} | {a} | {int(a in truth)} | {b} | {int(b in truth)} |\n'
    text+='\nДля binary gain: DCG = сумма 1/log2(rank+1) по релевантным позициям; IDCG использует **все** будущие позитивы, в том числе вне Top-50.\n\n'
    for m,parts in details.items():
        r=responses[m][i]
        terms=' + '.join(f'1/log2({p}+1)' for p in parts['relevant_ranks']) or '0'
        text+=f"**{m}**: valid={r['valid']}, fallback={r['fallback']}, error={r['error']}. Релевантные позиции {parts['relevant_ranks']}. DCG = {terms} = {parts['dcg']:.12f}. IDCG = {parts['idcg']:.12f}. nDCG = {parts['dcg']:.12f}/{parts['idcg']:.12f} = **{parts['ndcg_at_10']:.12f}** ({parts['ndcg_at_10']:.4f}).\n\n"
    text+=f"Парная разность ablation − main = {details['ablation']['ndcg_at_10']-details['main']['ndcg_at_10']:.12f}.\n\n"
    text+='**Задание Дмитрию:** откройте сохранённые строки, проверьте два Top-10 и будущие покупки, найдите релевантные позиции и на калькуляторе пересчитайте DCG, IDCG и обе дроби. Затем напишите, какие строки вы просмотрели, какие позиции нашли и какие два nDCG получили, либо укажите расхождение. До этого статус остаётся PENDING.\n'
    (S3/'evidence/personal_check.md').write_text(text)
    print(json.dumps(dict(metrics=metrics,paired_alias=paired_alias,dual_valid=paired_dual,coverage=coverage,
                         worksheet_line=i+1),indent=2))


if __name__ == '__main__':
    main()
