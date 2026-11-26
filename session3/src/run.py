"""Fresh HW9 calls through the unchanged HW8 Metal worker. No generation cache."""
import argparse
import datetime
import json
import time
from common import ROOT, S3, Worker, digest, dump, jsonl, parse_and_fallback, read, sha, source_hashes


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True, choices=['preflight','pilot','full'])
    p.add_argument('--arm', choices=['main','ablation'], default='main')
    a = p.parse_args()
    frozen = S3/'outputs/frozen'
    manifest = read(frozen/'manifest.json')
    for name, expected in manifest['files'].items():
        assert sha(frozen/name) == expected
    out = S3/'outputs'/('preflight_metal' if a.phase=='preflight' else a.arm+'_'+a.phase)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start, started = time.perf_counter(), stamp()
    config = read(ROOT/'session2/configs/decoding.json')
    model = read(ROOT/'session2/configs/model.json')
    worker = None
    records = []
    summary = dict(phase=a.phase, arm=a.arm, started_utc=started, completed=False,
                   cache_policy='disabled: every selected row receives a fresh model call, including duplicates',
                   model=model, decoding=config, source_sha256=source_hashes())
    try:
        load_start = time.perf_counter()
        worker = Worker(out, config)
        summary['worker_startup_wall_seconds'] = time.perf_counter()-load_start
        backend = worker.ready['backend']
        old_backend = read(ROOT/'session2/outputs/behavioral_full/backend.json')
        assert {k:v for k,v in backend.items() if k!='load_seconds'} == {k:v for k,v in old_backend.items() if k!='load_seconds'}
        dump(out/'backend.json', backend)
        if a.phase == 'preflight':
            counts = {}
            count_start = time.perf_counter()
            for arm in ('main','ablation'):
                requests = jsonl(frozen/(arm+'_requests.jsonl'))
                sizes = worker.request(dict(action='count',prompts=[r['prompt'] for r in requests]))
                assert sizes['status'] == 'counted' and len(sizes['tokens']) == 100
                assert max(sizes['tokens'])+config['max_tokens'] <= config['n_ctx']
                counts[arm] = sizes['tokens']
            dump(out/'prompt_fit.json', dict(counts=counts,context=config['n_ctx'],max_output=config['max_tokens'],
                pairwise_token_count_delta=[b-m for m,b in zip(counts['main'],counts['ablation'])],
                truncation=False, all_200_complete_prompts_fit=True))
            summary.update(completed=True,token_count_seconds=time.perf_counter()-count_start,
                           actual_generation_calls=0)
        else:
            fit = read(S3/'outputs/preflight_metal/prompt_fit.json')
            assert fit['all_200_complete_prompts_fit']
            requests = jsonl(frozen/(a.arm+'_requests.jsonl'))
            selected = requests[:3] if a.phase=='pilot' else requests
            if a.phase=='full':
                for arm in ('main','ablation'):
                    pilot = read(S3/'outputs'/(arm+'_pilot')/'summary.json')
                    assert pilot['completed'] and pilot['actual_generation_calls']==3
                    assert pilot['projected_100_seconds'] < 7200
            with (out/'responses.jsonl').open('w') as f:
                for index, request in enumerate(selected):
                    budget = 590 if a.phase=='pilot' else 7200
                    if time.perf_counter()-start > budget:
                        raise TimeoutError('Fixed run budget reached; preserve partial outputs')
                    if worker.process.poll() is not None:
                        worker.close()
                        worker = Worker(out,config)
                        with (out/'worker_restarts.jsonl').open('a') as restarts:
                            restarts.write(json.dumps(dict(before_cohort_line=index+1,ready=worker.ready))+'\n')
                    begin = time.perf_counter()
                    response = worker.request(dict(prompt=request['prompt']),
                        min(config['request_timeout_seconds'], max(1,budget-(time.perf_counter()-start))))
                    if response['status']=='generated':
                        assert response['user_prompt_sha256'] == request['prompt_sha256']
                        assert response['input_tokens'] == fit['counts'][a.arm][index]
                    parsed = parse_and_fallback(response.get('text'),request['candidates'],request['alias_to_id'],response['status'])
                    record = dict(user_id=request['user_id'],cohort_line=index+1,arm=a.arm,cache_hit=False,
                                  request_wall_seconds=time.perf_counter()-begin,response=response,**parsed)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    f.flush()
                    records.append(record)
                    if (index+1)%10==0 or a.phase=='pilot':
                        print(json.dumps(dict(completed=index+1,total=len(selected),status=response['status'])),flush=True)
            summary.update(completed=len(records)==len(selected), users=len(records),
                actual_generation_calls=sum(r['response']['status']=='generated' for r in records),
                valid=sum(r['valid'] for r in records),fallbacks=sum(r['fallback'] for r in records),
                parser_repairs=sum(r['parse_repaired'] for r in records),cache_hits=0,
                generation_seconds=sum(r['response'].get('inference_seconds',0) for r in records),
                served_request_wall_seconds=sum(r['request_wall_seconds'] for r in records),
                kv_reuse='HW8 worker calls llm.reset() before every generation; loaded weights reused within each command',
                projected_100_seconds=100*sum(r['response'].get('inference_seconds',0) for r in records)/len(records)+backend['load_seconds'])
            protocol = dict(input_protocol_hash=read(frozen/(a.arm+'_protocol.json'))['protocol_hash'],
                            common_protocol_hash=read(frozen/'common_protocol.json')['protocol_hash'],
                            model=model,decoding=config,backend=backend,source_sha256=summary['source_sha256'],
                            request_file_sha256=sha(frozen/(a.arm+'_requests.jsonl')),cache=False,phase=a.phase)
            protocol['protocol_hash'] = digest(protocol)
            dump(out/'protocol.json',protocol)
            summary['run_protocol_hash'] = protocol['protocol_hash']
    except Exception as exc:
        summary['fatal_error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if worker:
            worker.close()
        summary.update(elapsed_seconds=time.perf_counter()-start,finished_utc=stamp())
        dump(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_sha256','model','decoding')}),flush=True)


if __name__ == '__main__':
    main()
