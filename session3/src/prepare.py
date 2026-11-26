"""Freeze one preregistered alias reversal; no labels are consulted for selection."""
import copy
import hashlib
import json
import platform
import shutil
import sys
import time
from common import ROOT, S3, digest, dump, jsonl, read, sha, source_hashes


def rename_payload(payload, mapping):
    result = copy.deepcopy(payload)
    for name in ('history', 'recent_15_buys_oldest_first', 'candidates'):
        for row in result[name]:
            row[0] = mapping[row[0]]
    return result


def main():
    start = time.perf_counter()
    out = S3/'outputs/frozen'
    if out.exists():
        raise FileExistsError(out)
    design = read(S3/'reference/design.json')
    frozen = ROOT/'session2/outputs/frozen'
    checks = {}
    def verify(path, expected):
        actual = sha(path)
        assert actual == expected, str(path)
        checks[str(path.relative_to(ROOT))] = actual
    verify(S3/'reference/homework3_supplied.md', design['assignment_sha256'])
    verify(ROOT/'session2/outputs/baselines_seed0/protocol.json', design['protocol']['hw8_protocol_file_sha256'])
    verify(frozen/'behavioral_requests.jsonl', design['protocol']['hw8_original_requests_file_sha256'])
    verify(ROOT/'session2/configs/decoding.json', design['model']['hw8_decoding_file_sha256'])
    verify(ROOT/'session2/models'/design['model']['file'], design['model']['file_sha256'])
    manifest = read(frozen/'manifest.json')
    for name, expected in manifest['files'].items():
        verify(frozen/name, expected)
    baseline = read(ROOT/'session2/outputs/baselines_seed0/protocol.json')
    body = {k:v for k,v in baseline.items() if k!='protocol_hash'}
    assert digest(body) == baseline['protocol_hash']
    for name, expected in baseline['source_sha256'].items():
        verify(ROOT/'session2/data'/name, expected)
    snapshot = read(ROOT/'session2/reference/teacher_snapshot.json')
    for row in snapshot['files']:
        verify(ROOT/'vendor/RecSys-LLMs'/row['path'], row['sha256'])
    # The exact worker and parser implementation used in HW8 must still be present.
    old_command = read(ROOT/'session2/evidence/commands/behavioral_full.json')
    for name in ('inference_worker.py', 'experiment.py', 'run_inference.py'):
        key = 'session2/src/'+name
        verify(ROOT/key, old_command['source_sha256_at_start'][key])
    audit = read(ROOT/'session2/evidence/input_audit.json')
    assert audit['users'] == 100 and audit['history_counts_independently_rescanned']
    assert audit['recency_and_popularity_independently_rescanned']
    assert audit['max_selected_buy_date'] < baseline['cutoff'] == '2021-06-01'
    checks['session2/evidence/input_audit.json'] = sha(ROOT/'session2/evidence/input_audit.json')
    requests = jsonl(frozen/'behavioral_requests.jsonl')
    evaluation = read(frozen/'evaluation.json')
    users = baseline['users']
    assert len(users) == len(set(users)) == 100
    assert users == list(evaluation) == [r['user_id'] for r in requests]
    assert baseline['end_exclusive'] == '2021-07-01'
    instruction = (ROOT/'session2/prompts/behavioral.txt').read_text().strip()
    records, ablated = [], []
    expected_fields = {'cutoff_exclusive','repeat_purchases_allowed','distinct_purchased_assets',
                       'total_buy_events','history_columns','history','recent_15_buys_oldest_first',
                       'candidate_columns','candidates'}
    for index, r in enumerate(requests, 1):
        payload = r['payload']
        assert set(payload) == expected_fields
        rebuilt = instruction+'\n'+json.dumps(payload, separators=(',',':'))
        assert rebuilt.encode() == r['prompt'].encode()
        assert hashlib.sha256(rebuilt.encode()).hexdigest() == r['prompt_sha256']
        aliases = sorted(r['alias_to_id'])
        mapping = dict(zip(aliases, reversed(aliases)))
        assert len(set(mapping.values())) == len(aliases)
        changed = rename_payload(payload, mapping)
        inverse = {b:a for a,b in mapping.items()}
        assert rename_payload(changed, inverse) == payload
        reverse = {mapping[a]:isin for a,isin in r['alias_to_id'].items()}
        assert {inverse[a]:isin for a,isin in reverse.items()} == r['alias_to_id']
        assert [reverse[c[0]] for c in changed['candidates']] == r['candidates']
        assert [r['alias_to_id'][c[0]] for c in payload['candidates']] == r['candidates']
        assert len(r['candidates']) == len(set(r['candidates'])) == 50
        assert all(row[2] > 0 for row in payload['history'])
        assert all(row[1] > 0 for row in payload['recent_15_buys_oldest_first'])
        prompt = instruction+'\n'+json.dumps(changed, separators=(',',':'))
        assert all(isin not in prompt for isin in reverse.values()) and r['user_id'] not in prompt
        assert not any(s in prompt for s in ('"truth"','assetName','customerID','riskLevel','investmentCapacity'))
        row = dict(r, prompt=prompt, payload=changed, alias_to_id=reverse,
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        ablated.append(row)
        records.append(dict(cohort_line=index, main_record_sha256=digest(r),
                            ablation_record_sha256=digest(row), main_prompt_sha256=r['prompt_sha256'],
                            ablation_prompt_sha256=row['prompt_sha256'], candidates_sha256=digest(r['candidates']),
                            alias_permutation_sha256=digest(mapping), invertible=True,
                            structure_equal_after_inverse=True, candidate_order_equal=True))
    out.mkdir(parents=True)
    shutil.copyfile(frozen/'behavioral_requests.jsonl', out/'main_requests.jsonl')
    shutil.copyfile(frozen/'evaluation.json', out/'evaluation.json')
    (out/'ablation_requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ablated))
    assert (out/'main_requests.jsonl').read_bytes() == (frozen/'behavioral_requests.jsonl').read_bytes()
    common = dict(baseline, hw9_design_sha256=sha(S3/'reference/design.json'),
                  candidate_source='Unchanged frozen HW8 ordered CBF Top-50',
                  candidates_sha256=digest([r['candidates'] for r in requests]),
                  truth_sha256=sha(out/'evaluation.json'),
                  cohort_selection='teacher SHA256 seed-42 warm-user ordering, first 100, no truth filter',
                  development_reused_from_hw8=True, repeat_purchases_allowed=True)
    common.pop('protocol_hash')
    common['protocol_hash'] = digest(common)
    dump(out/'common_protocol.json', common)
    for arm in ('main','ablation'):
        protocol = dict(common_protocol_hash=common['protocol_hash'], arm=arm,
                        request_file_sha256=sha(out/(arm+'_requests.jsonl')),
                        transformation='identity' if arm=='main' else 'per-user lexicographic alias reversal, one permutation')
        protocol['protocol_hash'] = digest(protocol)
        dump(out/(arm+'_protocol.json'), protocol)
    dump(out/'manifest.json', dict(verified_hw8_sources=checks, records=records,
         files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
         users=100, nonempty=sum(bool(v['truth']) for v in evaluation.values()),
         checks=dict(main_byte_identical_to_hw8=True, bijective_roundtrip_all_100=True,
                     original_ISIN_order_all_100=True, no_future_truth_in_prompts=True),
         historical_input_audit_reused='Verified unchanged CSVs, frozen inputs and audit; no new full raw-data audit claimed',
         preprocessing_seconds=time.perf_counter()-start, source_sha256=source_hashes(),
         python=sys.version, platform=platform.platform()))
    print(json.dumps(dict(users=100, prompts_identical=True, alias_roundtrips=100,
                         preprocessing_seconds=time.perf_counter()-start)))


if __name__ == '__main__':
    main()
