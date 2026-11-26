"""Review the exact public source allowlist and bind it to actual inference code."""
import re
from common import ROOT, S3, dump, jsonl, read, sha


PUBLIC = ['README.md','reference/design.json','reference/homework3_supplied.md','reference/working_prompt.md',
          'configs/results.json','configs/execution.json','src/common.py','src/log_command.py','src/prepare.py',
          'src/run.py','src/analyze.py','src/test_checks.py','src/make_report.py','src/render_report.py',
          'src/render_pdf.py','src/audit_release.py']


def main():
    main_protocol=read(S3/'outputs/main_full/protocol.json')
    ablation=read(S3/'outputs/ablation_full/protocol.json')
    source_paths=['session3/src/common.py','session3/src/prepare.py','session3/src/run.py',
                  'session2/src/experiment.py','session2/src/inference_worker.py','session2/src/run_inference.py']
    used={}
    for name in source_paths:
        actual=sha(ROOT/name)
        assert actual==main_protocol['source_sha256'][name]==ablation['source_sha256'][name]
        used[name]=actual
    analysis_command=read(S3/'evidence/commands/analyze.json')
    assert analysis_command['exit_code']==0
    assert analysis_command['source_sha256_at_start']['session3/src/analyze.py']==sha(S3/'src/analyze.py')
    backend=main_protocol['backend']
    for arm in ('main','ablation'):
        native=(S3/'outputs'/(arm+'_full')/'native_runtime.log').read_text()
        assert 'offloaded 25/25 layers to GPU' in native and 'using device Metal (Apple M2)' in native
    execution=dict(model=main_protocol['model'],decoding=main_protocol['decoding'],
        hardware=dict(cpu='Apple M2',model='Mac14,2',ram_bytes=8589934592),
        python=read(S3/'outputs/frozen/manifest.json')['python'],
        platform=read(S3/'outputs/frozen/manifest.json')['platform'],
        runtime={k:backend[k] for k in ('llama_cpp_python','system_info','template_sha256','libraries','gpu_offload_supported')},
        inference_source_sha256=used,analysis_source_sha256=sha(S3/'src/analyze.py'),
        teacher_commit=read(ROOT/'session2/reference/teacher_snapshot.json')['commit'],
        inherited_team_main='d206088b7c2f7301a8ac1855c24ee985b3b83fe9',
        calls=dict(main_full=100,ablation_full=100,pilots=6,cache_hits=0),
        confirmed_backend='Metal; native runtime logs confirm 25/25 layers offloaded',
        personal_verification='PENDING')
    dump(S3/'configs/execution.json',execution)
    inputs=jsonl(S3/'outputs/frozen/main_requests.jsonl')
    identifiers=set(read(S3/'outputs/frozen/evaluation.json'))
    identifiers.update(asset for row in inputs for asset in row['alias_to_id'].values())
    review=[]
    for relative in PUBLIC:
        path=S3/relative
        text=path.read_text()
        assert path.suffix in ('.py','.md','.json')
        assert not any(re.search(r'(?<![A-Za-z0-9])'+re.escape(identifier)+r'(?![A-Za-z0-9])',text)
                       for identifier in identifiers), str(path)
        assert not any(word in relative for word in ('outputs/','evidence/','report/','submission/'))
        review.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size))
    dump(S3/'evidence/publication_review.json',dict(reviewed_by='Codex AI',files=review,
        actual_client_and_asset_identifiers_absent=True,source_only_or_aggregates=True,
        excluded='All data, instantiated prompts, responses, individual evidence, logs, models, PDF and native export',
        action='Local commit only. No push in this working session.'))
    print('\n'.join(row['path'] for row in review))


if __name__=='__main__':
    main()
