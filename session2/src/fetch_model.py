"""Download the pinned official GGUF and verify its published LFS SHA256."""
import json
from pathlib import Path
import time
import urllib.request
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from prepare_data import ROOT, sha

CHOICE = json.loads((ROOT/'session2/configs/model_choice.json').read_text())
REPO, REV, FILE = (CHOICE[k] for k in ('repository','revision','filename'))


def read(url):
    return subprocess.check_output(['curl','-4','--http1.1','--tls-max','1.2','-fLsS','--retry','3','--connect-timeout','15','--max-time','120',url])


def main():
    folder = ROOT/'session2/models'
    folder.mkdir(parents=True, exist_ok=True)
    evidence = ROOT/'session2/evidence/model_source_small'
    evidence.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/api/models/{REPO}/tree/{REV}?recursive=true'
    raw = (evidence/'official_tree.json').read_bytes() if (evidence/'official_tree.json').exists() else read(url)
    (evidence/'official_tree.json').write_bytes(raw)
    entries = json.loads(raw)
    entry, = [e for e in entries if e['path'] == FILE]
    expected = entry['lfs']['oid']
    assert expected == CHOICE['sha256'] and entry['size'] == CHOICE['bytes']
    target = folder/FILE
    start = time.perf_counter()
    if not target.exists():
        partial = target.with_suffix('.partial')
        # Unique query avoids stale CDN signed redirects; never log signed URLs.
        url = f'https://huggingface.co/{REPO}/resolve/{REV}/{FILE}?download=true&request={time.time_ns()}'
        direct=ROOT/'session2/evidence/model_small_direct.private'
        if not direct.exists():
            with direct.open('w') as f:
                subprocess.run(['curl','-4','--http1.1','--tls-max','1.2','-sSIL','--retry','2','--connect-timeout','10',
                                '--max-time','30','--output',str(evidence/'redirect_headers.private'),
                                '--write-out','%{url_effective}',url],stdout=f)
        if direct.exists() and 'us.aws.cdn.hf.co/' in direct.read_text():
            url=direct.read_text().strip()
        chunks=folder/(FILE+'.chunks')
        chunks.mkdir(exist_ok=True)
        size=entry['size']
        chunk_size=4*1024*1024
        def download(index):
            begin=index*chunk_size
            end=min(size,begin+chunk_size)-1
            expected_size=end-begin+1
            part=chunks/f'{index:03d}.part'
            if part.exists() and part.stat().st_size==expected_size:
                return part
            for attempt in range(8):
                resolve=[]
                probes=ROOT/'session2/evidence/cdn_probes.json'
                if probes.exists() and 'us.aws.cdn.hf.co/' in url:
                    healthy=sorted((p for p in json.loads(probes.read_text()) if p['exit']==0),key=lambda p:p['seconds'])
                    if healthy:
                        ip=healthy[(index+attempt)%len(healthy)]['ip']
                        resolve=['--resolve',f'us.aws.cdn.hf.co:443:{ip}']
                with (chunks/f'{index:03d}.log').open('a') as log:
                    result=subprocess.run(['curl','-4','--http1.1','--tls-max','1.2',*resolve,'-fLsS','--connect-timeout','10','--max-time','90',
                                    '--max-filesize',str(expected_size),'--range',f'{begin}-{end}',
                                    '--output',str(part),url],stdout=log,stderr=log)
                if part.exists() and part.stat().st_size==expected_size:
                    return part
            raise RuntimeError(f'Chunk {index} failed after eight bounded attempts')
            return part
        count=(size+chunk_size-1)//chunk_size
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(download,index) for index in range(count)]
            errors=[]
            for done,future in enumerate(as_completed(futures),1):
                try:
                    future.result()
                except Exception as exc:
                    errors.append(str(exc))
                    print(str(exc),flush=True)
                if done%10==0 or done==count:
                    print(f'completed download chunks {done}/{count}',flush=True)
        if errors:
            raise RuntimeError(str(errors))
        with partial.open('wb') as dest:
            import shutil
            for index in range(count):
                with (chunks/f'{index:03d}.part').open('rb') as source:
                    shutil.copyfileobj(source,dest)
        assert sha(partial) == expected, 'Downloaded GGUF digest mismatch'
        partial.rename(target)
    assert sha(target) == expected
    for name in ('README.md', 'LICENSE'):
        (evidence/name).write_bytes(read(f'https://huggingface.co/{REPO}/raw/{REV}/{name}'))
    record = dict(repository=REPO, revision=REV, filename=FILE, sha256=expected,
                  bytes=target.stat().st_size, license='apache-2.0',
                  source_tree=f'https://huggingface.co/{REPO}/tree/{REV}',
                  download_seconds=time.perf_counter()-start,
                  choice_reason='User selected a smaller open-weight model after measured network delays made the proposed 3B download impractical. No 3B inference was run.',
                  native_context=8192)
    (ROOT/'session2/configs/model.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
