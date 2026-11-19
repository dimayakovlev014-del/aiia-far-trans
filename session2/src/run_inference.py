"""Bounded real inference with model/settings/prompt-bound cache and explicit fallback."""
import argparse
import json
from pathlib import Path
import selectors
import subprocess
import sys
import time

from experiment import ROOT,digest,dump,jsonl,parse_and_fallback
from prepare_data import sha


class Worker:
    def __init__(self,folder,config):
        self.config=config
        self.log=(folder/'native_runtime.log').open('a')
        self.process=subprocess.Popen([sys.executable,str(ROOT/'session2/src/inference_worker.py')],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        self.selector=selectors.DefaultSelector()
        self.selector.register(self.process.stdout,selectors.EVENT_READ)
        self.ready=self.receive(config['load_timeout_seconds'])
        if self.ready.get('status')!='ready':
            raise RuntimeError(json.dumps(self.ready))

    def receive(self,timeout):
        if not self.selector.select(timeout):
            self.process.kill()
            self.process.wait()
            return dict(status='timeout',error=f'Worker exceeded {timeout} seconds')
        line=self.process.stdout.readline()
        if not line:
            return dict(status='worker_exit',error=f'Worker ended: {self.process.poll()}')
        return json.loads(line)

    def request(self,request,timeout=None):
        if self.process.poll() is not None:
            return dict(status='worker_exit',error=f'Worker ended: {self.process.returncode}')
        self.process.stdin.write(json.dumps(request,ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        return self.receive(timeout or self.config['request_timeout_seconds'])

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        self.selector.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['behavioral','semantic'],required=True)
    p.add_argument('--phase',choices=['pilot','full'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a fresh output directory; cache is shared separately')
    a.output.mkdir(parents=True)
    start=time.perf_counter()
    config=json.loads((ROOT/'session2/configs/decoding.json').read_text())
    model=json.loads((ROOT/'session2/configs/model.json').read_text())
    frozen=ROOT/'session2/outputs/frozen'
    manifest=json.loads((frozen/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        assert sha(frozen/name)==expected
    requests=jsonl(frozen/f'{a.arm}_requests.jsonl')
    selected=requests[:3] if a.phase=='pilot' else requests
    cache=ROOT/'session2/outputs/generation_cache'
    cache.mkdir(exist_ok=True)
    worker=None
    records=[]
    try:
        worker=Worker(a.output,config)
        backend=worker.ready['backend']
        dump(a.output/'backend.json',backend)
        # Tokenize every complete prompt before any inference; all 50 candidates stay.
        sizes=worker.request(dict(action='count',prompts=[r['prompt'] for r in requests]))
        if sizes.get('status')!='counted':
            raise RuntimeError(str(sizes))
        dump(a.output/'prompt_fit.json',dict(counts=sizes['tokens'],max_input=max(sizes['tokens']),
                                            context=config['n_ctx'],max_output=config['max_tokens']))
        if max(sizes['tokens'])+config['max_tokens']>config['n_ctx']:
            raise ValueError('One or more FULL prompts do not fit; no model calls made')
        if a.phase=='full':
            gate=ROOT/f'session2/outputs/{a.arm}_pilot/summary.json'
            pilot=json.loads(gate.read_text())
            # Ranking-format failures are the measured reliability outcome,
            # not a reason to omit users or change the frozen test prompt.
            # The pilot gate concerns real generation, context fit and budget.
            if pilot['users']!=3 or pilot['generated']!=3 or pilot['projected_100_seconds']>7200:
                raise RuntimeError('Successful bounded pilot/runtime gate not satisfied')
        with (a.output/'responses.jsonl').open('w') as f:
            for index,request in enumerate(selected):
                if a.phase=='pilot' and time.perf_counter()-start>590:
                    raise TimeoutError('Ten-minute pilot budget reached')
                if a.phase=='full' and time.perf_counter()-start>7200:
                    raise TimeoutError('Two-hour full-run budget reached; compatible calls remain cached')
                descriptor=dict(model=model,decoding=config,backend={k:v for k,v in backend.items() if k!='load_seconds'},
                                prompt=request['prompt'],worker_source_sha256=sha(ROOT/'session2/src/inference_worker.py'))
                key=digest(descriptor)
                target=cache/f'{key}.json'
                begin=time.perf_counter()
                hit=target.exists()
                if hit:
                    response=json.loads(target.read_text())
                else:
                    if worker.process.poll() is not None:
                        worker.close()
                        worker=Worker(a.output,config)
                        with (a.output/'worker_restarts.jsonl').open('a') as restart:
                            restart.write(json.dumps(dict(before_user=request['user_id'],ready=worker.ready))+'\n')
                    remaining=max(1,590-(time.perf_counter()-start)) if a.phase=='pilot' else config['request_timeout_seconds']
                    if a.phase=='full':
                        remaining=min(remaining,max(1,7200-(time.perf_counter()-start)))
                    response=worker.request(dict(prompt=request['prompt']),min(config['request_timeout_seconds'],remaining))
                    dump(target,response)
                if response['status']=='generated':
                    assert response['user_prompt_sha256']==request['prompt_sha256'], 'Worker prompt routing mismatch'
                parsed=parse_and_fallback(response.get('text'),request['candidates'],request.get('alias_to_id'),response['status'])
                record=dict(user_id=request['user_id'],arm=a.arm,cache_key=key,cache_hit=hit,
                            request_wall_seconds=time.perf_counter()-begin,response=response,**parsed)
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
                f.flush()
                records.append(record)
                print(json.dumps(dict(completed=index+1,total=len(selected),status=response['status'],valid=parsed['valid'],
                                      cache_hit=hit,seconds=response.get('inference_seconds'))),flush=True)
        inference=sum(r['response'].get('inference_seconds',0) for r in records)
        summary=dict(arm=a.arm,phase=a.phase,users=len(records),generated=sum(r['response']['status']=='generated' for r in records),
                     valid=sum(r['valid'] for r in records),fallbacks=sum(r['fallback'] for r in records),
                     parse_repairs=sum(r['parse_repaired'] for r in records),cache_hits=sum(r['cache_hit'] for r in records),
                     inference_seconds_including_cached=inference,
                     projected_100_seconds=100*inference/len(records)+backend['load_seconds'],
                     elapsed_seconds=time.perf_counter()-start,model=model,decoding=config,
                     protocol_hash=json.loads((frozen/f'{a.arm}_protocol.json').read_text())['protocol_hash'])
        dump(a.output/'summary.json',summary)
        protocol=dict(json.loads((frozen/f'{a.arm}_protocol.json').read_text()),inference_model=model,
                      decoding=config,backend=backend)
        protocol.pop('protocol_hash')
        protocol['protocol_hash']=digest(protocol)
        dump(a.output/'protocol.json',protocol)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as exc:
        dump(a.output/'incomplete.json',dict(error=f'{type(exc).__name__}: {exc}',users_completed=len(records),
                                            elapsed_seconds=time.perf_counter()-start))
        raise
    finally:
        if worker:
            worker.close()


if __name__=='__main__':
    main()
