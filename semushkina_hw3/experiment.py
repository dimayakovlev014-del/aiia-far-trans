"""Frozen pointwise comparison; batch checkpoints are the unit of recovery."""
import argparse,csv,hashlib,json,math,os,platform,subprocess,sys,time
from collections import Counter,defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
HW2=ROOT/'semushkina_hw2'
sys.dont_write_bytecode=True
sys.path.insert(0,str(HW2))
from profile import investor,latest_rows,category
from download import sha
PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def compact(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def save(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(compact(value)+'\n');tmp.replace(path)
def load(path):return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)

def prepare():
    start=time.perf_counter();cfg=load(HERE/'config.json');old=load(HW2/'results.json')
    assert old['status']=='complete'
    proto=old['protocol'];common=proto['common'];cutoff=common['cutoff'];data=HW2/'.local/data'
    for name,h in common['source_sha256'].items():assert sha(data/name)==h
    assert common['users']==[r['user_id'] for r in old['frozen']] and len(common['users'])==100
    users=set(common['users']);eligible=set(common['eligible_assets']);events=defaultdict(list);seen=set()
    # Dates of every retained historical event checked independently of prompts.
    for r in rows(data/'transactions.csv'):
        key=tuple(sorted(r.items()))
        if key in seen:continue
        seen.add(key)
        if r['customerID'] in users and r['ISIN'] in eligible and int(r['transactionID'])>=0 and r['transactionType']=='Buy' and r['timestamp'][:10]<cutoff:
            events[r['customerID']].append((r['timestamp'][:10],r['ISIN']))
    profiles=latest_rows(rows(data/'customer_information.csv'),'customerID',cutoff)
    assets=latest_rows(rows(data/'asset_information.csv'),'ISIN',cutoff)
    requests=[]
    for r in old['frozen']:
        uid=r['user_id'];p=r['behavioral_payload'];alias=r['alias_to_id'];forward={v:k for k,v in alias.items()}
        assert p['cutoff_exclusive']==cutoff
        assert Counter(i for t,i in events[uid])=={alias[a]:count for a,count,age in p['history']}
        assert p['recent_15_buys_oldest_first']==[[forward[i],(__import__('datetime').date.fromisoformat(cutoff)-__import__('datetime').date.fromisoformat(t)).days] for t,i in sorted(events[uid])[-15:]]
        assert investor(profiles.get(uid),cutoff)==r['extra_context']['investor_profile']
        candidates=r['candidates'][:10];assert candidates==r['baselines']['cbf']
        top=sorted(p['history'],key=lambda x:(-x[1],x[2],x[0]))[:cfg['history_top_assets']]
        query=dict(cutoff_exclusive=cutoff,repeat_purchases_allowed=True,
                   distinct_bought=p['distinct_purchased_assets'],total_buys=p['total_buy_events'],
                   history_columns=['alias','buy_count','age_days'],history=top,recent_buys=p['recent_15_buys_oldest_first'][-cfg['recent_buys']:])
        enriched=dict(query,**{k:r['extra_context']['investor_profile'][k]['value'] for k in ('segment','risk_tolerance','investment_capacity_band')})
        assert {k:v for k,v in enriched.items() if k not in ('segment','risk_tolerance','investment_capacity_band')}==query
        for rank,isin in enumerate(candidates):
            cand=p['candidates'][rank];assert alias[cand[0]]==isin
            cat=category(assets.get(isin),cutoff)
            assert next(x[1:] for x in r['extra_context']['candidate_categories'] if x[0]==cand[0])==cat
            doc=dict(zip(('alias','cbf_score','original_rank','buy_count','last_buy_age_days','distinct_buyers'),cand))
            doc.update(zip(('asset_type','asset_subtype','sector','industry'),cat))
            for arm,q in [('behavioral',query),('profile_enriched',enriched)]:
                text=PREFIX+'<Instruct>: '+cfg['instruction']+'\n<Query>: '+compact(q)+'\n<Document>: '+compact(doc)+SUFFIX
                assert not any(i in text for i in eligible)
                requests.append(dict(key=f'{uid}:{rank}:{arm}',user_id=uid,rank=rank,arm=arm,prompt=text,prompt_sha256=hashlib.sha256(text.encode()).hexdigest()))
    manifest=dict(config=cfg,hw2_results_sha256=sha(HW2/'results.json'),common_protocol_hash=proto['common_protocol_hash'],
        common=common,candidates_sha256=digest([(r['user_id'],r['candidates'][:10]) for r in old['frozen']]),
        requests_sha256=digest(requests),source_sha256={p.name:sha(p) for p in (HERE/'experiment.py',HERE/'config.json')},
        hw2_unchanged_hashes={p.name:sha(p) for p in HW2.iterdir() if p.is_file()},
        temporal_checks='all historical buys strictly pre-cutoff; independently rebuilt profiles/categories match HW2; no future truth passed to input builder',
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    manifest['protocol_hash']=digest(manifest)
    manifest['preparation_seconds']=time.perf_counter()-start
    path=HERE/'manifest.json'
    if path.exists():
        prior=load(path)
        assert prior['protocol_hash']==manifest['protocol_hash'],'Protocol changed; do not reuse scores'
        return prior,requests,old
    save(path,manifest)
    (HERE/'requests.jsonl').write_text(''.join(compact(r)+'\n' for r in requests))
    return manifest,requests,old

def read_batches(path,protocol):
    batches=[]
    if not path.exists():return batches
    with path.open('rb+') as f:
        while True:
            start=f.tell();line=f.readline()
            if not line:break
            if not line.endswith(b'\n'):
                # Only the interrupted, uncommitted final batch can be removed.
                f.truncate(start);break
            batch=json.loads(line);assert batch['protocol_hash']==protocol
            batches.append(batch)
    return batches

def run(phase):
    manifest,requests,old=prepare();cfg=manifest['config'];path=HERE/'scores.jsonl'
    batches=read_batches(path,manifest['protocol_hash']);done={x['key']:x for b in batches for x in b['scores']}
    limit=3 if phase=='smoke' else 100
    selected=set(manifest['common']['users'][:limit])
    pending=[r for r in requests if r['user_id'] in selected and r['key'] not in done]
    if not pending:
        print(f'{limit}/{limit} users already saved; no inference.',flush=True)
        if phase=='full':evaluate()
        return
    if phase=='full':
        smoke=load(HERE/'smoke.json');assert smoke['users']==3 and smoke['valid_scores']==60
        assert (HERE/'full.authorized').read_text().strip()==manifest['protocol_hash'],'Wait for smoke runtime review'
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; launch this command in a regular local Terminal. No CPU substitution.')
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    folder=HERE/'.local/model';hashes=load(folder/'hashes.json')
    for name,h in hashes.items():assert sha(folder/name)==h
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,padding_side='left')
    ids={r['key']:tokenizer.encode(r['prompt'],add_special_tokens=False) for r in requests}
    assert max(map(len,ids.values()))<=cfg['max_length'],'Stop: prompt exceeds frozen max_length, no truncation'
    no=tokenizer.convert_tokens_to_ids('no');yes=tokenizer.convert_tokens_to_ids('yes')
    assert tokenizer.encode('yes',add_special_tokens=False)==[yes] and tokenizer.encode('no',add_special_tokens=False)==[no]
    start=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(folder,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).to('mps').eval()
    torch.mps.synchronize();load_seconds=time.perf_counter()-start
    from importlib.metadata import distributions
    backend=dict(torch=torch.__version__,transformers=transformers.__version__,device='mps',dtype='float16',
                 dependencies={d.metadata['Name']:d.version for d in distributions()},
                 python=platform.python_version(),platform=platform.platform(),chip='Apple M4',memory_gb=16,
                 model_files_sha256=hashes,yes_token_id=yes,no_token_id=no,min_input_tokens=min(map(len,ids.values())),max_input_tokens=max(map(len,ids.values())))
    if 'backend' in manifest:assert manifest['backend']==backend
    else:manifest['backend']=backend;save(HERE/'manifest.json',manifest)
    phase_start=time.perf_counter();completed_before=len(done)
    with path.open('a') as stream:
        # Homogeneous-arm batches permit separately measured inference times.
        for uid in manifest['common']['users'][:limit]:
            for arm in ('behavioral','profile_enriched'):
                group=[r for r in pending if r['user_id']==uid and r['arm']==arm]
                for offset in range(0,len(group),cfg['batch_size']):
                    group_batch=group[offset:offset+cfg['batch_size']]
                    inputs=tokenizer.pad({'input_ids':[ids[r['key']] for r in group_batch]},padding=True,return_tensors='pt')
                    inputs={k:v.to('mps') for k,v in inputs.items()}
                    torch.mps.synchronize();begin=time.perf_counter()
                    with torch.inference_mode():
                        logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,:]
                        values=logits[:,[no,yes]].float().cpu().tolist()
                    torch.mps.synchronize();seconds=time.perf_counter()-begin
                    scored=[]
                    for request,(negative,positive) in zip(group_batch,values):
                        margin=positive-negative
                        if not math.isfinite(margin):raise RuntimeError('Nonfinite logits; saved earlier batches remain usable, no repair calls')
                        scored.append(dict(key=request['key'],user_id=uid,rank=request['rank'],arm=arm,
                            prompt_sha256=request['prompt_sha256'],input_tokens=len(ids[request['key']]),logit_no=negative,logit_yes=positive,score=margin))
                    batch=dict(protocol_hash=manifest['protocol_hash'],arm=arm,seconds=seconds,phase=phase,scores=scored)
                    stream.write(compact(batch)+'\n');stream.flush();os.fsync(stream.fileno())
                    done.update({r['key']:r for r in scored})
            count=sum(all(f'{u}:{rank}:{arm}' in done for rank in range(10) for arm in ('behavioral','profile_enriched')) for u in manifest['common']['users'][:limit])
            if count<=3 or count%10==0 or count==limit:print(f'{count}/{limit} users complete; scores saved.',flush=True)
    if phase=='smoke':
        bs=read_batches(path,manifest['protocol_hash']);inference=sum(b['seconds'] for b in bs)
        smoke=dict(users=3,valid_scores=sum(len(b['scores']) for b in bs),inference_seconds=inference,
                   wall_seconds=time.perf_counter()-phase_start,model_load_seconds=load_seconds,
                   projected_full_inference_seconds=inference/3*100,projected_full_seconds=inference/3*100+load_seconds,
                   max_input_tokens=backend['max_input_tokens'],protocol_hash=manifest['protocol_hash'])
        save(HERE/'smoke.json',smoke);print(compact(smoke),flush=True)
    else:evaluate()

def ndcg(ids,truth):
    ideal=sum(1/math.log2(k+1) for k in range(1,min(10,len(truth))+1))
    return sum(1/math.log2(k+1) for k,i in enumerate(ids,1) if i in truth)/ideal if ideal else 0.
def aggregate(users,rankings,truth):
    nonempty=[u for u in users if truth[u]]
    return dict(users=len(users),nonempty_truth_users=len(nonempty),ndcg_at_10=sum(ndcg(rankings[u],truth[u]) for u in users)/len(users),
                ndcg_at_10_nonempty=sum(ndcg(rankings[u],truth[u]) for u in nonempty)/len(nonempty) if nonempty else None)

def evaluate():
    m=load(HERE/'manifest.json');old=load(HW2/'results.json');batches=read_batches(HERE/'scores.jsonl',m['protocol_hash'])
    scores={x['key']:x for b in batches for x in b['scores']};assert len(scores)==2000,'Incomplete; no final metrics produced'
    users=m['common']['users'];frozen={x['user_id']:x for x in old['frozen']};truth={u:set(frozen[u]['truth']) for u in users}
    predictions={a:{} for a in ('behavioral','profile_enriched')}
    for a in predictions:
        for u in users:
            original=frozen[u]['candidates'][:10]
            order=sorted(range(10),key=lambda rank:(-scores[f'{u}:{rank}:{a}']['score'],rank))
            predictions[a][u]=[original[i] for i in order]
            assert len(set(predictions[a][u]))==10 and set(predictions[a][u])==set(original)
    metrics={a:aggregate(users,p,truth) for a,p in predictions.items()}
    paired=[]
    for u in users:
        b,e=predictions['behavioral'][u],predictions['profile_enriched'][u]
        paired.append(dict(user_id=u,changed=b!=e,behavioral_ndcg=ndcg(b,truth[u]),profile_ndcg=ndcg(e,truth[u]),delta=ndcg(e,truth[u])-ndcg(b,truth[u]),
                           known_risk_capacity=all(frozen[u]['extra_context']['investor_profile'][k]['status']=='observed' for k in ('risk_tolerance','investment_capacity_band'))))
    counts=lambda rows:dict(users=len(rows),changed_rankings=sum(x['changed'] for x in rows),improved=sum(x['delta']>0 for x in rows),worsened=sum(x['delta']<0 for x in rows),unchanged=sum(x['delta']==0 for x in rows))
    segments={}
    for known,label in [(True,'known_54'),(False,'unknown_46')]:
        subgroup=[x['user_id'] for x in paired if x['known_risk_capacity']==known]
        assert len(subgroup)==(54 if known else 46)
        segments[label]=dict(metrics={a:aggregate(subgroup,p,truth) for a,p in predictions.items()},paired=counts([x for x in paired if x['user_id'] in subgroup]))
    examples={}
    for label,test in [('improved',lambda x:x['delta']>0),('worsened',lambda x:x['delta']<0),('changed_same_ndcg',lambda x:x['changed'] and x['delta']==0)]:
        chosen=next((x for x in sorted(paired,key=lambda x:not bool(truth[x['user_id']])) if test(x)),None)
        examples[label]=chosen
    chosen=examples['improved'] or examples['worsened'] or next(x for x in paired if truth[x['user_id']])
    u=chosen['user_id'];original=frozen[u]['candidates'][:10]
    verification=dict(status='Personal verification: PENDING',performed_by='AI calculation; awaiting student confirmation',user_id=u,truth=sorted(truth[u]),
        original_cbf_top10=original,candidates=[dict(id=i,original_rank=k+1,behavioral_score=scores[f'{u}:{k}:behavioral']['score'],profile_score=scores[f'{u}:{k}:profile_enriched']['score']) for k,i in enumerate(original)],methods={})
    for a in predictions:
        ids=predictions[a][u];gains=[1/math.log2(k+1) if i in truth[u] else 0 for k,i in enumerate(ids,1)]
        ideal=sum(1/math.log2(k+1) for k in range(1,min(10,len(truth[u]))+1))
        verification['methods'][a]=dict(top10=ids,relevant_positions=[k for k,i in enumerate(ids,1) if i in truth[u]],dcg=sum(gains),idcg=ideal,ndcg=ndcg(ids,truth[u]))
    for name,h in m['hw2_unchanged_hashes'].items():assert sha(HW2/name)==h,'HW2 changed'
    output=dict(status='complete',protocol_hash=m['protocol_hash'],manifest=m,metrics=metrics,hw2_metrics=old['metrics'],paired=counts(paired),per_user=paired,segments=segments,
        examples=examples,personal_verification=verification,predictions=predictions,
        changed_vs_cbf={a:sum(predictions[a][u]!=frozen[u]['candidates'][:10] for u in users) for a in predictions},
        reliability={a:dict(users=100,valid_rankings=100,invalid_rankings=0,valid_scores=1000,invalid_scores=0,repairs=0,fallback=0) for a in predictions},
        timing=dict(preparation_seconds=m['preparation_seconds'],inference_seconds={a:sum(b['seconds'] for b in batches if b['arm']==a) for a in predictions},
                    scope='Includes smoke scores reused in full; inference timing excludes model loading, tokenization and disk checkpoints'),
        smoke=load(HERE/'smoke.json'),checks=dict(hw2_unchanged=True,cohort_and_labels_match_hw2=True,candidate_top10_fixed=True,one_profile_ablation_only=True,temporal_inputs_checked=True,saved_scores=2000))
    save(HERE/'results.json',output)
    with (HERE/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['user_id','method','rank','asset_id','score','ndcg_at_10'])
        for u in users:
            for a in predictions:
                for k,i in enumerate(predictions[a][u],1):w.writerow([u,a,k,i,scores[f'{u}:{original_rank(frozen[u],i)}:{a}']['score'],ndcg(predictions[a][u],truth[u])])
    print(compact(dict(metrics=metrics,paired=output['paired'],segments=segments)),flush=True)

def original_rank(frozen,isin):return frozen['candidates'][:10].index(isin)

if __name__=='__main__':
    import fcntl
    lock=(HERE/'.local/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','smoke','full','evaluate'],default='prepare');a=p.parse_args()
    if a.phase=='prepare':m,_,_=prepare();print('Prepared 100 users; temporal checks passed; protocol',m['protocol_hash'])
    elif a.phase=='evaluate':evaluate()
    else:run(a.phase)
