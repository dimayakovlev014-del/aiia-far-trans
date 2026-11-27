"""Download only the pinned small reranker, reusing any verified files."""
import hashlib,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    c=json.loads((HERE/'config.json').read_text());folder=HERE/'.local/model';folder.mkdir(parents=True,exist_ok=True)
    tree=json.loads(subprocess.check_output(['curl','-fsSL','--max-time','30',f"https://huggingface.co/api/models/{c['model']}/tree/{c['revision']}?recursive=true"]))
    names={'config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','README.md'}
    for entry in tree:
        name=entry['path']
        if name not in names:continue
        path=folder/name;expected=entry.get('lfs',{}).get('oid')
        if not path.exists():
            tmp=path.with_suffix(path.suffix+'.partial')
            subprocess.run(['curl','-4','-fsSL','--connect-timeout','20','--max-time','600','-o',str(tmp),f"https://huggingface.co/{c['model']}/resolve/{c['revision']}/{name}?download=true"],check=True)
            assert tmp.stat().st_size==entry['size']
            if expected:assert sha(tmp)==expected
            tmp.rename(path)
        assert path.stat().st_size==entry['size']
        if expected:assert sha(path)==expected
        print('Verified',name,flush=True)
    (folder/'hashes.json').write_text(json.dumps({p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
