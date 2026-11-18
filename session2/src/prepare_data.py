"""Reuse and verify the existing archive and untouched teacher snapshot."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    provenance = json.loads((ROOT/'reference/provenance.json').read_text())
    archive = ROOT.parent/'llm4rec_a07/reference/FAR-Trans.zip'
    digest = sha(archive)
    assert digest == provenance['archive']['sha256']
    target = ROOT/'session2/data'
    target.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((ROOT/'session2/reference/teacher_snapshot.json').read_text())
    for entry in snapshot['files']:
        assert sha(ROOT/'vendor/RecSys-LLMs'/entry['path']) == entry['sha256'], entry['path']
    extracted = {}
    with zipfile.ZipFile(archive) as z:
        for name in ('transactions.csv','asset_information.csv','customer_information.csv','close_prices.csv'):
            member, = [n for n in z.namelist() if n.split('/')[-1] == name and not n.startswith('__MACOSX/')]
            path = target/name
            if not path.exists():
                with z.open(member) as source, path.open('wb') as dest:
                    import shutil
                    shutil.copyfileobj(source, dest)
            extracted[name] = dict(sha256=sha(path), bytes=path.stat().st_size, zip_member=member)
    output = dict(archive_sha256=digest, source='Existing Session 1 archive; not redownloaded',
                  teacher_commit=snapshot['commit'], teacher_hashes_verified=len(snapshot['files']), files=extracted)
    (ROOT/'session2/evidence/data_manifest.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
