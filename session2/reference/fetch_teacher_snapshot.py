"""Download only the two teaching packages at a fixed public Git commit."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = "f6850861ea20a51e1fb018ee4e5f840c2ac759e0"
PROJECT = Path(__file__).resolve().parents[2]
DEST = PROJECT / "vendor" / "RecSys-LLMs"
PREFIXES = ("project/far-trans/", "project/far-trans-llm/")

def read(url):
    req = urllib.request.Request(url, headers={"User-Agent": "LLM4Rec-coursework"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read()

def main():
    tree = json.loads(read(f"https://api.github.com/repos/dryjins/RecSys-LLMs/git/trees/{COMMIT}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("Incomplete source tree")
    entries = [x for x in tree["tree"] if x["type"] == "blob" and x["path"].startswith(PREFIXES)]
    if any(x.get("size", 0) > 5_000_000 for x in entries):
        raise RuntimeError("Unexpected large source file: inspect before downloading")

    def fetch(entry):
        path = entry["path"]
        data = read(f"https://raw.githubusercontent.com/dryjins/RecSys-LLMs/{COMMIT}/{path}")
        target = DEST / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f"Would overwrite different source: {target}")
        target.write_bytes(data)
        return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_blob": entry["sha"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = sorted(pool.map(fetch, entries), key=lambda x: x["path"])
    manifest = {"repository": "https://github.com/dryjins/RecSys-LLMs", "commit": COMMIT,
                "retrieved_on": "2026-11-18", "scope": list(PREFIXES), "files": files,
                "note": "Teacher source snapshot; no model, inference, results or student experiment commit created."}
    (PROJECT / "session2/reference/teacher_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"commit": COMMIT, "files": len(files), "bytes": sum(x["bytes"] for x in files)}))

if __name__ == "__main__":
    main()
