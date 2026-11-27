# Semushkina Homework 3

**Investor profile value in pointwise financial asset reranking.**

Sofya Semushkina, team AIIA. This is one completed development experiment on
the same deterministic 100 FAR-Trans users as HW2. The LLM is used only as a
Ranker. H1 asks whether segment, risk tolerance and investment capacity improve
pointwise reranking when list-generation errors are removed.

## Results

| Method | nDCG@10 all 100 | nDCG@10 nonempty 11 |
|---|---:|---:|
| Popularity | 0.0286 | 0.2599 |
| CBF | 0.0201 | 0.1827 |
| Latest buy | 0.0277 | 0.2520 |
| Latest buy or sell | 0.0271 | 0.2464 |
| Original CBF order | 0.0201 | 0.1827 |
| HW2 behavior-only listwise | 0.0214 | 0.1946 |
| HW2 profile-enriched listwise | 0.0214 | 0.1946 |
| HW3 behavior-only pointwise | **0.0282** | **0.2564** |
| HW3 profile-enriched pointwise | **0.0274** | **0.2491** |

H1 is **not supported** in this sample. The profile changes 99 rankings;
0 user scores improve, 1 worsens and 99 remain unchanged. Each new arm has
100 valid rankings, 0 invalid rankings, 0 repairs and 0 fallbacks. Only 5 of
the 11 nonempty users have a positive in the fixed CBF Top-10.

Risk/capacity are known for 54 users and unknown for 46; segment is known for
all 100. The known subgroup has identical mean scores across arms. The one
decrease occurs in the unknown subgroup, which still receives segment.
This result does not establish statistical significance or a universal absence
of profile value. HW2-to-HW3 comparisons also change the model and candidate
pool, so they cannot isolate scoring architecture alone.

Preparation took 1.4939 s. Actual inference, including reused smoke scores,
took 161.2511 s for behavior-only and 171.7972 s for profile-enriched. Loading,
tokenization and checkpoint writes are excluded from these inference timings.

## Protocol and files

- Model: `Qwen/Qwen3-Reranker-0.6B`.
- Revision: `e61197ed45024b0ed8a2d74b80b4d909f1255473`.
- Experiment source commit: `f76dc87e1abb089626faee3e5793056380801dea`.
- Cutoff exclusive: 2021-06-01; truth window: [2021-06-01, 2021-07-01).
- Repeat purchases allowed; empty truth contributes zero.
- Both arms independently score the same original CBF Top-10 candidates.
- Score: `logit(yes) - logit(no)`; exact ties retain CBF order.
- The only input ablation removes segment, risk tolerance and capacity.
- Opaque aliases; no real asset names or future inputs.
- MPS float16 on Apple M4, 16 GB RAM; batch size 10; max_length 512.
- Complete prompts use 243–351 tokens; there is no input truncation.
- One smoke test on 3 users, reused in the one complete 100-user run.

`results_summary.json` preserves exact aggregate metrics, subgroup analysis,
timings, model-file hashes and protocol hashes. The Word report contains the
failure analysis and both rankings/scores for the personal check. It is the
report snapshot prepared before repository publication. **Personal verification
is PENDING**; no student confirmation is claimed.

`experiment.py` implements preparation, batch scoring and evaluation;
`verify.py` independently audits saved scores and teacher metrics;
`test_checkpoint.py` checks interrupted-write recovery and protocol rejection;
`make_report.py` creates the Word report from checked complete results.

## Reproduction prerequisites

The public upload contains source and aggregate evidence. Exact reproduction
also requires the author's original local HW2 artifacts, which are deliberately
not published here:

- `../semushkina_hw2/results.json` with the frozen inputs;
- `../semushkina_hw2/profile.py`, the unchanged temporal helper;
- `../semushkina_hw2/.local/data/` with the four original FAR-Trans CSV files.

The scripts verify their hashes and reuse them read-only. Obtain these
prerequisites from the author before running; the public source upload alone
does not contain the data needed for a fresh inference run. Raw prompts,
user-level score files, model weights, environments and the AI-session export
are excluded from Git. No HW2 experiments need to be repeated.

On a Mac with MPS, Python 3.9.6 and those prerequisites, from the repository root:

```sh
/usr/bin/python3 -m venv semushkina_hw3/.local/venv
semushkina_hw3/.local/venv/bin/python -m pip install -r semushkina_hw3/requirements.txt
semushkina_hw3/.local/venv/bin/python semushkina_hw3/download.py
env PYTHONDONTWRITEBYTECODE=1 semushkina_hw3/.local/venv/bin/python -m unittest discover -s semushkina_hw3 -p test_checkpoint.py -v
env PYTHONDONTWRITEBYTECODE=1 semushkina_hw3/.local/venv/bin/python semushkina_hw3/launch.py
```

After smoke, inspect `smoke.json` and its runtime forecast. To release the full
run, write the exact `protocol_hash` from `manifest.json` into
`semushkina_hw3/full.authorized`. The launcher waits for this review gate.
Resume with the same command after interruption. Each batch is appended to
`scores.jsonl`, flushed and fsynced; compatible saved scores are skipped.
The manifest records the exact installed dependency versions and backend.

To audit a completed local run without further model calls:

```sh
env PYTHONDONTWRITEBYTECODE=1 semushkina_hw3/.local/venv/bin/python semushkina_hw3/verify.py
```

The Word builder requires `python-docx` in a separate document runtime and is
run only after the independent verification succeeds. There is no training,
extra LLM repair call, model search or additional ablation in this experiment.
