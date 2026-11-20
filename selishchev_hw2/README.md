# HW2: asset-name masking in an LLM reranker (FAR-Trans)

Author: Aleksandr Selishchev, team AIIA (HSE LLM4Rec, Session 2 "Make It Run").

Relation to the team's `session2/` (Dmitry Yakovlev): his arm replaces all asset
metadata with numeric behavioral features and aliases (Qwen2.5-0.5B GGUF). This
folder is a separate, narrower ablation on the unchanged teacher prompt and model:
only the asset name is removed. Both use the teacher's frozen top-50 CBF candidates,
the shared `../vendor/RecSys-LLMs` copy (teacher commit `f685086`) and
`PYTHONHASHSEED=0`, which the scripts set themselves.

## Question and change

Session 1 question (Ranker track): does an open LLM that reorders a fixed list of
candidate assets beat the candidate order it receives? Ablation (a) of that plan:
does the LLM's ranking depend on real asset names, which can activate knowledge
from pre-training, or only on structured fields and purchase history?

**One change:** in the course LLMRank-style reranker
(`dryjins/RecSys-LLMs`, `project/far-trans-llm/ranker.py`, commit `f685086`), the
`assetName` value of every asset in the prompt (purchase history and all 50
candidates) is replaced by the constant `MASKED`. Instruction text, users, cutoff,
50 CBF candidates and their order, ISIN output format, model, revision, greedy
decoding, 512-token output limit, parser and fallback rule are unchanged.
ISINs stay visible because they are the required output IDs.

Compared methods on the same 100 users (starter defaults):
Popularity, CBF, Latest buy, Latest buy or sell (unchanged starter),
`candidate_order_first10` (first 10 of the 50 candidates),
`llmrank` (course ranker, names visible) and `llmrank_masked` (this change).

## Run

```bash
cd selishchev_hw2                                  # teacher code: ../vendor (team repo)
python3.12 -m venv .venv && source .venv/bin/activate     # torch 2.11 needs Python >= 3.10
pip install -r requirements.txt
python test_pipeline.py --data-dir /path/to/FAR-Trans     # fake model, contract checks only

python run_baselines.py --data-dir /path/to/FAR-Trans     # -> outputs/baseline
# first call only: add --allow-download to fetch the model
python masked_ranker.py --mask-names off --data-dir /path/to/FAR-Trans \
  --output-dir outputs/llmrank --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device mps --on-invalid fallback
python masked_ranker.py --mask-names on --data-dir /path/to/FAR-Trans \
  --output-dir outputs/llmrank_masked --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device mps --on-invalid fallback
python evaluate.py --data-dir /path/to/FAR-Trans          # checks + RESULTS.md
python show_user.py --data-dir /path/to/FAR-Trans         # material for a by-hand check
```

`--device` accepts `auto|cpu|cuda|mps`; `mps` (Apple Silicon) and `cpu` use float32,
as in the course runtime. Both LLM runs must use the same device.

## What `evaluate.py` checks

- same 100 users, cutoff, window, data SHA-256 and candidate count as the starter's
  `protocol.json`; the four baseline scores equal the starter's;
- every prompt holds the same 50 candidates in the same order as the starter's
  `contexts.jsonl`, and no date on or after the cutoff;
- masked and unmasked prompts are identical once `assetName` is set to `MASKED`;
- truth labels rebuilt from `transactions.csv` independently give the same
  per-user counts and nDCG@10 values;
- each LLM ranking re-scored through the unchanged starter (`--responses`) gives
  the same score under the same `protocol_hash`.

It also reports invalid outputs (fallback to the candidate order), valid-only
scores, overlap with the candidate order and the candidate-list ceiling.

## Committed vs local

Committed: code and the aggregate `RESULTS.md`. Local only (team `.gitignore`):
prompts, generations, caches, `*.jsonl` and user-level `outputs/`.
