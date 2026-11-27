# HW3: grammar-constrained decoding for the LLM reranker (FAR-Trans, development)

Author: Aleksandr Selishchev, team AIIA (HSE LLM4Rec, Session 3 / Homework 3).
Track: the LLM is used as a **Ranker** only; it reorders the starter's fixed 50 CBF
candidates. No training.

## Why

In HW2 (`../selishchev_hw2/`) the course reranker (Qwen3-1.7B, greedy) returned an
invalid list for 23 of 100 users (31 with asset names masked): ISINs outside the 50
candidates, duplicates or wrong length. Invalid users fall back to the candidate
order, and for 9 of the 11 users with future purchases this hid every LLM decision.

## Main method and ablation

* **Main (`llmrank_constrained`)**: the unchanged course prompt and model, greedy
  decoding, but every generation step is masked to tokens that keep the answer a
  JSON array of exactly 10 unique ISINs from the user's 50 candidates
  (`constrained.py`; a character automaton walked jointly with a trie of the
  tokenizer's ASCII tokens). After `]` only end-of-sequence is allowed. The
  processor records *interventions*: steps where the unconstrained greedy token was
  not allowed.
* **Ablation (`llmrank_constrained_masked`)**: the same, with every `assetName` in
  the prompt replaced by `MASKED` (Session 1 ablation (a), HW2 change). It tests what
  the asset names contribute once every output is valid.
* **Controls**: the initial candidate order (first 10 of 50 CBF = CBF), the four
  starter baselines, and the provided course reranker with free decoding: rerun on
  the same device and dtype as the HW3 runs (`--decoding free`, `llmrank_free`) when
  time allows, and the HW2 CPU float32 runs `../selishchev_hw2/outputs/llmrank{,_masked}`
  (same model, revision and prompts).

Runtime options: `--device cpu --dtype float32` reproduces the HW2 setting (about
50 s per user on an Apple M4); `--device mps --dtype bfloat16` runs on the Apple GPU
in the dtype the course reference used on CUDA. All arms compared with each other
must use the same device and dtype; `evaluate.py` prints them per run.

Development setting, defined before any HW3 run: the starter's deterministic
100-user warm sample, cutoff 2021-06-01, 30-day window, repeats allowed, binary
nDCG@10 (empty truth = 0). These are development results, not official scores.

## Run (from this folder, with the HW2 virtual environment)

Set HF_HUB_OFFLINE=1 when huggingface.co is unreachable (transformers 4.57.6 queries the Hub in its tokenizer check even with local_files_only=True).

```bash
source ../selishchev_hw2/.venv/bin/activate
python test_constrained.py                                   # grammar + synthetic decoding
python run_ranker.py --mask-names off --prepare-only --data-dir DATA --output-dir outputs/prepare --model Qwen/Qwen3-1.7B
python test_constrained.py --tokenizer                       # real Qwen3 tokenizer, real candidate lists
python test_pipeline.py --data-dir DATA                      # fake generators, evaluation contract
python ../selishchev_hw2/run_baselines.py --data-dir DATA --output-dir outputs/baseline
COMMON="--data-dir DATA --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --device cpu --on-invalid fallback"
python run_ranker.py --mask-names off $COMMON --output-dir outputs/llmrank_constrained
python run_ranker.py --mask-names on  $COMMON --output-dir outputs/llmrank_constrained_masked
python evaluate.py --data-dir DATA                           # checks + RESULTS.md
python show_user.py --data-dir DATA [--user ID]              # material for a hand check
```

`evaluate.py` checks that all runs share the starter's users, window, data hashes,
candidates and order; that the constrained runs received exactly the same prompts as
the free runs; that masked prompts differ only in `assetName`; that no prompt data
is dated on or after the cutoff; that labels rebuilt from `transactions.csv` give
the same per-user nDCG@10; and that each ranking re-scored by the unchanged starter
gives the same score under the same protocol hash. It reports invalid outputs,
repairs, fallbacks, interventions, coverage of future purchases by the candidates,
paired per-user wins/ties/losses and bootstrap 95% intervals.

Committed: code and the aggregate `RESULTS.md`. Local only: `outputs/`.
