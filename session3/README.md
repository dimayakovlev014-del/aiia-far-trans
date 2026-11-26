# HW9 / Session 3: one alias-reversal diagnostic

Dmitry Yakovlev, AIIA. This is a development adaptation of the fixed CBF-candidate
Ranker from HW8. The planned LightGCN/LambdaMART project comparison remains open.
The design was supplied before the new experiment; the already-used HW8 cohort
contains 100 warm users and 11 nonempty future truth sets. No tuning on new scores.

The local HW9 experiment makes new Qwen2.5-0.5B-Instruct Q4_K_M Metal calls for the
original prompts and exactly one consistent per-user lexicographic alias reversal.
All candidate rows, numerical features, chronology, model parameters, parsing and
fallback stay fixed. Caching is disabled, including for duplicate prompts and
pilot requests. Loaded weights are reused within a command, while the unchanged
HW8 worker resets the token state before each generation.

Private inputs, model weights and native logs remain local. Obtain and verify the
HW8 artifacts described in `../session2/README.md` first. Run from the project root
with the existing CPython 3.9.6 virtual environment and pinned Metal runtime:

```sh
env PYTHONHASHSEED=0 session2/.venv/bin/python session3/src/log_command.py --name prepare -- session2/.venv/bin/python session3/src/prepare.py
env PYTHONHASHSEED=0 session2/.venv/bin/python session3/src/log_command.py --name preflight_metal -- session2/.venv/bin/python session3/src/run.py --phase preflight
env PYTHONHASHSEED=0 session2/.venv/bin/python session3/src/log_command.py --name main_pilot -- session2/.venv/bin/python session3/src/run.py --phase pilot --arm main
env PYTHONHASHSEED=0 session2/.venv/bin/python session3/src/log_command.py --name ablation_pilot -- session2/.venv/bin/python session3/src/run.py --phase pilot --arm ablation
env PYTHONHASHSEED=0 session2/.venv/bin/python session3/src/log_command.py --name main_full -- session2/.venv/bin/python session3/src/run.py --phase full --arm main
env PYTHONHASHSEED=0 session2/.venv/bin/python session3/src/log_command.py --name ablation_full -- session2/.venv/bin/python session3/src/run.py --phase full --arm ablation
env PYTHONHASHSEED=0 session2/.venv/bin/python session3/src/log_command.py --name contract_tests -- session2/.venv/bin/python -m unittest discover -s session3/src -p 'test_*.py' -v
env PYTHONHASHSEED=0 session2/.venv/bin/python session3/src/log_command.py --name analyze -- session2/.venv/bin/python session3/src/analyze.py
```

Each output folder and command name must be fresh; existing evidence is never
overwritten. These commands are for a clean reproduction with the verified HW8
artifacts, not for rerunning over this completed local run. Metal must be allowed
by the operating-system sandbox; the initial sandbox worker exit was preserved in
`outputs/preflight/` and `evidence/commands/preflight.*`, with no generation.

`outputs/frozen/manifest.json` verifies every HW8 frozen file, model/config/data
hashes, teacher source hashes and all 100 transformations. Full run protocols bind
input, common protocol, model, decoding, backend and actual source hashes.
`outputs/analysis/control_reuse_proof.json` documents exact reuse of *local HW8*
classical and semantic outputs. Semantic 100/100 fallback is not successful LLM
quality. `configs/results.json` contains only aggregates, never client IDs.

After the reviewed local commit exists, render its English draft with:

```sh
session2/.venv/bin/python session3/src/make_report.py --experiment-commit HEAD
session2/.venv/bin/python session3/src/render_report.py
session2/.venv/bin/python session3/src/render_pdf.py
```

The individual Markdown/HTML/PDF remains local, excluded from Git. Personal
verification stays PENDING until Dmitry confirms the
specific new worksheet in `evidence/personal_check.md`. The final packet is one
individual PDF plus this session's genuine native export, captured after the
working turn ends. No fabricated export, remote push or course submission is
performed here.
