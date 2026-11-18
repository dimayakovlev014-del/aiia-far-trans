# HW8 / Session 2

Prepared on 18 November 2026 for Dmitry Yakovlev, team AIIA.

The working implementation uses the unchanged four teacher baselines, frozen Top-50 CBF candidates and a behavioral-only aliased local LLM reranker. The actual run status and measured outputs live in ignored local `outputs/` and `evidence/` directories; no teacher reference score is a student run.

- Main working prompt: `../PROMPT_SESSION2.md`.
- Exact supplied assignment: `reference/homework2_supplied.md`.
- Teacher code: `../vendor/RecSys-LLMs/project/far-trans` and `far-trans-llm`.
- Teacher commit/hashes: `reference/teacher_snapshot.json`.
- Existing data: the unchanged Session 1 FAR-Trans archive; see local provenance.

Keep the four teacher baselines and default 100-user teaching protocol. The proposed student change is behavioral-only aliased LLM reranking of fixed CBF candidates. This pilot is a scoped implementation toward the earlier LightGCN/LambdaMART question, not a claim that the complete question has been tested.

One individual PDF (body <=5 pages plus references) and genuine AI-session export are required. The older project plan is internal working material, not a second required PDF. Personal confirmation stays **PENDING** until Dmitry actually confirms his result check; AI checks cannot satisfy that requirement. Capture this main session's genuine native export only after it ends.

Raw data, instantiated prompts, user-level outputs, model files, logs, personal evidence and reports are excluded from Git by default. Publish reviewed source and aggregate results only. The user created the public team repository https://github.com/dimayakovlev014-del/aiia-far-trans; it is configured locally as origin. Verify a successful push before claiming source has been uploaded.

## Reproduce on the measured Mac

Run from the project root. CPython **3.9.6 / macOS arm64** is used for both retrieval and inference; `PYTHONHASHSEED=0` stabilizes floating summation in the teacher's sets without changing its code or tie rules. Another Python implementation/version can change near-tie ordering. Preserve the frozen local candidate lists when resuming.

```sh
python3 -m venv session2/.venv
session2/.venv/bin/python session2/src/setup_runtime.py
session2/.venv/bin/python session2/src/prepare_data.py
session2/.venv/bin/python -m unittest discover -s vendor/RecSys-LLMs/project/far-trans -p 'test_*.py' -v
session2/.venv/bin/python -m unittest discover -s vendor/RecSys-LLMs/project/far-trans-llm -p 'test_*.py' -v
env PYTHONHASHSEED=0 session2/.venv/bin/python -m unittest discover -s session2/src -p 'test_*.py' -v
env PYTHONHASHSEED=0 session2/.venv/bin/python vendor/RecSys-LLMs/project/far-trans/starter.py run --data-dir session2/data --output-dir session2/outputs/baselines_seed0
env PYTHONHASHSEED=0 session2/.venv/bin/python session2/src/experiment.py freeze
session2/.venv/bin/python session2/src/audit_inputs.py
session2/.venv/bin/python session2/src/fetch_model.py
session2/.venv/bin/python session2/src/capture_environment.py
```

The archive is reused from the sibling `llm4rec_a07/reference/FAR-Trans.zip` and verified against the earlier provenance. It is not distributed here. The initial downloaded Metal 0.3.36 wheel failed ZIP validation despite matching its official digest; use the verified **0.3.16** Metal wheel pinned in `configs/runtime.json`, not the CUDA dependencies of the teacher's machine.

The user chose the smaller **Qwen2.5-0.5B-Instruct Q4_K_M** after measured network delays made downloading the proposed 3B model impractical. Its official revision, license, file digest and native 8192-token context are pinned in `configs/model_choice.json`. The final observed model file manifest is `configs/model.json`. No 3B inference is implied by its preserved partial downloads. The small model is a feasibility compromise, not a collective team choice or a test of the planned 3B configuration.

Wrap real commands with `src/log_command.py --name UNIQUE_NAME -- COMMAND...` to retain their raw stdout/stderr, exit status and externally measured wall time. Use the command names below for the report's timing lookup.

```sh
session2/.venv/bin/python session2/src/log_command.py --name behavioral_pilot -- session2/.venv/bin/python session2/src/run_inference.py --arm behavioral --phase pilot --output session2/outputs/behavioral_pilot
session2/.venv/bin/python session2/src/log_command.py --name behavioral_full -- session2/.venv/bin/python session2/src/run_inference.py --arm behavioral --phase full --output session2/outputs/behavioral_full
# Optional controlled teacher input arm, with identical model/backend/settings:
session2/.venv/bin/python session2/src/log_command.py --name semantic_pilot -- session2/.venv/bin/python session2/src/run_inference.py --arm semantic --phase pilot --output session2/outputs/semantic_pilot
session2/.venv/bin/python session2/src/log_command.py --name semantic_full -- session2/.venv/bin/python session2/src/run_inference.py --arm semantic --phase full --output session2/outputs/semantic_full
session2/.venv/bin/python session2/src/evaluate.py
session2/.venv/bin/python session2/src/make_report.py --experiment-commit YOUR_ACTUAL_STUDENT_COMMIT
session2/.venv/bin/python session2/src/render_report.py
```

Pilot: three frozen users, at most ten minutes. Full arm: all 100, measured pilot projection below two hours, preserving compatible cached responses. No candidate truncation; every full prompt is token-counted before inference. A native-context overflow in the optional semantic arm must be documented, not repaired by silently removing candidates or rewriting the teacher prompt. `--output` must be fresh to preserve evidence. A later compatible resume can use a fresh output folder and the shared cache; model/settings/prompt/source differences create separate keys. If rerunning completed analysis, choose a new derived directory instead of overwriting raw outputs.

`evaluate.py` independently recalculates nDCG, compares all common protocol fields and reparses raw generations. Main metrics retain all fallback and empty-truth users. It writes a concrete AI-performed check to `evidence/personal_check.md`, with personal confirmation **PENDING**. Rendering creates the English Markdown/HTML draft; print it locally to a `_draft.pdf` and check page count/layout. No report, source push, message or course submission is performed merely by running these scripts.

This HW8 operationalization deliberately uses CBF in place of planned LightGCN and does not implement LambdaMART. The original information-matched LightGCN/LambdaMART research question remains pending. `../protocol/pending.json` is preserved.
