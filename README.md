# AIIA — FAR-Trans recommendation project

Repository: https://github.com/dimayakovlev014-del/aiia-far-trans

LLM4Rec, HSE University. Team: Alexander Selishchev, Dmitry Yakovlev, Sofya Semushkina.

## Status

Session 1 defined a Ranker study. Session 2 / HW8 implementation is being prepared. No student HW8 model inference or benchmark results have been produced yet.

The longer-term question compares a frozen LLM reranker with LambdaMART on identical behavioral inputs and fixed LightGCN candidates, with a secondary analysis of users who purchased 1–3 distinct assets before the cutoff. This is a research plan, not a measured claim or a statement that every method choice has been agreed by all members.

For the minimal HW8 pilot we will use the teacher starter's fixed CBF candidates and implement a behavioral-only LLM prompt with consistent asset aliases. This scoped pilot does not yet answer the complete LightGCN/LambdaMART question.

## Teacher starter

Original source: https://github.com/dryjins/RecSys-LLMs

Pinned teacher commit: `f6850861ea20a51e1fb018ee4e5f840c2ac759e0`.

The unchanged teaching packages are in:

- `vendor/RecSys-LLMs/project/far-trans`
- `vendor/RecSys-LLMs/project/far-trans-llm`

File hashes and provenance are recorded in `session2/reference/teacher_snapshot.json`. Teacher implementation and published reference results are credited to the original repository and must not be presented as this team's work or measurements.

## HW8 protocol

Keep all four provided baselines: Popularity, CBF, Latest buy, Latest buy or sell. Use the deterministic 100 warm-user sample, cutoff 2021-06-01 exclusive, a 30-day future Buy window, repeated purchases allowed, and Top-10 output. Report both all-user macro nDCG@10 (empty truth counts as zero) and nonempty-truth nDCG@10 with cohort counts.

These teaching settings do not define the final project benchmark. Its pending settings remain in `protocol/pending.json`.

Alexander Selishchev's Session 2 homework change (HW2 / HW8) (asset-name masking in the teacher LLM reranker) is in `selishchev_hw2/`.

See `session2/README.md` and the vendored READMEs for the current workflow. The original data must be obtained separately from https://researchdata.gla.ac.uk/1658/ and supplied through the starter's `--data-dir` argument. Do not commit the dataset.

## Submission and reproducibility

Each student submits one individual PDF report and a genuine AI-session export. The team repository does not replace either file; no second group PDF is required. Student code, exact inference settings, aggregate results and reproducible run commands will be added after implementation. Personal verification must reflect each student's actual check.

Raw data, model weights, instantiated user prompts, generations, per-user outputs, native AI logs, private discussion and personal reports are excluded from Git. Source prompt templates and reviewed aggregate results can be shared.

Repository preparation and documentation were produced by Dmitry Yakovlev with Codex assistance. Model quality, profit and investment suitability have not been established.
