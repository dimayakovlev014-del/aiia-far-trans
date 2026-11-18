"""Compose an evidence-linked English draft from completed measured outputs only."""
import argparse
from datetime import date
import json
import subprocess

from experiment import ROOT,dump
from prepare_data import sha


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-commit',required=True)
    args=parser.parse_args()
    commit=subprocess.check_output(['git','rev-parse',args.experiment_commit],cwd=ROOT,text=True).strip()
    comparison='session2/outputs/comparison/'
    metrics=read(comparison+'metrics.json')
    subgroup=read(comparison+'subgroup.json')
    coverage=read(comparison+'coverage.json')
    retrieval=read(comparison+'retrieval.json')
    personal=read('session2/evidence/personal_check.json')
    model=read('session2/configs/model.json')
    environment=read('session2/evidence/environment.json')
    decode=read('session2/configs/decoding.json')
    baseline=read('session2/evidence/commands/baselines_seed0.json')
    freeze=read('session2/evidence/commands/freeze_seed0.json')
    teacher=read('session2/reference/teacher_snapshot.json')['commit']
    backend=read('session2/outputs/behavioral_full/backend.json')
    fit=read('session2/outputs/behavioral_full/prompt_fit.json')
    both='semantic' in metrics
    names={'popularity':'Popularity','cbf':'CBF','latest_buy':'Latest buy','latest_buy_or_sell':'Latest buy or sell',
           'candidate_order':'CBF candidate order','behavioral':'Behavioral LLM + fallback','semantic':'Teacher semantic LLM + fallback'}
    table='| Method | Users | Nonempty | All nDCG@10 | Nonempty nDCG@10 | Runtime | Failed |\n|---|---|---|---|---|---|---|\n'
    timing={}
    for arm in coverage:
        pilot=read(f'session2/evidence/commands/{arm}_pilot.json')
        full=read(f'session2/evidence/commands/{arm}_full.json')
        timing[arm]=dict(pilot=pilot['elapsed_seconds'],full=full['elapsed_seconds'],
                         combined=pilot['elapsed_seconds']+full['elapsed_seconds'])
    for name in ('popularity','cbf','latest_buy','latest_buy_or_sell','candidate_order','behavioral','semantic'):
        if name not in metrics:
            continue
        m=metrics[name]
        runtime=f"{timing[name]['combined']:.1f} s R" if name in timing else ('shared B' if name!='candidate_order' else 'shared F')
        failed=coverage[name]['fallbacks'] if name in coverage else 0
        table+=f"| {names[name]} | {m['users']} | {m['nonempty_truth_users']} | {m['ndcg_at_10']:.4f} | {m['ndcg_at_10_nonempty']:.4f} | {runtime} | {failed} |\n"
    subtable='| Method | Short-history users | Nonempty | All nDCG@10 | Nonempty nDCG@10 |\n|---|---|---|---|---|\n'
    for name in ('popularity','cbf','latest_buy','latest_buy_or_sell','behavioral','semantic'):
        if name not in subgroup:
            continue
        m=subgroup[name]
        score=lambda v:'n/a' if v is None else f'{v:.4f}'
        subtable+=f"| {names[name]} | {m['users']} | {m['nonempty_truth_users']} | {score(m['ndcg_at_10'])} | {score(m['ndcg_at_10_nonempty'])} |\n"
    b=coverage['behavioral']
    delta=metrics['behavioral']['ndcg_at_10']-metrics['candidate_order']['ndcg_at_10']
    semantic_text=(f"The unchanged teacher semantic prompt was also run on all 100 users with the same model, backend, decoding and candidates. Its all-user score was {metrics['semantic']['ndcg_at_10']:.4f}, with {coverage['semantic']['valid']}/100 valid generations and {coverage['semantic']['fallbacks']} fallbacks. This compares two input packages: metadata/profile/ISIN history versus aliased numerical history and CBF information. Several fields and serialization change together, so it does not identify the causal effect of masking alone."
                   if both else 'The additional unchanged semantic-prompt arm was not completed. The observed difference from candidate order cannot be attributed specifically to the prompt/input intervention.')
    failure=ROOT/'session2/evidence/inspected_failure.json'
    failure_text='No invalid responses or generation failures were observed.'
    if failure.exists():
        f=json.loads(failure.read_text())
        r=f['record']
        failure_text=f"An AI-inspected {f['arm']} failure had validation reason `{r['error']}`. Its untouched raw text is retained; the entire original first-ten candidate list was used, with no repair model call. See `evidence/inspected_failure.json`."
    coverage_text=' '.join(f"{names[a]}: {v['generated']} generated, {v['valid']} valid, {v['parse_repairs']} parse extractions, {v['fallbacks']} fallbacks; valid-only diagnostic nDCG {v['valid_only_diagnostic']['ndcg_at_10']:.4f} over {v['valid_only_diagnostic']['users']} users."
                           if v['valid_only_diagnostic']['ndcg_at_10'] is not None else f'{names[a]}: no valid responses.' for a,v in coverage.items())
    report=f'''# Behavioral-Only LLM Reranking of Fixed FAR-Trans Candidates

**DRAFT — personal result verification PENDING; native session export pending after this session ends.**

**Student:** Dmitry Yakovlev. **Team:** AIIA (Alexander Selishchev, Dmitry Yakovlev, Sofya Semushkina). **Date:** {date.today().isoformat()}. **Assignment:** HW8 / Session 2.

**GitHub:** https://github.com/dimayakovlev014-del/aiia-far-trans (PUBLIC, authenticated access and existing remote HEAD verified). **Student experiment commit:** `{commit}`. This is a local source commit; this session does not claim a new remote upload. **Teacher source commit:** `{teacher}`.

## Abstract

This local pilot implements a behavioral-only, aliased LLM reranker of 50 fixed content-based filtering (CBF) candidates. All four provided classical baselines are retained. Under the homework protocol, the same 100 warm users contain only 11 users with future purchases. A frozen quantized Qwen2.5-0.5B model achieves fallback-inclusive all-user nDCG@10 of {metrics['behavioral']['ndcg_at_10']:.4f}, compared with {metrics['candidate_order']['ndcg_at_10']:.4f} for the original candidate order and {metrics['popularity']['ndcg_at_10']:.4f} for Popularity. There are {b['valid']} valid model responses and {b['fallbacks']} fallbacks. This is an input and reranking pilot, not the planned LightGCN/LambdaMART comparison. Personal verification remains pending.

## 1. Introduction

My accepted Session 1 question concerns whether a frozen local LLM helps users with 1–3 distinct past purchased assets compared with an information-matched LambdaMART reranker on fixed LightGCN candidates. HW8 narrows implementation to the supplied working starter: **CBF replaces the planned LightGCN generator**, and original CBF order is the required reranking control. Neither LightGCN nor LambdaMART is implemented here; the original primary research question remains unanswered. All 100 homework users remain in the common evaluation; short history is only a secondary subgroup.

The concrete pilot question is whether a zero-shot model can use cutoff-derived behavioral counts, recency and candidate scores/ranks to reorder those same candidates. Removing names, profiles and identities is an input intervention. It neither eliminates pretrained knowledge nor creates a pure reasoning comparison.

## 2. Related Work

The following literature descriptions reuse Codex's saved Session 1 source checks, with attribution; they were not reread or personally verified by Dmitry in this session. FAR-Trans [1] motivates retrospective financial purchase prediction and warns that its release includes estimated and reconstructed information. LightGCN [2] motivates the future collaborative generator, not a model run here. Hou et al. [3] provide the closest history-conditioned zero-shot ranking design and examine recency and presentation order. RankGPT [4] motivates preserving ranking outputs and treating order/format failures as empirical limitations; passage search is a different task. Spadea and Seneviratne [5] already apply LLM/KG approaches to FAR-Trans with a different retrieval and evaluation design. These studies do not establish the narrowly proposed same-information LightGCN/LambdaMART subgroup comparison. Current technical sources are the pinned teacher code [6], official Qwen model [7] and local runtime [8].

<!-- pagebreak -->

## 3. Method

### 3.1 Data and teaching protocol

The unchanged existing FAR-Trans ZIP was reused, not downloaded again; its SHA256 and four extracted CSV hashes are saved in `evidence/data_manifest.json`. The teacher's loader and scoring functions remain unchanged. Metadata and positive prices must exist strictly before **2021-06-01**. Exact duplicate full transaction rows and negative-ID reconstructed initial positions are excluded; the actual loader excludes **38,019** reconstructed-position rows and finds no exact duplicate exclusions. Filtering visible dates still does not certify faithful historical provenance: the release contains reconstructed interactions and estimated values. The earlier questionnaire-date problem remains a source limitation. No profiles or later semantic metadata enter the student prompt; CBF uses the teacher's timestamp-filtered asset categories, subcategories, sector, industry and market.

The cohort uses the teacher's seed-42 SHA256 ordering of as-of-known warm users, selecting 100 without consulting truth. Labels are distinct eligible Buy assets in **[2021-06-01, 2021-07-01)**. Repeat purchases are allowed. Binary macro nDCG@10 uses all future eligible positives for IDCG, including those outside Top-50. Empty truth contributes zero to the primary 100-user mean; the nonempty mean and its count are separate. These are teaching settings; `protocol/pending.json` remains unchanged for the final project.

### 3.2 Own change and controlled candidates

The teacher semantic ranker supplies profiles, asset metadata/ISINs, all historical Buy counts and the latest 15 dated Buy events. My student variant replaces that input builder, while reusing its strict JSON parser and unchanged CBF retrieval. Aliases are assigned by deterministic hash ordering of the union of history and candidates, independently of candidate rank, and are consistent across both. The full 50-candidate pool is retained. The prompt contains total/distinct history size, counts and last-Buy ages for each historical asset, the latest 15 Buy events as ages, and each candidate's alias, four-significant-digit CBF score, original rank, historical Buy count, last-Buy age and distinct-buyer popularity. All event-derived features use retained transactions strictly before cutoff. No customer identifier, original asset identifier, name, profile, prices or future label is in the actual student prompt. Candidate scores nevertheless carry metadata-derived information.

Source template: `prompts/behavioral.txt`. Exact instantiated inputs and alias maps are local-only `outputs/frozen/behavioral_requests.jsonl`; future truth is separate `evaluation.json`. Inference receives only the prompt. All candidates, labels and users are frozen before model calls. The first baseline run exposed process-dependent floating summation order in the teacher's Python sets. Its outputs and the failed equality check are preserved; canonical baselines and input freezing use **PYTHONHASHSEED=0**, keeping teacher score/tie rules intact.

### 3.3 Local model, reliability and resumption

The local pilot choice is **{model['repository']}**, revision `{model['revision']}`, Q4_K_M file `{model['filename']}`, Apache-2.0 license. Official LFS and downloaded SHA256 match; full digest is in `configs/model.json`. Hardware: Apple M2, 8 GB unified memory, arm64 macOS; Python 3.9.6 and llama-cpp-python {backend['llama_cpp_python']} with Metal. Exact package versions, runtime-library digests and embedded GGUF chat-template digest are saved. The embedded template receives one user message and its generation prefix. Settings: greedy temperature 0, seed 42, output at most 512 tokens, context {decode['n_ctx']}, batch {decode['n_batch']}, microbatch {decode['n_ubatch']}, four CPU threads, all layers requested on Metal, Flash Attention, no grammar or truncation. All complete student prompts fit: maximum {fit['max_input']} input tokens plus 512 output tokens.

Three real pilot users (the first three frozen cohort users) precede each full arm. This is a feasibility check, not a validation/tuning set: no prompt or model choice is selected by their future-purchase scores. Full-run gates use measured projected duration below two hours; compatible pilot calls are reused by a cache binding model revision/file, prompt, decoding, backend and worker source. Responses retain raw text, tokens, timings, finish reason and validation. Exactly ten distinct eligible candidate IDs are required. Fenced/embedded JSON extraction is recorded; no semantic editing or regeneration occurs. Invalid output uses the entire original candidate-order first ten. Timeouts, worker exits, model-load errors and generation exceptions have separate statuses and cannot become valid LLM coverage through fallback. A failed model load stops the experiment without measured metrics.

<!-- pagebreak -->

## 4. Experiments

### 4.1 Measured local results

All rows below are current local runs, not copied teacher scores. The four classical values agree with the teacher's published rounded reference values, but are independently executed here.

{table}

**Timing labels.** B = **{baseline['elapsed_seconds']:.2f} s** externally measured for the entire shared four-baseline CLI run, including loading/hashing/output. Per-method times were not measured. F = **{freeze['elapsed_seconds']:.2f} s** shared input freezing, including retrieval, feature/prompt preparation and hashes; candidate-order has no model call. R = externally measured pilot + resumed full-run commands, including two model loads and the full run's three cache hits; it is incremental to F, excludes dataset/model downloads, and is not a single cold end-to-end pipeline measurement.

{coverage_text}

The behavioral arm changes {b['changed_top10_users']}/100 Top-10 lists relative to candidate order. Its all-user paired mean difference is **{delta:+.4f}**. Pure generation time across the 100 unique behavioral requests, including the three pilot-origin calls once, is **{b['generation_seconds']:.1f} s**; median per-user generation is {b['median_generation_seconds']:.2f} s. Peak recorded behavioral worker process memory is {b['peak_worker_memory_bytes']/1024**3:.2f} GiB (process ru_maxrss, not total system/Metal memory). No statistical significance, causal effect, profit or suitability is claimed.

### 4.2 Retrieval and short-history diagnostics

Top-50 retrieves **{retrieval['retrieved_positives']}/{retrieval['total_positives']}** relevant assets, missing {retrieval['missed_positives']}; micro Recall@50 is {retrieval['recall50_micro']:.4f}, and macro Recall@50 among 11 nonempty users is {retrieval['recall50_nonempty_macro']:.4f}. Only {retrieval['nonempty_users_with_retrieved_positive']} nonempty users have a retrieved positive. A truth-aware candidate oracle, used only as a post-run diagnostic, reaches all-user nDCG@10 {retrieval['candidate_oracle_ndcg10']['ndcg_at_10']:.4f}. Future positives were never inserted into candidates.

The 1–3-distinct-assets subgroup contains 63 users and just **one** nonempty-truth user (one positive), so it cannot substantiate a general short-history benefit. Its zero-truth users are retained:

{subtable}

<!-- pagebreak -->

## 5. Discussion

### 5.1 Findings, failures and next experiment

The behavioral policy's measured difference from original CBF order is {delta:+.4f} all-user nDCG@10. Popularity scores {metrics['popularity']['ndcg_at_10']:.4f}. An LLM run is not evidence of an improvement merely because it reorders candidates; retrieval misses, fallback and the sparse 11-user label support restrict interpretation. {semantic_text}

{failure_text} Full coverage, failures and valid-only diagnostics are in `outputs/comparison/coverage.json`; the main table always includes fallback users. Preparation and download failures are also preserved, including TLS timeouts and a newer official Metal wheel that matched its release digest but failed ZIP validation; the validated 0.3.16 wheel was used instead. These are environment failures, distinct from actual inference failures.

The release's reconstruction/estimation provenance, one bank and period, quantization, greedy decoding, single presentation order, tiny labelled subgroup and CBF retrieval constrain generalization. Numeric masking removes useful semantic inputs as well as identities; the model's prior knowledge is still present. One next controlled experiment is a past-snapshot validation study of the same numerical schema on fixed LightGCN candidates, comparing LambdaMART with the LLM and original generator order. Fit/tune only on earlier windows, freeze before a new held-out evaluation, and retain all users with subgroup reporting. This would begin to address the original Session 1 question; it has not been executed here.

<!-- pagebreak -->

### 5.2 Team discussion and individual contribution

No new Session 2 team discussion was supplied. In the saved 9 November discussion, Sasha proposed Ranker/Top-50, short histories and masking, and later replaced MF with LightGCN. His forwarded 13 November update and individual report describe agreement on LightGCN/Ranker, the 1–3-distinct-assets subgroup, all-user evaluation and an additional learned comparator. Sofya's positions and those agreements remain attributed to **Sasha's account**; her original replies/full new chat are unavailable. Sasha retains semantic names/profiles in his own design; I accepted a behavioral-only, information-matched LLM/LambdaMART comparison. The final team model/input choices remain unresolved, and this local Qwen pilot is not a new collective decision.

My accepted individual direction is the behavioral-only Ranker question and subgroup definition, developed with Codex. This session used Codex to implement the input intervention, local backend, caching, validation, comparison and report. The four baselines, data policies, semantic prompt and parser are teacher source reuse. Prior literature/data-provenance evidence and preparation materials are shared inputs, not new personal work. No new implementation or verification by another team member is claimed.

### 5.3 Concrete result check — AI-performed, personal PENDING

Codex independently recalculated saved behavioral response **line {personal['jsonl_line']}**, using its mapped Top-10 and the separate future truth. Relevant ranks are **{personal['relevant_ranks']}**. DCG@10 = **{personal['dcg']:.10f}**, IDCG@10 = **{personal['idcg']:.10f}**, nDCG@10 = **{personal['ndcg']:.10f}**. Exact file paths, IDs, discounts and the division are in `session2/evidence/personal_check.md`. Dmitry has not yet confirmed inspecting this example; his earlier Session 1 questionnaire-date check is not a Session 2 result check. The report remains a draft until his actual check and its outcome are recorded.

## 6. AI Usage Disclosure

I used Codex in the established preparation/review and main-workflow arrangement. The separate preparation conversation read the assignment, related it to Session 1, saved the pinned teacher source and prepared the working prompt. This main workspace session reused those sources and the existing archive, wrote and ran code/tests, made actual local Qwen calls, independently checked inputs/metrics, inspected raw failures where present, and drafted/rendered this report. The 58 starter, 26 teacher LLM and eight student contract tests passed; synthetic generator tests are not model inference. Codex's checks are AI-performed, not my personal verification. No native transcript was edited or fabricated. This still-active session's final native JSONL must be captured by its cwd/session metadata after work ends; no active-log snapshot is presented as final. No paid API/cloud service, external message or course submission was used. Only one individual PDF plus the genuine session export is required.

<!-- references -->

## References

[1] J. Sanz-Cruzado, N. Droukas, and R. McCreadie. “FAR-Trans: An Investment Dataset for Financial Asset Recommendation.” Fin-RecSys, 2024. [Paper](https://arxiv.org/html/2407.08692v1); [official data](https://researchdata.gla.ac.uk/1658/). Literature checks reused from Session 1, 13 Nov. 2026.

[2] X. He et al. “LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.” SIGIR, 2020. doi:10.1145/3397271.3401063. [Full text](https://arxiv.org/html/2002.02126v4). Checks reused from Session 1.

[3] Y. Hou et al. “Large Language Models are Zero-Shot Rankers for Recommender Systems.” ECIR, 2024. [Full text](https://arxiv.org/html/2305.08845v2). Checks reused from Session 1.

[4] W. Sun et al. “Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents.” EMNLP, 2023, pp. 14918–14937. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.923/). Checks reused from Session 1.

[5] F. Spadea and O. Seneviratne. “Parallel and Multi-Stage Knowledge Graph Retrieval for Behaviorally Aligned Financial Asset Recommendations.” RAGE-KG, CEUR-WS 4079, 2025, pp. 97–106. [Full text](https://ceur-ws.org/Vol-4079/paper8.pdf). Checks reused from Session 1.

[6] dryjins. “RecSys-LLMs: FAR-Trans starter and LLM baselines.” [Pinned source](https://github.com/dryjins/RecSys-LLMs/tree/{teacher}/project/far-trans-llm). Preparation supplied snapshot; this session verified all 22 source hashes, read code and ran tests. Accessed 18 Nov. 2026.

[7] Qwen Team. “Qwen2.5-0.5B-Instruct-GGUF.” [Pinned official model](https://huggingface.co/{model['repository']}/tree/{model['revision']}). Revision `{model['revision']}`; Apache-2.0 license. File SHA256 `{model['sha256']}`. Official metadata/card/license checked, 18 Nov. 2026.

[8] abetlen/llama-cpp-python. [Official runtime documentation](https://github.com/abetlen/llama-cpp-python#installation); [Metal release 0.3.16](https://github.com/abetlen/llama-cpp-python/releases/tag/v0.3.16-metal). Runtime/Metal support, exact wheel digest and embedded template checked locally, 18 Nov. 2026.
'''
    folder=ROOT/'session2/report'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'report.md').write_text(report)
    dump(folder/'report_provenance.json',dict(status='DRAFT',personal_confirmation='PENDING',experiment_commit=commit,
         teacher_commit=teacher,source_of_results='Current local saved outputs, never teacher reference table',
         results_sha256={p.name:sha(p) for p in (ROOT/comparison).glob('*.json')},
         preparation_reuse=True,native_export='PENDING after active session ends'))
    ready=ROOT/'session2/submission'
    ready.mkdir(exist_ok=True)
    (ready/'READINESS.md').write_text('''# HW8 readiness — DRAFT

- Required four baselines, 100-user model run and AI checks are complete; see report and evidence.
- Personal result check: PENDING. Inspect `../evidence/personal_check.md` and record Dmitry's actual confirmation/outcome before finalizing.
- Native session export: PENDING. Capture the genuine main session only after its active work ends, matching cwd/session metadata; do not edit it or relabel an older log.
- Experiment source is committed locally; this session does not claim a new remote upload. The supplied PUBLIC team repository and existing remote HEAD were verified.
- Finalize the same individual report after confirmation; until then use `yakovlev_dmitry_session2_report_draft.pdf`. No second project/group PDF is required.
- No course submission, message, invitation or external upload was performed in this session.
''')
    print('Composed English draft from completed local evidence; commit '+commit)


if __name__=='__main__':
    main()
