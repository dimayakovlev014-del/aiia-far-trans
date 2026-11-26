"""English five-body-page draft, generated from audited local aggregates."""
import argparse
import subprocess
from common import ROOT, S3, dump, read, sha


def fmt(value):
    return 'n/a' if value is None else f'{value:.4f}'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment-commit',required=True)
    a=p.parse_args()
    commit=subprocess.check_output(['git','rev-parse',a.experiment_commit+'^{commit}'],cwd=ROOT,text=True).strip()
    r=read(S3/'configs/results.json')
    m,c,t=r['metrics'],r['coverage'],r['runtime']
    pm,pa,dv=r['paired_main_vs_cbf'],r['paired_alias_vs_main'],r['dual_valid_diagnostic']
    retrieval=r['retrieval']
    cases=read(S3/'outputs/analysis/cases.json')
    check=read(S3/'evidence/personal_check.json')
    assert check['status']=='PENDING' and r['personal_verification']=='PENDING'
    names=[('popularity','Popularity'),('cbf','CBF'),('latest_buy','Latest buy'),
           ('latest_buy_or_sell','Latest buy or sell'),('candidate_order','Initial CBF order'),
           ('semantic','Teacher semantic + fallback'),('main','HW9 main + fallback'),('ablation','HW9 reversed aliases + fallback')]
    table='| Method | All 100 | Nonempty 11 | Valid / fallback | Origin |\n|---|---|---|---|---|\n'
    for key,label in names:
        counts=f"{c[key]['valid']} / {c[key]['fallbacks']}" if key in c else 'n/a / 0'
        table+=f"| {label} | {fmt(m[key]['ndcg_at_10'])} | {fmt(m[key]['ndcg_at_10_nonempty'])} | {counts} | {'new HW9' if key in ('main','ablation') else 'local HW8'} |\n"
    reasons=sorted(set(c['main']['validation_reasons'])|set(c['ablation']['validation_reasons'])|set(c['semantic']['validation_reasons']))
    labels={'ValueError: Duplicate ISINs':'Duplicate IDs',
            'ValueError: Expected exactly ten string ISINs':'Wrong count / non-string list',
            'ValueError: IDs outside candidate pool':'Outside candidate pool',
            'ValueError: ISIN outside candidate set':'Unknown alias / ID',
            'ValueError: No JSON array found':'No JSON array',
            'ValueError: Malformed JSON array':'Malformed JSON'}
    errors='| Invalid-output category | Main | Reversed | Semantic HW8 |\n|---|---|---|---|\n'
    for reason in reasons:
        errors+='| '+labels.get(reason,reason)+' | '+' | '.join(str(c[x]['validation_reasons'].get(reason,0)) for x in ('main','ablation','semantic'))+' |\n'
    case_text=[]
    for key,label in [('first_invalid','First invalid'),('first_dual_valid_changed','First dual-valid change'),
                      ('first_comparable_no_effect','First dual-valid unchanged'),('first_nonempty_ndcg_difference','First nonempty score change')]:
        case=cases[key]
        if not case['class_count']:
            case_text.append(f'**{label}:** this class contains 0 users.')
            continue
        line=case['cohort_line']
        ma,ab=case['main'],case['ablation']
        parts=case['components']
        if key=='first_invalid':
            reason=ma['error'] if not ma['valid'] else ab['error']
            detail=f'Validation reports `{reason}`. The failing arm uses the complete initial CBF first ten; no regeneration or partial-list merge occurs.'
        elif key=='first_dual_valid_changed':
            overlap=len(set(ma['ranking'])&set(ab['ranking']))
            detail=f'Both outputs validate after mapping to original assets, but their ordered Top-10 differs ({overlap}/10 common assets). There are {len(case["truth"])} future positives; nDCG is {fmt(parts["main"]["ndcg_at_10"])} versus {fmt(parts["ablation"]["ndcg_at_10"])}.'
        elif key=='first_comparable_no_effect':
            detail=f'Both outputs validate and decode to the identical Top-10. The common score is {fmt(parts["main"]["ndcg_at_10"])}; {len(case["truth"])} future positives are present.'
        else:
            missing=len(case['truth'])-len(case['retrieved_truth'])
            cause=('The score difference follows an invalid output switching to original-order fallback.'
                   if ma['fallback'] or ab['fallback'] else
                   'Both outputs are valid: movement of the retrieved positive within Top-10 explains the score change, without a fallback switch.')
            detail=(f'Main valid={ma["valid"]}, reversed valid={ab["valid"]}. Relevant ranks change from '
                    f'{parts["main"]["relevant_ranks"]} to {parts["ablation"]["relevant_ranks"]}; '
                    f'nDCG changes {fmt(parts["main"]["ndcg_at_10"])} → {fmt(parts["ablation"]["ndcg_at_10"])}. '
                    f'{missing}/{len(case["truth"])} future positives are outside Top-50. '
                    +cause)
        case_text.append(f'**{label}:** cohort line {line} (class size {case["class_count"]}). {detail}')
    body=f'''# Alias Sensitivity in a Local Financial Recommendation Ranker

**Dmitry Yakovlev · AIIA · 26 November 2026 · HW9 / Session 3**

**Repository:** [aiia-far-trans](https://github.com/dimayakovlev014-del/aiia-far-trans). **Local experiment commit:** `{commit}` (prepared locally, not pushed in this session). **Draft:** HW9 personal verification PENDING.

## Abstract

We test whether one consistent renaming of opaque item aliases changes a frozen local LLM Ranker on fixed financial-asset candidates. This narrow development adaptation reuses the HW8 CBF Top-50 and the same 100 warm users, only 11 of whom have future eligible purchases. Two new, cache-free Qwen2.5-0.5B-Instruct Q4_K_M runs use original behavioral prompts or a per-user lexicographic alias reversal, preserving all numerical features and row order. Fallback-inclusive all-user nDCG@10 is **{fmt(m['main']['ndcg_at_10'])}** for main, **{fmt(m['ablation']['ndcg_at_10'])}** for reversed aliases and **{fmt(m['candidate_order']['ndcg_at_10'])}** for initial CBF order. Renaming changes {pa['changed_top10_order']}/100 final lists; valid coverage is {c['main']['valid']}/100 and {c['ablation']['valid']}/100. The result establishes sensitivity to this particular naming, not a general benefit, a leakage test or the planned LightGCN/LambdaMART comparison.

## 1. Introduction

The Session 1 direction asks whether an LLM helps users with 1–3 distinct historical purchased assets as a Ranker, with LightGCN retrieval and a matched LambdaMART comparator. HW9 continues that direction through a **development adaptation**: retrieval is the working HW8 CBF; the classical matched control is its unchanged candidate order. LightGCN and LambdaMART remain separate future work. The LLM is a Ranker only: it changes the selected ten and their order, without learning embeddings, training a retriever or adding candidates.

The narrow question is: **with users, features, candidates and decoding fixed, does one bijective renaming of item aliases change the final recommendations, validity and nDCG@10?** The cohort was already used in HW8. The design and permutation rule were supplied before the new runs; no users or settings were chosen from new outcomes. The all-user result is primary; short history remains a diagnostic subgroup.

## 2. Related Work

Hou et al. [1] study history-conditioned zero-shot recommendation ranking and show sensitivity to candidate presentation order. RankGPT [2] generates permutations of identifiers and also tests initial passage order; passage search differs from financial purchase prediction. Here row order stays fixed, isolating one alias intervention rather than an order shuffle. Hua et al. [3] show that indexing matters in trained P5 recommendation models. That motivates examining identifiers but does not predict the effect of renaming aliases in a frozen model. Codex reopened these primary texts in HW9: [1] §2.2/§3.2, [2] §3.1/§6.5/Appendix B.3, and [3] §1/§3.

FAR-Trans [4] provides the financial setting; its reconstructed interactions and estimated values restrict historical interpretation. The LightGCN design [5] and FAR-Trans provenance checks are reused, with attribution, from Session 1. Teacher source [6], the pinned Qwen model [7] and Metal runtime [8] define the implemented adaptation. Prior Qwen/license/runtime checks come from HW8; actual model, template and library hashes were verified anew here.

<!-- pagebreak -->

## 3. Method

### 3.1 Fixed development protocol

The unchanged teacher selects the first 100 as-of-known warm users in ascending SHA256(`42:` + customer ID) order, without future-truth filtering. The cutoff is **2021-06-01**; labels are distinct eligible Buy assets in **[2021-06-01, 2021-07-01)**. Repeat purchases are allowed. Assets require metadata and positive prices strictly before cutoff. Exact duplicate transaction rows and negative-ID reconstructed initial positions are excluded by the provided loader. Date filtering does not certify provenance of all reconstructed data.

All comparisons use the same users, eligible assets, labels and binary macro nDCG@10. For each user, DCG is the sum of 1/log2(rank+1) over relevant recommendations; IDCG uses the first min(10, number of all future positives) ideal ranks, including positives outside Top-50. Empty truth scores zero. We report both all-100 and nonempty-11 means. The official final protocol/common test set is still pending; this reused development set is not an official benchmark.

### 3.2 Main and single ablation

The new main run uses exactly the HW8 behavioral prompts, verified **byte for byte** for all 100 users. Aliases cover the union of historical and candidate assets. Inputs contain pre-cutoff buy counts and ages, the latest 15 Buy events in their original order, and every candidate's CBF score, original rank, buy count, last-buy age and distinct-buyer popularity. CBF scores convey metadata-derived information. No raw customer/asset IDs, names, profiles or future truth are in the behavioral prompt. The worker receives only that prompt; truth and alias decoding remain in separate local evaluation artifacts.

For each user, sort existing aliases lexicographically and map them to that same list in reverse. Apply this bijection only to alias cells in `history`, `recent_15_buys_oldest_first`, `candidates` and the decoder. Candidate/history rows, original ISINs, features, instruction, model, parser, seed and fallback remain fixed. Inversion recovers each original payload and decoder; ordered candidate ISINs agree for all 100. Complete inputs contain {r['tokens']['main_min']}–{r['tokens']['main_max']} tokens in both arms, with **{r['tokens']['paired_count_differences']} paired count differences** and no truncation. Token identities change in {r['tokens']['input_token_id_hashes_differ']}/100 pairs, so equal length does not mean identical tokenization. This is one naming-sensitivity test, not removal of features, proof of leakage, or an average over permutations.

### 3.3 Execution and output policy

Both arms use the existing **Qwen2.5-0.5B-Instruct GGUF Q4_K_M**, revision `9217f5db79a29953eb74d5343926648285ec7e67`, model SHA256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`. The unchanged HW8 worker uses CPython 3.9.6 / llama-cpp-python 0.3.16 on Apple M2, 8 GiB RAM, arm64 macOS. Native logs confirm Metal and 25/25 layers offloaded. Embedded chat template and runtime-library hashes match HW8.

Exact settings: temperature 0, seed 42, context 8192, max output 512, top-p 1, top-k 0, min-p 0, repeat penalty 1, four threads, batch 512, microbatch 128, Flash Attention, stop tokens `<|im_end|>` / `<|endoftext|>`, no grammar and no truncation. All 200 complete prompts were counted before the first generation. Each arm then made three technical pilot calls and 100 new full-run calls. No cache was read or written, even for duplicate prompts or pilot rows. Loaded weights persist within each command; the worker resets token state for every generation.

The reused teacher parser requires exactly ten distinct candidate IDs after alias decoding. Fenced/embedded JSON extraction is logged as a repair; no IDs are edited or regenerated. An invalid answer replaces the entire output with the initial CBF first ten. Every user therefore has a legal final list, while invalid generation remains an explicit failure.

<!-- pagebreak -->

## 4. Experiments

### 4.1 Main results and controls

Every row uses the same 100 users (11 nonempty). All scores below are measured locally. Four classical controls and the semantic comparator reuse verified HW8 outputs, **not** the teacher's published results. Exact saved inputs, prediction files, common protocol, model/decoding/backend, cache descriptors and raw parsing were checked before reuse. The semantic prompt retains its original code and full candidate order, but all 100 responses fail validation: that row measures CBF fallback, not successful semantic LLM ranking.

{table}

Main minus initial CBF has paired all-user mean **{pm['mean_paired_delta']:+.4f}**: {pm['improved']} users improve, {pm['worsened']} worsen and {pm['tied']} tie. Main changes {pm['changed_top10_order']} ordered lists. Reversed aliases minus main has mean **{pa['mean_paired_delta']:+.4f}**: {pa['improved']} improve, {pa['worsened']} worsen and {pa['tied']} tie. These are descriptive paired results, without a significance or population-level benefit claim. Evidence retains unrounded per-user numerators, denominators and deltas; displayed nDCG is rounded to four decimals.

### 4.2 Coverage and subgroup

CBF Top-50 retrieves **{retrieval['retrieved_positives']}/{retrieval['total_positives']}** future positives (micro Recall@50 {fmt(retrieval['recall50_micro'])}); macro Recall@50 over the 11 nonempty users is {fmt(retrieval['recall50_nonempty_macro'])}. Only {retrieval['nonempty_users_with_retrieved_positive']}/11 have any retrieved positive. A post-run truth-aware candidate oracle reaches all-user nDCG@10 {fmt(retrieval['candidate_oracle_ndcg10']['ndcg_at_10'])}; labels were never used to insert candidates. The 1–3-distinct-assets subgroup has **{retrieval['short_history_users']} users, {retrieval['short_history_nonempty']} nonempty**, and {retrieval['short_history_retrieved_positives']} retrieved positives. Every method scores {fmt(r['subgroup']['main']['ndcg_at_10'])} there (nonempty mean {fmt(r['subgroup']['main']['ndcg_at_10_nonempty'])}). This subgroup cannot support a short-history benefit claim.

### 4.3 Runtime and audit

HW9 measured preparation/hash/transformation time is **{t['hw9_preprocessing_seconds']:.3f} s** (the prior HW8 data/retrieval preparation is reused). Complete-input preflight costs {t['hw9_preflight_external_seconds']:.2f} s externally. Main full generation totals **{t['main']['full_generation_seconds']:.2f} s**, versus **{t['ablation']['full_generation_seconds']:.2f} s** for reversal. Model initialization inside the worker takes {t['main']['full_worker_load_seconds']:.3f}/{t['ablation']['full_worker_load_seconds']:.3f} s; broader worker startup takes {t['main']['full_worker_startup_wall_seconds']:.3f}/{t['ablation']['full_worker_startup_wall_seconds']:.3f} s, including hash/import overhead. Pilot + full external command times are {t['main']['external_pilot_plus_full_seconds']:.2f}/{t['ablation']['external_pilot_plus_full_seconds']:.2f} s. All **206** generation calls are new; pilots are not reused. Summing successful preparation, preflight and both pilot/full commands gives {t['hw9_bounded_pipeline_external_seconds']:.2f} s, excluding idle coding, analysis, PDF rendering, downloads and the failed sandbox startup. No training occurs.

Reused HW8 timing: four baselines share {t['reused_hw8_controls']['classical_four_shared_external_seconds']:.2f} s, input freezing shares {t['reused_hw8_controls']['freezing_shared_seconds']:.2f} s, and semantic pilot + full commands total {t['reused_hw8_controls']['semantic_external']['combined_seconds']:.2f} s. Semantic full has {t['reused_hw8_controls']['semantic_full_cache_hits']} cache hits; these are historical timings, not HW9 generated tokens. The first sandbox worker exited with -11 before generation; its raw log and failure summary remain separate from successful Metal execution.

The frozen manifest checks model, decoding, data, teacher sources, every HW8 frozen file and all 100 input pairs. Common and separate run hashes are saved. Independent DCG/IDCG recomputation agrees with both HW8 and teacher metrics; all 100 users, truth, candidate order, output legality and raw parse/fallback were checked.

<!-- pagebreak -->

### 4.4 Alias sensitivity and failures

| Diagnostic | All 100 | Dual-valid only |
|---|---|---|
| Users | 100 | {dv['users']} ({dv['nonempty_truth_users']} nonempty) |
| Changed ordered Top-10 | {pa['changed_top10_order']} | {dv['changed_top10_order']} |
| Exact full Top-10 order | {pa['exact_top10_order']} | {dv['exact_top10_order']} |
| Exact Top-10 set | {pa['exact_top10_set']} | {dv['exact_top10_set']} |
| Mean set Jaccard | {fmt(pa['mean_top10_jaccard'])} | {fmt(dv['mean_top10_jaccard'])} |
| nDCG improved / worsened / tied | {pa['improved']} / {pa['worsened']} / {pa['tied']} | {dv['improved']} / {dv['worsened']} / {dv['tied']} |
| Mean paired nDCG delta | {pa['mean_paired_delta']:+.4f} | {fmt(dv['mean_paired_delta'])} |

Among dual-valid requests, main/reversed all-user means are {fmt(dv['metrics']['main']['ndcg_at_10'])}/{fmt(dv['metrics']['ablation']['ndcg_at_10'])}; their nonempty means are {fmt(dv['metrics']['main']['ndcg_at_10_nonempty'])}/{fmt(dv['metrics']['ablation']['ndcg_at_10_nonempty'])}. This conditions on the response outcome and is diagnostic only. The primary comparison retains every fallback and empty truth set. Valid outputs need JSON extraction in {c['main']['parser_repairs']}/{c['ablation']['parser_repairs']} cases; these repairs do not change selected IDs. Ascending-alias valid lists occur {c['main']['ascending_alias_order_valid']}/{c['main']['valid']} versus {c['ablation']['ascending_alias_order_valid']}/{c['ablation']['valid']} times. This is an output-pattern observation, not a causal explanation of model internals.

{errors}

These categories are mutually exclusive terminal validation reasons; a response can have additional defects masked by the first failed check. Full main/reversed generations finish by length limit in {c['main']['finish_reasons'].get('length',0)}/{c['ablation']['finish_reasons'].get('length',0)} cases, otherwise by stop. All model calls complete; no timeout or generation exception is hidden by fallback. The validity cross-table (main/reversed) is {', '.join(k+': '+str(v) for k,v in sorted(r['validity_cross_table'].items()))}.

### 4.5 Deterministic case analysis

Cases are selected after scoring in the existing cohort order, with no response-driven reruns. Local cohort line numbers identify saved rows without publishing client IDs. Untouched generations, alias maps, final lists and score components are retained in `outputs/analysis/cases.json`.

{(chr(10)*2).join(case_text)}

<!-- pagebreak -->

## 5. Discussion

### 5.1 Answer and limitations

One preregistered alias reversal changes **{pa['changed_top10_order']}/100** ordered final lists and yields an all-user paired nDCG change of **{pa['mean_paired_delta']:+.4f}**. Even holding row order and token count fixed, particular token identities can affect generation and whether the policy falls back. The dual-valid comparison separates some naming sensitivity from invalid-output fallback, but is conditional on validity. We cannot infer which numerical features the model used or generalize to all permutations, models or users. This is not an experiment on feature removal or leakage.

Only 11 future-truth users and six with retrieved positives support score changes; 89 empty truth sets contribute zero to the primary mean. Equal scores often coexist with changed lists. CBF retrieval misses 12 of 18 positives. The short-history subgroup has one labelled user with no retrieved positive, so a model operating on these candidates cannot demonstrate a subgroup gain. Quantization, a very small model, one fixed decoding/period, reuse of the development cohort and FAR-Trans reconstructed/estimated provenance further limit interpretation. We claim neither financial suitability nor a validated investment policy. The official final evaluation remains reserved for the announced common protocol/set; HW9 does not wait for its release.

### 5.2 Team and individual contribution

The saved 9 November discussion records Sasha's Ranker/Top-50 and masking suggestions and his change from MF to LightGCN. His forwarded 13 November update and individual report describe agreement on all-user evaluation, the 1–3-distinct-assets subgroup and an additional learned comparator; Sofya's positions and these agreements are attributed to Sasha's account because original later replies are unavailable. Dmitry accepted a behavioral, information-matched LLM/LambdaMART direction with Codex's help. The model, semantic inputs and final learned comparison remain unresolved.

Remote main was fetched and this HW9 branch starts from `d206088b7c2f7301a8ac1855c24ee985b3b83fe9`, preserving Sasha's `16a4d4b` and `d206088` commits. Their separate `selishchev_hw2/` records name masking and CPU/MPS work, not this HW9 ablation or a new team decision. Codex inspected repository history and aggregate documentation; no direct numerical comparison is made because exact private protocol/input identity is unproved. No new team conversation was supplied.

Dmitry's recorded contribution is his accepted research direction, request for this controlled experiment and previous HW8 personal result check. Codex implemented and ran the HW9 layer, audited the saved evidence and drafted this report. Teacher baselines/parser, prior HW8 worker/model/inputs and saved provenance are reused, not claimed as new student implementation or independent work by another member.

### 5.3 Personal check — PENDING

The new worksheet `session3/evidence/personal_check.md` selects cohort **line {check['cohort_line']}**, separately from HW8 line 30 where possible. It gives both saved Top-10 lists, future purchases, relevant ranks, DCG, IDCG and division. Codex's independent calculation is evidence of an AI check. **Dmitry has not yet confirmed this HW9 check**, and the draft does not claim personal verification is complete. The final report must reflect the actual confirmation, not the old HW8 check.

In that AI-worked example, one future positive moves from rank 2 to rank 3 in two valid lists. DCG is 1/log2(3) = 0.630929753571 versus 1/log2(4) = 0.5; IDCG is 1, hence nDCG is 0.6309 versus 0.5000. The matching raw future Buy is on CSV line 236200; its inspection is an AI check, pending Dmitry's own comparison and calculation.

## 6. AI Usage Disclosure

The preparation/review conversation supplied `PROMPT_SESSION3.md`, the assignment and fixed `design.json`. This working session reused them and the earlier archive, source, model and local HW8 control outputs. Codex implemented the alias transformation and new paths/runner, executed 206 real local calls, ran five contract tests, verified controls, calculated metrics/cases and drafted/rendered the report. It reopened the three primary texts in §2; FAR-Trans/LightGCN and model-license checks are attributed to earlier sessions. No personal reading or HW9 verification by Dmitry is invented.

This PDF is a draft. A genuine native export of this working session must be captured after the turn ends; no older/peer log or manufactured transcript substitutes for it. The final individual packet is one PDF plus one session export. No remote push, external message or course submission is made here. Reproduction commands are in `session3/README.md`; private raw evidence is intentionally excluded from public Git.

<!-- references -->

## References

[1] Y. Hou et al. “Large Language Models are Zero-Shot Rankers for Recommender Systems.” ECIR, 2024. [Primary text](https://arxiv.org/html/2305.08845v2). Opened by Codex, 26 Nov. 2026, §2.2/§3.2.

[2] W. Sun et al. “Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents.” EMNLP, 2023, pp. 14918–14937. doi:10.18653/v1/2023.emnlp-main.923. [Primary PDF](https://aclanthology.org/2023.emnlp-main.923.pdf). Opened by Codex, 26 Nov. 2026, §3.1/§6.5/Appendix B.3.

[3] W. Hua, S. Xu, Y. Ge, and Y. Zhang. “How to Index Item IDs for Recommendation Foundation Models.” arXiv:2305.06569v3, 2023. [Primary text](https://arxiv.org/html/2305.06569v3). Opened by Codex, 26 Nov. 2026. P5 indexing, not an alias-invariance result for this model.

[4] J. Sanz-Cruzado, N. Droukas, and R. McCreadie. “FAR-Trans: An Investment Dataset for Financial Asset Recommendation.” Fin-RecSys, 2024. [Paper](https://arxiv.org/html/2407.08692v1); [official data](https://researchdata.gla.ac.uk/1658/). Session 1 checks reused with attribution.

[5] X. He et al. “LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.” SIGIR, 2020. doi:10.1145/3397271.3401063. [Paper](https://arxiv.org/abs/2002.02126). Session 1 checks reused; not run here.

[6] dryjins. “RecSys-LLMs: FAR-Trans starter and LLM baselines.” [Pinned source](https://github.com/dryjins/RecSys-LLMs/tree/f6850861ea20a51e1fb018ee4e5f840c2ac759e0/project). Teacher commit `f6850861ea20a51e1fb018ee4e5f840c2ac759e0`; all snapshot hashes verified locally in HW9.

[7] Qwen Team. “Qwen2.5-0.5B-Instruct-GGUF.” [Official revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/9217f5db79a29953eb74d5343926648285ec7e67). Apache-2.0; existing Q4_K_M file SHA256 reverified in HW9. Official card/license checks reused from HW8.

[8] abetlen/llama-cpp-python. [Official Metal 0.3.16 release](https://github.com/abetlen/llama-cpp-python/releases/tag/v0.3.16-metal). Pinned HW8 runtime reused; import, backend, embedded template and dylib hashes reverified in HW9.
'''
    folder=S3/'report'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'report.md').write_text(body)
    dump(folder/'report_provenance.json',dict(experiment_commit=commit,results_sha256=sha(S3/'configs/results.json'),
        report_sha256=sha(folder/'report.md'),personal_verification='PENDING',
        model_sha256=read(ROOT/'session2/configs/model.json')['sha256'],
        common_protocol_hash=r['common_protocol_hash'],run_protocol_hashes=r['run_protocol_hashes']))
    print('Wrote English draft for local commit '+commit)


if __name__=='__main__':
    main()
