"""Build the Word report only from completed, independently checked metrics."""
import json,math
from pathlib import Path
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs'

def fmt(v):return '—' if v is None else f'{v:.4f}'
def build():
    r=json.loads((HERE/'results.json').read_text());assert r['status']=='complete' and r.get('independent_verification',{}).get('passed')
    m=r['manifest'];cfg=m['config'];b=r['metrics']['behavioral'];e=r['metrics']['profile_enriched'];delta=e['ndcg_at_10']-b['ndcg_at_10']
    check=r['personal_verification'];u=check['user_id'];truth=set(check['truth']);methods=check['methods']
    previous=json.loads((HERE.parent/'semushkina_hw2/results.json').read_text())
    truths={x['user_id']:set(x['truth']) for x in previous['frozen']}
    d=Document();sec=d.sections[0];sec.page_width=Inches(8.27);sec.page_height=Inches(11.69)
    sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.7)
    normal=d.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10.5)
    normal.paragraph_format.space_after=Pt(5);normal.paragraph_format.line_spacing=1.02
    for name,size in [('Title',19),('Heading 1',13),('Heading 2',11)]:
        s=d.styles[name];s.font.name='Times New Roman';s.font.size=Pt(size);s.font.color.rgb=RGBColor(0,0,0)
        s.paragraph_format.space_before=Pt(8);s.paragraph_format.space_after=Pt(4)
    for border in list(d.styles.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
    def p(text,style=None):return d.add_paragraph(text,style)
    def heading(text):d.add_heading(text,level=1)
    def table(headers,rows):
        t=d.add_table(rows=1,cols=len(headers));t.style='Light Shading Accent 1'
        for c,h in zip(t.rows[0].cells,headers):c.text=str(h)
        for row in rows:
            cells=t.add_row().cells
            for c,value in zip(cells,row):c.text=str(value)
        for row in t.rows:
            pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.space_after=Pt(2);par.paragraph_format.space_before=Pt(2)
                    for run in par.runs:run.font.name='Times New Roman';run.font.size=Pt(9)
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        return t
    p('Investor profile value in pointwise financial asset reranking','Title')
    p('Sofya Semushkina | HSE University | Homework 3 | 27 November 2026\nTeam AIIA: Dmitry Yakovlev, Aleksandr Selishchev, Sofya Semushkina\nRepository: https://github.com/dimayakovlev014-del/aiia-far-trans\nExperiment commit: '+m['base_commit'])
    heading('Abstract')
    p(f'This development experiment tests whether investor segment, risk tolerance and investment capacity improve financial asset purchase reranking once list-generation failures are removed. A frozen Qwen3-Reranker-0.6B scores the same CBF Top-10 independently for 100 investors. One ablation removes only these three profile fields. Mean nDCG@10 changes from {fmt(b["ndcg_at_10"])} to {fmt(e["ndcg_at_10"])} across all users and from {fmt(b["ndcg_at_10_nonempty"])} to {fmt(e["ndcg_at_10_nonempty"])} across 11 users with future purchases. The absolute primary difference is {delta:+.4f}. Both methods produce 100 valid rankings with no repairs or fallback. This single, small development sample does not establish statistical significance or investment suitability.')
    heading('Introduction')
    p('The team studies zero-shot LLM reranking for financial asset recommendation, including context value, robustness and reproducibility. My question is whether the investor profile improves reranking when candidates are scored independently. H1 predicts a positive nDCG@10 difference relative to an otherwise identical pointwise ranker without segment, risk tolerance and capacity. The null hypothesis is no improvement. HW2 found equal aggregate scores but 90 enriched listwise failures, leaving profile signal confounded with output reliability. HW3 removes this failure mechanism; it does not redefine the shared project. CBF remains the teaching proxy for stronger final-project retrieval, including LightGCN.')
    heading('Related Work')
    p('FAR-Trans [1] provides financial transactions and contextual records; observed purchases are the target rather than profit. Hou et al. [2] motivate frozen LLM candidate ranking, while the Qwen3 model card [3] specifies relevance scoring from yes/no logits. HW2 supplies the directly relevant prior comparison [4]. The new comparison changes model and candidate pool as well as scoring architecture, so HW2-to-HW3 differences cannot isolate the effect of pointwise scoring. The controlled causal contrast within HW3 removes only the three profile fields. No new broad literature review or trained control is added.')
    heading('Method')
    p('The LLM acts only as a Ranker. For each user it receives the original CBF first ten, scores each separately, and sorts by logit(yes) minus logit(no); exact ties retain CBF order. No candidates can be added. The shared query contains the cutoff, repeat-purchase policy, buy totals, five most frequent historical assets with counts/recency and the three latest buys. The candidate document contains its opaque alias, CBF score/rank, buy count/recency, historical distinct buyers and temporally admissible type/subtype/sector/industry. Both arms use identical instructions and candidate text. Only segment, risk_tolerance and investment_capacity_band are appended in the enriched query. No real asset names or raw ISINs enter the model.')
    p('Records are strictly before 2021-06-01. Questionnaire-derived fields also require an admissible questionnaire date no later than the record; unverifiable predictions and dates stay unknown. HW2 temporal joins are reused read-only and independently checked against raw historical transactions and metadata. All 100 users remain. Complete prompts contain 243–351 tokens, below max_length 512, with no truncation. The official non-thinking yes/no prefix and suffix are used. Scores are not calibrated purchase probabilities.')
    d.add_page_break()
    heading('Experiments')
    p('The fixed teaching setup uses the same deterministic 100 users, eligible assets and binary labels as HW2: cutoff 2021-06-01, evaluation [2021-06-01, 2021-07-01), repeat purchases allowed. Empty truth counts as zero; 11 users have future positives. IDCG uses every eligible future positive, including those outside Top-10. Four required teacher baselines and HW2 LLM metrics are reused under the verified common protocol hash. Original CBF order is the matched control for both new arms.')
    labels={'popularity':'Popularity','cbf':'CBF','latest_buy':'Latest buy','latest_buy_or_sell':'Latest buy or sell','candidate_order':'Original CBF order','behavioral':'HW2 behavior listwise','profile_enriched':'HW2 profile listwise'}
    rows=[[labels[a],fmt(v['ndcg_at_10']),fmt(v['ndcg_at_10_nonempty'])] for a,v in r['hw2_metrics'].items()]
    rows += [[label,fmt(r['metrics'][a]['ndcg_at_10']),fmt(r['metrics'][a]['ndcg_at_10_nonempty'])] for a,label in [('behavioral','HW3 behavior pointwise'),('profile_enriched','HW3 profile pointwise')]]
    table(['Method','All 100','Nonempty 11'],rows)
    table(['HW3 arm','Valid rankings','Invalid','Repairs','Fallback'],[[a,100,0,0,0] for a in ('Behavior-only','Profile-enriched')])
    table(['Profile subgroup','Users / nonempty','Behavior all / nonempty','Profile all / nonempty'],[[label,f'{s["metrics"]["behavioral"]["users"]} / {s["metrics"]["behavioral"]["nonempty_truth_users"]}',fmt(s['metrics']['behavioral']['ndcg_at_10'])+' / '+fmt(s['metrics']['behavioral']['ndcg_at_10_nonempty']),fmt(s['metrics']['profile_enriched']['ndcg_at_10'])+' / '+fmt(s['metrics']['profile_enriched']['ndcg_at_10_nonempty'])] for label,s in r['segments'].items()])
    paired=r['paired'];timing=r['timing'];backend=m['backend']
    p(f'Profile versus behavior: {paired["changed_rankings"]} changed rankings; {paired["improved"]} improved, {paired["worsened"]} worsened and {paired["unchanged"]} unchanged user scores. Changes versus original CBF: behavior {r["changed_vs_cbf"]["behavioral"]}, profile {r["changed_vs_cbf"]["profile_enriched"]}. Every arm contains 100 users and 1000 finite scores; these are 100 observed users, not 2000 users.')
    p(f'Data preparation: {timing["preparation_seconds"]:.4f} s. Behavior inference: {timing["inference_seconds"]["behavioral"]:.4f} s; profile inference: {timing["inference_seconds"]["profile_enriched"]:.4f} s. Inference timing includes the smoke scores reused in the full run and device synchronization, excluding loading, tokenization and checkpoints. The three-user smoke took {r["smoke"]["inference_seconds"]:.4f} s of inference and projected {r["smoke"]["projected_full_seconds"]:.4f} s for the full workload including one model load. There were no training, repeated seeds, repair calls or extra ablations.')
    p(f'Model: {cfg["model"]}; revision {cfg["revision"]}. Hardware: Apple M4, 16 GB RAM, MPS float16; batch size 10, max_length 512, seed 42, evaluation mode, SDPA, no generation or KV cache. Python {backend["python"]}, torch {backend["torch"]}, transformers {backend["transformers"]}. Exact dependencies and model-file hashes are in manifest.json. Each batch is appended to scores.jsonl, flushed and fsynced. Resume validates the protocol, skips saved scores and discards only an unfinished final JSONL record. A three-test checkpoint/evaluator suite passed. An import conflict was fixed before any scoring; HW2 files remain unchanged.')
    heading('Discussion')
    p(f'The primary profile effect is {delta:+.4f}. '+('H1 has directional support in this development sample, without a claim of statistical significance.' if delta>0 else 'H1 is not supported by this development sample; this does not prove a universal absence of profile value.'))
    for label,title in [('improved','Improvement'),('worsened','Deterioration'),('changed_same_ndcg','Changed ranking with unchanged nDCG')]:
        x=r['examples'][label]
        if x:
            uid=x['user_id'];changes=[]
            for asset in sorted(truths[uid]):
                b_ids=r['predictions']['behavioral'][uid];p_ids=r['predictions']['profile_enriched'][uid]
                changes.append(f'{asset}: B position {b_ids.index(asset)+1 if asset in b_ids else "outside Top-10"}, P position {p_ids.index(asset)+1 if asset in p_ids else "outside Top-10"}')
            detail='; '.join(changes) if changes else 'Future truth is empty, so changes among these candidates cannot change nDCG.'
            p(f'{title}: user {uid}, behavior {x["behavioral_ndcg"]:.4f}, profile {x["profile_ndcg"]:.4f}, difference {x["delta"]:+.4f}. {detail}')
        else:p(f'{title}: no such case exists in the saved results; no example is invented.')
    d.add_page_break()
    heading('Discussion continued')
    p(f'With only 11 nonempty users, most observations contribute zero regardless of the ranking. Only {r["independent_verification"]["nonempty_users_with_retrieved_positive"]} of those 11 have a relevant asset in CBF Top-10. A Top-10-only reranker cannot recover a missed positive and cannot change the recommended set; it changes positions only. Segment is known for 100 users, but risk tolerance and capacity are admissible for 54; 46 retain unknown values. The unknown subgroup still receives segment, so it is not a profile-free control. The model is trained for generic relevance rather than investor behavior. These factors can limit or obscure profile effects; the experiment does not identify their separate causal contributions.')
    p('HW2 failures were malformed or overlong generated lists, duplicate IDs and noncandidate IDs; 90 enriched outputs used CBF fallback. HW3 scores only existing candidates and constructs the permutation deterministically. Finite scores eliminate this output-format failure mechanism, but they do not guarantee meaningful ranking quality. One model, prompt, cutoff and cohort do not establish repeated-run reproducibility or robustness. Purchase relevance establishes neither profitability nor suitability. A future same-information LambdaMART control remains unimplemented.')
    d.add_heading('Personal verification PENDING',level=2)
    p(f'AI-prepared check for user {u}. Future truth: {", ".join(check["truth"])}. Scores are raw yes-minus-no logit differences; B and P positions below identify both complete Top-10 rankings. IDs appear only in evaluation artifacts, not model prompts.')
    rows=[]
    for c in check['candidates']:
        i=c['id'];rows.append([c['original_rank'],i,fmt(c['behavioral_score']),fmt(c['profile_score']),methods['behavioral']['top10'].index(i)+1,methods['profile_enriched']['top10'].index(i)+1,int(i in truth)])
    table(['CBF','Asset ID','B score','P score','B pos','P pos','Rel'],rows)
    for a,label in [('behavioral','Behavior'),('profile_enriched','Profile')]:
        x=methods[a]
        p(f'{label}: relevant positions {x["relevant_positions"]}; DCG = Σ rel(k)/log2(k+1) = {x["dcg"]:.10f}; IDCG = {x["idcg"]:.10f}, using min(10, |truth|) ideal positives. nDCG = DCG / IDCG = {x["ndcg"]:.10f} ({x["ndcg"]:.4f}).')
    p('Personal verification: PENDING. The arithmetic and saved IDs were checked by AI; Sofya has not yet confirmed a personal verification. Running the prepared command is distinct from personally checking the result.')
    d.add_heading('Team discussion and individual contribution',level=2)
    p('The supplied team allocation is Dmitry on behavior-only reranking, Aleksandr on asset-name masking, and Sofya on admissible investor-profile context. Shared work is the team/instructor retrieval, aliases, evaluator and HW2 protocol. Sofya specified this profile hypothesis, one-field-group ablation, fixed population and resource limits, and launched the prepared inference in her local Terminal. Codex implemented and checked the code, analyzed the saved outputs and drafted this report. No new team meeting, teammate approval or unaided student verification is claimed. Final retrieval, a matched learned control and official-test protocol remain unresolved.')
    heading('AI Usage Disclosure')
    p('Codex assisted source inspection, implementation, checkpoint tests, temporal checks, inference preparation, metrics and drafting. Only the official model instructions were consulted online; no broad literature search was conducted. The student must review the report and confirm the personal calculation. The native AI-session export will be supplied separately; no session export was fabricated. Code is local on sofya-hw3-pointwise-profile and has not been pushed to GitHub.')
    d.add_page_break();heading('References')
    for text in [
        '[1] J. Sanz-Cruzado, N. Droukas and R. McCreadie. FAR-Trans: An Investment Dataset for Financial Asset Recommendation. 2024. https://arxiv.org/abs/2407.08692. Dataset: https://researchdata.gla.ac.uk/1658/.',
        '[2] Y. Hou et al. Large Language Models are Zero-Shot Rankers for Recommender Systems. ECIR 2024. https://arxiv.org/abs/2305.08845. Reference retained from Session 1.',
        '[3] Qwen. Qwen3-Reranker-0.6B model card and Transformers scoring example. Revision '+cfg['revision']+'. https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/'+cfg['revision']+'. Accessed 27 November 2026.',
        '[4] S. Semushkina. Homework 2 local report, saved results and subsequent read-only failure audit, 27 November 2026. Team repository: https://github.com/dimayakovlev014-del/aiia-far-trans.',
        '[5] Homework 3: Test Your Project Idea. Supplied course file homework3.md; teaching protocol and report requirements.'
    ]:p(text)
    OUT.mkdir(exist_ok=True);path=OUT/'Semushkina_Homework3_Report.docx';d.save(path);print(path)

if __name__=='__main__':build()
