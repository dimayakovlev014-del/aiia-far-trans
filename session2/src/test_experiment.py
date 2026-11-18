"""Meaningful synthetic contract checks; fake outputs here are never called model runs."""
import copy
import json
import math
import unittest

from experiment import behavioral_prompt,parse_and_fallback,teacher_ranker,data_adapter


class Contracts(unittest.TestCase):
    def setUp(self):
        self.ids=[f'ISIN_{i:03d}' for i in range(55)]
        self.context=dict(cutoff='2021-06-01',user_id='DO_NOT_INCLUDE_CUSTOMER',
                          profile={'secret':'DO_NOT_INCLUDE_PROFILE'},
                          candidate_assets=[{'id':i,'name':'DO_NOT_INCLUDE_NAME'} for i in self.ids[:50]],
                          eligible_asset_ids=self.ids)
        self.history={self.ids[0]:2,self.ids[54]:1}
        self.events=[('2021-05-01',self.ids[0],'Buy'),('2021-05-31',self.ids[0],'Buy'),
                     ('2021-05-20',self.ids[54],'Buy')]
        self.scores={i:0.5 for i in self.ids}
        self.pop={i:9 for i in self.ids}

    def build(self,events=None,context=None):
        return behavioral_prompt(context or self.context,self.history,
                                 self.events if events is None else events,self.scores,self.pop)

    def test_full_pool_alias_roundtrip_and_no_semantics(self):
        text,reverse,payload=self.build()
        self.assertEqual(len(payload['candidates']),50)
        self.assertEqual([reverse[r[0]] for r in payload['candidates']],self.ids[:50])
        self.assertEqual(len(reverse),51)
        for secret in self.ids+['DO_NOT_INCLUDE_CUSTOMER','DO_NOT_INCLUDE_PROFILE','DO_NOT_INCLUDE_NAME']:
            self.assertNotIn(secret,text)
        self.assertEqual(payload['candidates'][0][3:],[2,1,9])
        self.assertEqual([r[1] for r in payload['recent_15_buys_oldest_first']],[31,12,1])
        self.assertEqual(text,self.build()[0])

    def test_strict_future_boundary_and_sell_exclusion(self):
        altered=self.events+[('2021-06-01',self.ids[0],'Buy'),('2021-06-10',self.ids[1],'Buy'),
                             ('2021-05-31',self.ids[2],'Sell')]
        self.assertEqual(self.build()[0],self.build(altered)[0])

    def test_alias_not_candidate_rank_and_history_consistency(self):
        _,reverse,payload=self.build()
        lookup={v:k for k,v in reverse.items()}
        self.assertNotEqual([r[0] for r in payload['candidates']],sorted(lookup.values())[:50])
        for alias,n,age in payload['history']:
            self.assertEqual(n,self.history[reverse[alias]])
        c=copy.deepcopy(self.context)
        c['candidate_assets'].reverse()
        self.assertEqual(reverse,self.build(context=c)[1])

    def test_count_mismatch_and_short_pool_rejected(self):
        with self.assertRaises(ValueError):
            self.build(events=self.events[:-1])
        c=copy.deepcopy(self.context)
        c['candidate_assets']=c['candidate_assets'][:49]
        with self.assertRaises(ValueError):
            self.build(context=c)

    def test_native_repaired_and_mapping(self):
        _,reverse,payload=self.build()
        aliases=[r[0] for r in payload['candidates'][:10]]
        r=parse_and_fallback(json.dumps(aliases),self.ids[:50],reverse)
        self.assertTrue(r['valid'])
        self.assertEqual(r['ranking'],self.ids[:10])
        r=parse_and_fallback('```json\n'+json.dumps(aliases)+'\n```',self.ids[:50],reverse)
        self.assertTrue(r['parse_repaired'])

    def test_malformed_unknown_duplicate_and_history_only_fallback(self):
        _,reverse,payload=self.build()
        aliases=[r[0] for r in payload['candidates'][:10]]
        other=next(a for a,i in reverse.items() if i==self.ids[54])
        for raw in ('bad','[',json.dumps(aliases[:9]),json.dumps([aliases[0]]*10),
                    json.dumps(aliases[:9]+['unknown']),json.dumps(aliases[:9]+[other])):
            r=parse_and_fallback(raw,self.ids[:50],reverse)
            self.assertTrue(r['fallback'])
            self.assertFalse(r['valid'])
            self.assertEqual(r['ranking'],self.ids[:10])

    def test_timeout_and_model_not_run_never_valid(self):
        for status in ('timeout','model_load_error','generation_exception','worker_exit','context_overflow'):
            r=parse_and_fallback(json.dumps(self.ids[:10]),self.ids[:50],status=status)
            self.assertEqual(r['status'],status)
            self.assertFalse(r['valid'])
            self.assertTrue(r['fallback'])

    def test_binary_macro_metric_full_truth_denominator(self):
        import importlib.util
        from experiment import TEACHER
        spec=importlib.util.spec_from_file_location('test_st',TEACHER/'far-trans/starter.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ranking=['a','x','b']+[f'irrelevant_{i}' for i in range(7)]
        expected=(1+1/math.log2(4))/(1+1/math.log2(3)+1/math.log2(4))
        value=module.ndcg10(ranking,{'a','b','outside_pool'})
        self.assertAlmostEqual(value,expected)
        self.assertEqual(module.ndcg10(ranking,set()),0)
        agg=module.aggregate([dict(ndcg_at_10=value,relevant_count=3),dict(ndcg_at_10=0,relevant_count=0)])
        self.assertAlmostEqual(agg['ndcg_at_10'],value/2)
        self.assertEqual(agg['nonempty_truth_users'],1)


if __name__=='__main__':
    unittest.main()
