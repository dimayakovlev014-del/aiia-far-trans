"""Tests for the intervention and the scientific comparison contract."""
import math
import unittest
from common import ROOT, S3, jsonl, parse_and_fallback, read
from prepare import rename_payload
from analyze import independent


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = jsonl(S3/'outputs/frozen/main_requests.jsonl')
        cls.ablated = jsonl(S3/'outputs/frozen/ablation_requests.jsonl')

    def test_exact_main_bytes(self):
        self.assertEqual((S3/'outputs/frozen/main_requests.jsonl').read_bytes(),
                         (ROOT/'session2/outputs/frozen/behavioral_requests.jsonl').read_bytes())

    def test_all_100_bijections_and_nonalias_features(self):
        for a,b in zip(self.main,self.ablated):
            aliases=sorted(a['alias_to_id'])
            mapping=dict(zip(aliases,reversed(aliases)))
            self.assertEqual(rename_payload(b['payload'],mapping),a['payload'])
            self.assertEqual([b['alias_to_id'][r[0]] for r in b['payload']['candidates']],a['candidates'])
            self.assertEqual(a['candidates'],b['candidates'])
            self.assertEqual(a['prompt'].split('\n',1)[0],b['prompt'].split('\n',1)[0])

    def test_alias_decoding_preserves_intended_assets(self):
        import json
        for a,b in zip(self.main,self.ablated):
            for req in (a,b):
                forward={i:alias for alias,i in req['alias_to_id'].items()}
                text=json.dumps([forward[i] for i in req['candidates'][:10]])
                result=parse_and_fallback(text,req['candidates'],req['alias_to_id'])
                self.assertTrue(result['valid'])
                self.assertEqual(result['ranking'],a['candidates'][:10])

    def test_invalid_output_keeps_entire_original_fallback(self):
        for req in (self.main[0],self.ablated[0]):
            result=parse_and_fallback('["A001","A001"]',req['candidates'],req['alias_to_id'])
            self.assertFalse(result['valid'])
            self.assertTrue(result['fallback'])
            self.assertEqual(result['ranking'],req['candidates'][:10])

    def test_metric_all_truth_defines_denominator(self):
        result=independent(['x','z','y'],{'x','y','outside'})
        self.assertAlmostEqual(result['dcg'],1+1/math.log2(4))
        self.assertAlmostEqual(result['idcg'],1+1/math.log2(3)+1/math.log2(4))
        self.assertEqual(independent(['x'],set())['ndcg_at_10'],0)


if __name__ == '__main__':
    unittest.main()
