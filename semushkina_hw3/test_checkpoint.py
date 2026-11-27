import json,tempfile,unittest
from pathlib import Path
from experiment import read_batches,ndcg

class CheckpointTests(unittest.TestCase):
    def test_interrupted_tail_preserves_committed_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'scores.jsonl'
            batch={'protocol_hash':'frozen','scores':[{'key':'u:0:behavioral','score':0.5}]}
            committed=(json.dumps(batch)+'\n').encode()
            p.write_bytes(committed+b'{"unfinished":')
            self.assertEqual(read_batches(p,'frozen'),[batch])
            self.assertEqual(p.read_bytes(),committed)

    def test_incompatible_protocol_cannot_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'scores.jsonl';p.write_text(json.dumps({'protocol_hash':'old','scores':[]})+'\n')
            with self.assertRaises(AssertionError):read_batches(p,'new')

    def test_ndcg_uses_all_truth_and_empty_users(self):
        self.assertEqual(ndcg(['a','b'],set()),0)
        self.assertAlmostEqual(ndcg(['a','b'],{'b'}),0.6309297535714575)
        self.assertLess(ndcg(['a','b'],{'b','outside'}),0.6309297535714575)

if __name__=='__main__':unittest.main()
