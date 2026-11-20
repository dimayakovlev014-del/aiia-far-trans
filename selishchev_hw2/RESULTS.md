# HW2 results (aggregate only)

Protocol hash `50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417`, cutoff 2021-06-01, window to 2021-07-01 (exclusive).

- llmrank: Qwen/Qwen3-1.7B @ 70d244cc86cc, device cpu, dtype torch.float32, torch 2.11.0, Apple M4, RAM 16 GB
- llmrank_masked: Qwen/Qwen3-1.7B @ 70d244cc86cc, device cpu, dtype torch.float32, torch 2.11.0, Apple M4, RAM 16 GB

| Method | Users | Users with future purchases | nDCG@10 all | nDCG@10 nonempty | Runtime, s | Failed cases |
|---|---:|---:|---:|---:|---:|---:|
| popularity | 100 | 11 | 0.0286 | 0.2599 | 2.6 (whole run, 4 baselines) | 0 |
| cbf | 100 | 11 | 0.0201 | 0.1827 | 2.6 (whole run, 4 baselines) | 0 |
| latest_buy | 100 | 11 | 0.0277 | 0.2520 | 2.6 (whole run, 4 baselines) | 0 |
| latest_buy_or_sell | 100 | 11 | 0.0271 | 0.2464 | 2.6 (whole run, 4 baselines) | 0 |
| candidate_order_first10 | 100 | 11 | 0.0201 | 0.1827 | n/a (no model; listed by the LLM runs) | 0 |
| llmrank | 100 | 11 | 0.0201 | 0.1827 | 5231.5 | 23 |
| llmrank_masked | 100 | 11 | 0.0201 | 0.1827 | 4631.4 | 31 |

```json
{
  "llmrank": {
    "valid_users": 77,
    "valid_nonempty_users": 2,
    "valid_only_ndcg_all": 0.007962950555395563,
    "valid_only_candidate_order_ndcg_all": 0.007962950555395563,
    "valid_only_ndcg_nonempty": 0.3065735963827292,
    "valid_only_candidate_order_ndcg_nonempty": 0.3065735963827292,
    "users_ranking_differs_from_candidate_order": 76,
    "mean_overlap_with_candidate_first10": 5.54,
    "error_types": [
      "ValueError"
    ]
  },
  "llmrank_masked": {
    "valid_users": 69,
    "valid_nonempty_users": 1,
    "valid_only_ndcg_all": 0.008886191199499396,
    "valid_only_candidate_order_ndcg_all": 0.008886191199499396,
    "valid_only_ndcg_nonempty": 0.6131471927654584,
    "valid_only_candidate_order_ndcg_nonempty": 0.6131471927654584,
    "users_ranking_differs_from_candidate_order": 67,
    "mean_overlap_with_candidate_first10": 6.9,
    "error_types": [
      "ValueError"
    ]
  },
  "masked_vs_unmasked": {
    "identical_top10_lists": 23,
    "mean_top10_overlap": 6.86,
    "nonempty_users_better_masked": 0,
    "nonempty_users_worse_masked": 0
  },
  "candidate_ceiling": {
    "nonempty_users_with_a_future_buy_among_50_candidates": 6,
    "best_possible_rerank_ndcg_all": 0.04915749606264272
  }
}
```

Checks failed: 0
