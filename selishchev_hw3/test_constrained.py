"""Tests of the answer grammar and of constrained greedy decoding (no model needed).

python test_constrained.py              # synthetic vocabulary, runs anywhere
python test_constrained.py --tokenizer  # also the real Qwen3 tokenizer from the local HF cache
                                         # (pinned revision; no model weights loaded)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'vendor' / 'RecSys-LLMs' / 'project' / 'far-trans-llm'))
from constrained import State, allowed_tokens, build_vocabulary, LIST_LENGTH  # noqa: E402
from ranker import parse_ranking  # noqa: E402  (unchanged course parser)

REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'


def random_candidates(rng, n=50):
    ids = set()
    while len(ids) < n:
        prefix = rng.choice(['GRS', 'GRF', 'DE000A', 'CH', 'IE00B', 'GRC'])
        ids.add((prefix + ''.join(rng.choice('0123456789ABCDEF') for _ in range(12)))[:12])
    return sorted(ids)


def simulate(trie, text_by_id, eos, candidates, rng, max_steps=600):
    """Greedy decoding with random scores, masked exactly as the logits processor does."""
    generated, cand = [], frozenset(candidates)
    for _ in range(max_steps):
        state = State().feed(''.join(text_by_id[t] for t in generated), cand)
        assert state is not None
        allowed = eos if state.phase == 'done' else allowed_tokens(state, trie, cand)
        assert allowed, ('dead end', state.phase, state.prefix)
        token = max(allowed, key=lambda t: rng.random())
        if token in eos:
            return generated, True
        generated.append(token)
    return generated, False


def check_grammar():
    cands = ['GRS000000001', 'GRS000000002', 'DE0000000003'] + [f'GRS10000000{i}' for i in range(10)]
    cs = frozenset(cands)
    ok = '[\n  ' + ',\n  '.join(f'"{c}"' for c in cands[:10]) + '\n]'
    assert State().feed(ok, cs).phase == 'done'
    assert State().feed('["GRS000000001", "GRS000000001"', cs) is None, 'duplicate must be rejected'
    assert State().feed('["GRS999', cs) is None, 'non-candidate must be rejected'
    nine = '[' + ','.join(f'"{c}"' for c in cands[:9]) + ']'
    assert State().feed(nine, cs) is None, 'short list must be rejected'
    eleven = '[' + ','.join(f'"{c}"' for c in cands[:10]) + ','
    assert State().feed(eleven, cs) is None, 'eleventh element must be rejected'
    assert State().feed('Here is', cs) is None and State().feed(' ' * 9, cs) is None
    print('grammar: ok')


def check_synthetic(trials=200):
    rng = random.Random(0)
    pieces = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",[] \n')
    pieces |= {'GR', 'GRS', 'S0', '00', '000', '",', '",\n', '"]', '\n  "', '  ', '\n', '["', ', "', 'DE', '"\n]',
               '0A', 'IE', 'B1', '\n\n', '        ', 'Here', ' is', '{', '}', '<|im_end|>'}
    vocab = list(enumerate(sorted(pieces)))
    eos_id = len(vocab)
    trie, text_by_id = build_vocabulary(vocab)
    assert all('<' not in t and '{' not in t for t in text_by_id.values())
    for _ in range(trials):
        cands = random_candidates(rng)
        seq, finished = simulate(trie, text_by_id, [eos_id], cands, rng)
        assert finished, 'did not finish'
        ranking, repaired = parse_ranking(''.join(text_by_id[t] for t in seq), set(cands))
        assert len(ranking) == LIST_LENGTH and not repaired
    print(f'synthetic vocabulary: {trials}/{trials} random decodes parse as valid ten-ISIN lists')


def check_real_tokenizer(trials=30):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', revision=REVISION, local_files_only=True)
    trie, text_by_id = build_vocabulary((i, tok.decode([i])) for i in range(len(tok)))
    eos = sorted({tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>'), tok.eos_token_id})
    print(f'real tokenizer: {len(tok)} tokens, {len(text_by_id)} usable in answers, eos ids {eos}')
    prompts = HERE / 'outputs' / 'prepare' / 'ranker_prompts.jsonl'
    if prompts.exists():
        cand_lists = [[a['ISIN'] for a in json.loads(json.loads(l)['prompt'].split('\n', 1)[1])['candidate_assets']]
                      for l in prompts.read_text().splitlines()[:trials]]
    else:
        rng0 = random.Random(1)
        cand_lists = [random_candidates(rng0) for _ in range(trials)]
    rng = random.Random(0)
    for cands in cand_lists:
        seq, finished = simulate(trie, text_by_id, eos, cands, rng)
        assert finished
        decoded = tok.decode(seq, skip_special_tokens=True)
        assert decoded == ''.join(text_by_id[t] for t in seq), 'token texts do not concatenate to the decoded text'
        ranking, repaired = parse_ranking(decoded, set(cands))
        assert len(ranking) == LIST_LENGTH and not repaired
    print(f'real tokenizer: {len(cand_lists)}/{len(cand_lists)} random constrained decodes are valid; '
          f'candidate lists from {"real prompts" if prompts.exists() else "random IDs"}')


if __name__ == '__main__':
    check_grammar()
    check_synthetic()
    if '--tokenizer' in sys.argv:
        check_real_tokenizer()
    print('ALL CONSTRAINT TESTS PASSED')
