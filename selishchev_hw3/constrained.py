"""Grammar-constrained greedy decoding for the course LLM reranker (HW3 main change).

The model still sees the unchanged course prompt and still decodes greedily; the only
difference is that at every generation step the tokens that cannot lead to the
required answer are masked out. The required answer is the format the course
parser accepts: a JSON array of exactly ten unique ISINs taken from the user's 50
candidates (whitespace between elements allowed). After the closing bracket only
end-of-sequence tokens are allowed. The processor also counts *interventions*:
steps at which the model's unconstrained greedy choice was not allowed.
"""
from __future__ import annotations

import json
import time

WS = (' ', '\n')
MAX_WS_RUN = 8      # whitespace characters allowed in a row between elements
LIST_LENGTH = 10


class State:
    """Immutable character-level automaton state for the answer grammar."""
    __slots__ = ('phase', 'used', 'prefix', 'count', 'ws')

    def __init__(self, phase='pre', used=frozenset(), prefix='', count=0, ws=0):
        self.phase, self.used, self.prefix, self.count, self.ws = phase, used, prefix, count, ws

    def step(self, ch, candidates):
        """Return the next state, or None if ``ch`` is not allowed here."""
        p = self.phase
        if p in ('pre', 'open', 'sep', 'after') and ch in WS:
            return State(p, self.used, '', self.count, self.ws + 1) if self.ws < MAX_WS_RUN else None
        if p == 'pre':
            return State('open', self.used, '', self.count) if ch == '[' else None
        if p in ('open', 'sep'):
            return State('id', self.used, '', self.count) if ch == '"' else None
        if p == 'id':
            if ch == '"':
                if self.prefix in candidates and self.prefix not in self.used:
                    return State('after', self.used | {self.prefix}, '', self.count + 1)
                return None
            new = self.prefix + ch
            if any(c.startswith(new) and c not in self.used for c in candidates):
                return State('id', self.used, new, self.count)
            return None
        if p == 'after':
            if ch == ',' and self.count < LIST_LENGTH:
                return State('sep', self.used, '', self.count)
            if ch == ']' and self.count == LIST_LENGTH:
                return State('done', self.used, '', self.count)
            return None
        return None  # 'done': nothing but end-of-sequence

    def feed(self, text, candidates):
        state = self
        for ch in text:
            state = state.step(ch, candidates)
            if state is None:
                return None
        return state


class Trie:
    """Character trie over the token strings that can occur in an answer."""
    __slots__ = ('children', 'ids')

    def __init__(self):
        self.children, self.ids = {}, []

    def add(self, text, token_id):
        node = self
        for ch in text:
            node = node.children.setdefault(ch, Trie())
        node.ids.append(token_id)


ALPHABET = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",[] \n')


def build_vocabulary(token_strings):
    """token_strings: iterable of (token_id, decoded_text). Returns (trie, text_by_id)."""
    trie, text_by_id = Trie(), {}
    for token_id, text in token_strings:
        if text and set(text) <= ALPHABET:
            trie.add(text, token_id)
            text_by_id[token_id] = text
    return trie, text_by_id


def allowed_tokens(state, trie, candidates):
    """All token ids whose full text is a valid continuation from ``state``."""
    found, stack = [], [(state, trie)]
    while stack:
        current, node = stack.pop()
        for ch, child in node.children.items():
            nxt = current.step(ch, candidates)
            if nxt is not None:
                found.extend(child.ids)
                if child.children:
                    stack.append((nxt, child))
    return found


def candidate_ids_from_prompt(prompt):
    payload = json.loads(prompt.split('\n', 1)[1])
    return [row['ISIN'] for row in payload['candidate_assets']]


def make_processor_class():
    import torch
    from transformers import LogitsProcessor

    class CandidateListProcessor(LogitsProcessor):
        def __init__(self, candidates, trie, text_by_id, prompt_length, eos_ids):
            self.candidates = frozenset(candidates)
            self.trie, self.text_by_id = trie, text_by_id
            self.prompt_length, self.eos_ids = prompt_length, list(eos_ids)
            self.steps = self.interventions = 0
            self.intervention_steps = []

        def __call__(self, input_ids, scores):
            generated = input_ids[0, self.prompt_length:].tolist()
            text = ''.join(self.text_by_id[t] for t in generated)
            state = State().feed(text, self.candidates)
            if state is None:  # cannot happen when every step was masked
                raise RuntimeError('constrained decoding left the grammar')
            allowed = self.eos_ids if state.phase == 'done' else allowed_tokens(state, self.trie, self.candidates)
            if not allowed:
                raise RuntimeError('no allowed token in constrained decoding')
            top = int(scores[0].argmax())
            if top not in set(allowed):
                self.interventions += 1
                self.intervention_steps.append(self.steps)
            self.steps += 1
            mask = torch.full_like(scores, float('-inf'))
            mask[:, allowed] = 0.0
            return scores + mask

    return CandidateListProcessor


def make_generator_class():
    """Subclass of the unchanged course LocalGenerator (imported from the vendored runtime)."""
    from runtime import LocalGenerator

    class ConstrainedGenerator(LocalGenerator):
        """Course LocalGenerator with (a) an explicit dtype and (b) optional constrained decoding.

        __init__ repeats the course loading code so that the dtype can be chosen
        (the course code uses bfloat16 on CUDA and float32 elsewhere); with
        constrained=False, generate() is the course generation call unchanged.
        """

        def __init__(self, model, revision=None, device='auto', allow_download=False,
                     dtype='float32', constrained=True):
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            started = time.perf_counter()
            torch.set_num_threads(4)
            self.torch, self.model_id, self.revision = torch, model, revision
            self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
            kwargs = {'local_files_only': not allow_download, 'trust_remote_code': False}
            if revision:
                kwargs['revision'] = revision
            self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
            torch_dtype = getattr(torch, dtype)
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype, **kwargs)
            self.model.to(self.device).eval()
            self.constrained = constrained
            self.metadata = {'model': model, 'requested_revision': revision,
                             'resolved_revision': getattr(self.model.config, '_commit_hash', None),
                             'device': self.device, 'dtype': str(torch_dtype),
                             'torch_version': torch.__version__,
                             'transformers_version': transformers.__version__,
                             'tokenizer_class': type(self.tokenizer).__name__,
                             'do_sample': False, 'thinking': False, 'context_limit': 8192}
            self.load_seconds = time.perf_counter() - started
            if not constrained:
                self.vocab_seconds = 0.0
                self.metadata['decoding'] = 'greedy_free'
                return
            started = time.perf_counter()
            tok = self.tokenizer
            self.trie, self.text_by_id = build_vocabulary(
                (i, tok.decode([i])) for i in range(len(tok)))
            self.eos_ids = sorted({i for i in (tok.convert_tokens_to_ids('<|im_end|>'),
                                               tok.convert_tokens_to_ids('<|endoftext|>'),
                                               tok.eos_token_id) if isinstance(i, int) and i >= 0})
            self.vocab_seconds = time.perf_counter() - started
            self.processor_class = make_processor_class()
            self.metadata = dict(self.metadata, decoding='greedy_grammar_constrained',
                                 constraint='JSON array of exactly 10 unique candidate ISINs, then EOS',
                                 constraint_max_whitespace_run=MAX_WS_RUN,
                                 constraint_vocabulary_tokens=len(self.text_by_id),
                                 constraint_eos_ids=self.eos_ids)

        def _sync(self):
            if self.device == 'cuda':
                self.torch.cuda.synchronize()
            elif self.device == 'mps':
                self.torch.mps.synchronize()

        def generate(self, prompt, max_new_tokens=160):
            from transformers import LogitsProcessorList
            messages = [{'role': 'user', 'content': prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True,
                                                      enable_thinking=False)
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            n = inputs['input_ids'].shape[1]
            if n + max_new_tokens > 8192:
                raise ValueError('Prompt exceeds the fixed context budget; reduce candidate/history count')
            processor = (self.processor_class(candidate_ids_from_prompt(prompt), self.trie,
                                              self.text_by_id, n, self.eos_ids) if self.constrained else None)
            extra = {'logits_processor': LogitsProcessorList([processor])} if processor else {}
            self._sync()
            started = time.perf_counter()
            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                              do_sample=False,
                                              pad_token_id=self.tokenizer.eos_token_id, **extra)
            self._sync()
            tokens = outputs[0, n:]
            result = {'text': self.tokenizer.decode(tokens, skip_special_tokens=True),
                      'input_tokens': n, 'output_tokens': len(tokens),
                      'seconds': time.perf_counter() - started}
            if processor:
                result.update(constraint_steps=processor.steps,
                              constraint_interventions=processor.interventions,
                              constraint_intervention_steps=processor.intervention_steps)
            return result

    return ConstrainedGenerator
