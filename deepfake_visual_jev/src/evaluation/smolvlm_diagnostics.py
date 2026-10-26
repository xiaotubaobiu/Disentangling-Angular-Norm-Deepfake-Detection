"""Held-out negation/composition and token-length-matched semantic controls."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset
import yaml

from data.semantic_dataset import SemanticTemplateSampler
from data.semantic_language import LanguageBank, TextEntry, _collect_and_validate_entries


PAIRED_FIELDS = ('question_text_id', 'candidate_text_ids', 'candidate_semantic_ids',
                 'candidate_mask', 'target_index', 'question_id', 'variant_id', 'render_id')


def extended_language_bank(original, diagnostics_path):
    config = yaml.safe_load(Path(diagnostics_path).read_text())
    raw = copy.deepcopy(original.raw)
    questions = []
    for source in config['negation']['questions']:
        questions.append({'id': source['id'], 'split': 'diagnostic', 'kind': 'question',
                          'yes_means': source['yes_means'], 'text': source['text']})
    raw['tasks']['T2']['negation'] = {
        'questions': questions, 'answer_sets': [config['negation']['answer_set']]}
    for task in ('T1', 'T5'):
        spec = config['composition'][task]
        raw['tasks'][task]['composition'] = {
            'questions': [{'id': f'zzz_composition_{task}_question', 'kind': 'question',
                           'split': 'diagnostic', 'text': spec['question']}],
            'candidate_sets': [{'id': f'zzz_composition_{task}_candidates',
                'options': [{'id': f'zzz_composition_{task}_{entry["semantic"]}',
                             'kind': 'candidate', 'split': 'diagnostic', **entry}
                            for entry in spec['candidates']]}]}
    extended = LanguageBank(raw, _collect_and_validate_entries(raw))
    for entry in original.text_entries:
        if extended.text_id(entry.name) != entry.text_id or extended.text(entry.text_id) != entry.text:
            raise ValueError('diagnostic texts changed a frozen text identity')
    return extended


class NegationDataset(Dataset):
    def __init__(self, base, bank, *, seed):
        self.base = base
        self.sampler = SemanticTemplateSampler(bank, seed)
        self.questions = bank.task('T2', 'negation').questions
        if len(self.questions) != 4:
            raise ValueError('negation protocol requires two exact positive/negative pairs')

    def __len__(self):
        return len(self.base) * 4

    def __getitem__(self, index):
        base_index, question_index = divmod(index, 4)
        question = self.questions[question_index]
        result = self.sampler.render(self.base[base_index], 'T2', 'negation', 0,
                                     force_question=question.name, force_candidate_set='answer_train_short')
        result['question_pair_id'] = f'T2::negation::{question_index // 2}'
        return result


def _hash(*values):
    return int.from_bytes(hashlib.sha256('\x1f'.join(map(str, values)).encode()).digest()[:8], 'big')


class MatchedTextPool:
    """Precompute neutral texts; select by sample/render key, never target.

    The original string determines only the required full-context token count.
    Replacement words are sampled from a fixed non-forensic vocabulary. Every
    admitted replacement is checked against all opposite held-out text contexts.
    """
    def __init__(self, bank, prompts, diagnostics_path):
        config = yaml.safe_load(Path(diagnostics_path).read_text())['length_matched_controls']
        self.seed = config['seed']
        self.prompts = prompts
        self.entries = list(bank.text_entries)
        self._by_id = {e.text_id: e for e in self.entries}
        self._neutral_ids = {}
        self.pools = {}
        questions = bank.entries('heldout', 'question')
        candidates = bank.entries('heldout', 'candidate')
        self._length_cache = {}
        for kind, sources, contexts in [('questions', questions, candidates), ('candidates', candidates, questions)]:
            for source in sources:
                targets = [self._length(source.text, c.text) if kind == 'questions'
                           else self._length(c.text, source.text) for c in contexts]
                standalone = len(prompts.processor.tokenizer(source.text, add_special_tokens=False)['input_ids'])
                rng = random.Random(_hash(self.seed, kind, standalone))
                pool = []
                for _attempt in range(20000):
                    count = rng.randint(1, standalone + 3)
                    neutral = ' '.join(rng.choices(config['neutral_vocabulary'], k=count))
                    lengths = [self._length(neutral, c.text) if kind == 'questions'
                               else self._length(c.text, neutral) for c in contexts]
                    if lengths != targets:
                        continue
                    if neutral not in self._neutral_ids:
                        identity = len(self.entries)
                        entry = TextEntry(name=f'zzzz_matched_{identity:05d}', text_id=identity,
                                          text=neutral, split='matched_control',
                                          kind='question' if kind == 'questions' else 'candidate')
                        self._neutral_ids[neutral] = identity
                        self.entries.append(entry)
                        self._by_id[identity] = entry
                    pool.append(self._neutral_ids[neutral])
                    if len(pool) == 16:
                        break
                if len(pool) != 16:
                    raise ValueError(f'cannot exactly match token length for {source.name}')
                self.pools[(kind, source.text_id)] = tuple(pool)
        self.bank = SimpleNamespace(text_entries=tuple(self.entries))

    def _length(self, question, candidate):
        key = (question, candidate)
        if key not in self._length_cache:
            self._length_cache[key] = len(self.prompts.encode_pair(question, candidate))
        return self._length_cache[key]

    def text(self, identity):
        return self._by_id[identity].text

    def replace(self, meaningful, kind):
        if kind not in ('candidates', 'questions'):
            raise ValueError('unknown matched control kind')
        result = dict(meaningful)
        for field in PAIRED_FIELDS:
            value = meaningful[field]
            result[f'meaningful_{field}'] = value.clone() if isinstance(value, torch.Tensor) else value
        identity = (self.seed, meaningful['sample_id'], meaningful['render_id'], kind)
        result['control_pair_id'] = f'{_hash(*identity):016x}'
        if kind == 'questions':
            pool = self.pools[(kind, int(meaningful['question_text_id']))]
            selected = pool[_hash(*identity, 'question') % len(pool)]
            result['question_text_id'] = torch.tensor(selected, dtype=torch.long)
            result['question_id'] = self._by_id[selected].name
            result['question_text'] = self.text(selected)
        else:
            ids = meaningful['candidate_text_ids'].clone()
            texts, names, used = list(meaningful['candidate_texts']), list(meaningful['candidate_ids']), set()
            for slot in range(int(meaningful['candidate_mask'].sum())):
                pool = self.pools[(kind, int(ids[slot]))]
                offset = _hash(*identity, slot) % len(pool)
                selected = next((pool[(offset + k) % len(pool)] for k in range(len(pool))
                                 if pool[(offset + k) % len(pool)] not in used), None)
                if selected is None:
                    raise ValueError('neutral candidate pool cannot provide distinct alternatives')
                used.add(selected)
                ids[slot] = selected
                texts[slot], names[slot] = self.text(selected), self._by_id[selected].name
            result['candidate_text_ids'] = ids
            result['candidate_texts'], result['candidate_ids'] = tuple(texts), tuple(names)
        result['render_id'] = result['control_pair_id']
        result['variant_id'] = f'matched_{kind}::{meaningful["variant_id"]}'
        # This is an input contract, independent of hidden labels/semantic IDs.
        original_q = self.text(int(meaningful['question_text_id']))
        changed_q = self.text(int(result['question_text_id']))
        for slot in range(int(meaningful['candidate_mask'].sum())):
            original_c = self.text(int(meaningful['candidate_text_ids'][slot]))
            changed_c = self.text(int(result['candidate_text_ids'][slot]))
            if self._length(original_q, original_c) != self._length(changed_q, changed_c):
                raise ValueError('matched control changed full input token count')
        return result


class MatchedControlDataset(Dataset):
    def __init__(self, meaningful, pool, *, kind):
        self.meaningful = meaningful
        self.pool, self.kind = pool, kind
        self.slots = tuple(i for i, (task, _question) in enumerate(meaningful.slots)
                           if kind == 'candidates' or task in ('T2', 'T4'))
        if not self.slots:
            raise ValueError('matched control has no eligible tasks')

    def __len__(self):
        return len(self.meaningful.base_dataset) * len(self.slots)

    def __getitem__(self, index):
        base, slot = divmod(index, len(self.slots))
        return self.pool.replace(self.meaningful[base * len(self.meaningful.slots) + self.slots[slot]], self.kind)
