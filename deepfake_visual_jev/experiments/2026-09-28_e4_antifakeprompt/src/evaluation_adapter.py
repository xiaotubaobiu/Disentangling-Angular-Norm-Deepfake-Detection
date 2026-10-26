"""Use audited E2 metrics/controls with AntifakePrompt text and visual tokens."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import hashlib
import json
import torch
from torch import nn
from torch.utils.data import Dataset

from antifake_model import soft_prompt_question
from experiment_data import ExhaustiveSuite, ROOT, PHASE, sha256, atomic_json
from data.semantic_language import SEMANTIC_TO_ID, TextEntry
from evaluation.smolvlm_diagnostics import MatchedTextPool, MatchedControlDataset, NegationDataset


class TextLookup:
    def __init__(self, bank):
        self.entries = {e.text_id: e.text for e in bank.text_entries}
        self.entries.update({900000: 'Is this photo real [*]?', 900001: 'yes', 900002: 'no'})

    def text(self, index):
        return self.entries[int(index)]

    def lookup(self, questions, candidates, mask):
        # Scalar lookup ids support the existing tensor permutation auditor.
        # They are decoded into strings before reaching the actual scorer.
        return questions[:, None], candidates[:, :, None]


class MetricAdapter(nn.Module):
    def __init__(self, scorer, lookup):
        super().__init__()
        self.scorer, self.lookup = scorer, lookup

    def _decode(self, questions, candidates):
        qs = [self.lookup.text(i) for i in questions.flatten().tolist()]
        cs = [[self.lookup.text(i) if i >= 0 else '' for i in row] for row in candidates.squeeze(-1).tolist()]
        return qs, cs

    def forward(self, images, questions, candidates, mask):
        q, c = self._decode(questions, candidates)
        return self.scorer(images, q, c, mask)

    def extract_features(self, images):
        return self.scorer.extract_features(images)

    def classify_features(self, features, questions, candidates, mask):
        q, c = self._decode(questions, candidates)
        return self.scorer.classify_features(features, q, c, mask)


class OfficialBinaryDataset(Dataset):
    def __init__(self, base, bank):
        self.base = base
        self.template = ExhaustiveSuite(base, bank, split='train', tasks=('T2',))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.template[index * len(self.template.slots)]
        label = int(row['label'])
        row.update(question_text_id=torch.tensor(900000), question_id='official_antifake',
            question_yes_means=torch.tensor(SEMANTIC_TO_ID['real']), question_yes_means_name='real',
            question_pair_id='', candidate_text_ids=torch.tensor([900001, 900002, -1]),
            candidate_semantic_ids=torch.tensor([SEMANTIC_TO_ID['yes'], SEMANTIC_TO_ID['no'], -1]),
            candidate_mask=torch.tensor([True, True, False]), target_index=torch.tensor(label),
            target_semantic='no' if label else 'yes', variant_id='official_antifake::yes_no',
            render_id='official_antifake::' + row['sample_id'])
        return row


class DualTokenizerLengths:
    def __init__(self, llm_tokenizer, qformer_tokenizer):
        self.processor = SimpleNamespace(tokenizer=llm_tokenizer)
        self.qformer = qformer_tokenizer

    def encode_pair(self, question, candidate):
        question = soft_prompt_question(question)
        tok = self.processor.tokenizer
        lm_len = len(tok(question)['input_ids']) + len(tok(candidate)['input_ids']) - 1
        q_len = len(self.qformer(question)['input_ids'])
        if q_len >= 1024:
            raise ValueError('unexpected long Q-former input')
        # Existing pool compares len(encode_pair). Packing both integer lengths
        # enforces exact matches at BOTH text encoders, without allocating tokens.
        return range(lm_len * 1024 + q_len)


def matched_pool(bank, scorer):
    prompts = DualTokenizerLengths(scorer.backbone.llm_tokenizer, scorer.backbone.tokenizer)
    config = PHASE / 'configs/matched_controls.yaml'
    path = PHASE / 'results/protocol/matched_pool.json'
    identity = {'config_sha256': sha256(config),
        'assets_sha256': sha256(PHASE / 'results/assets/manifest.json'),
        'bank_sha256': hashlib.sha256(json.dumps([asdict(e) for e in bank.text_entries], sort_keys=True).encode()).hexdigest(),
        'match': 'exact_llm_and_qformer_lengths'}
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get('identity') != identity:
            raise ValueError('saved matched text pool provenance differs')
        pool = MatchedTextPool.__new__(MatchedTextPool)
        pool.seed, pool.prompts = payload['seed'], prompts
        pool.entries = [TextEntry(**row) for row in payload['entries']]
        pool._by_id = {e.text_id: e for e in pool.entries}
        pool._length_cache = {}
        pool.pools = {(row['kind'], row['source']): tuple(row['choices']) for row in payload['pools']}
        pool.bank = SimpleNamespace(text_entries=tuple(pool.entries))
        return pool
    pool = MatchedTextPool(bank, prompts, config)
    atomic_json(path, {'identity': identity, 'seed': pool.seed,
        'entries': [asdict(e) for e in pool.entries],
        'pools': [{'kind': k, 'source': s, 'choices': list(c)} for (k, s), c in pool.pools.items()]})
    return pool


def condition_dataset(base, bank, condition, legacy, pool=None):
    if condition == 'official_binary':
        return OfficialBinaryDataset(base, bank)
    if condition == 'fixed_h1':
        return legacy.FixedH1Dataset(base, bank)
    if condition == 'negation':
        return NegationDataset(base, bank, seed=42)
    if condition == 'composition':
        return ExhaustiveSuite(base, bank, split='composition', tasks=('T1', 'T5'))
    if condition == 'counterfactual':
        return legacy.CounterfactualEvaluationDataset(base, bank, tasks=('T2', 'T4'), split='heldout', seed=42)
    suite = ExhaustiveSuite(base, bank, split='train' if condition == 'seen' else 'heldout')
    if condition.startswith('meaningless_matched_'):
        return MatchedControlDataset(suite, pool, kind=condition.removeprefix('meaningless_matched_'))
    if condition.startswith('meaningless_'):
        return legacy.PairedMeaninglessEvaluationDataset(suite, bank, control=condition)
    if condition in {'seen', 'heldout', 'permutation'}:
        return suite
    raise ValueError(condition)
