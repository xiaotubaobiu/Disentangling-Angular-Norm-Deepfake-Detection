"""Catch loss-mask, padding, order and frozen-gradient regressions."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / 'src'))


def test_last_attended_token_supports_left_right_and_internal_padding():
    from antifake_model import last_attended
    hidden = torch.arange(12).reshape(3, 4, 1).float()
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]])
    assert last_attended(hidden, mask).flatten().tolist() == [1., 7., 10.]
    with pytest.raises(ValueError, match='empty'):
        last_attended(hidden[:1], torch.zeros(1, 4))


def test_answer_nll_ignores_prefix_and_padding_but_applies_causal_shift():
    from antifake_model import answer_nll
    # Only labels at positions 2 and 3 score, from logits at 1 and 2.
    logits = torch.tensor([[[50., -50.], [0., 0.], [0., 1.0986122887], [-20., 20.]]])
    labels = torch.tensor([[-100, -100, 0, 1]])
    sums, counts = answer_nll(logits, labels)
    assert sums.item() == pytest.approx(0.980829253, abs=1e-6)
    assert counts.item() == 2
    # A different ignored prefix prediction cannot change the candidate score.
    logits[0, 0] *= -1
    assert answer_nll(logits, labels)[0].item() == pytest.approx(0.980829253, abs=1e-6)


@pytest.fixture
def tiny_backbone(tmp_path):
    from upstream_loader import import_upstream
    import_upstream(PHASE / 'upstream')
    from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModelTextualInversion
    from lavis.models.blip2_models.modeling_llama import LlamaForCausalLMTextualInversion
    from lavis.models.blip2_models.blip2_vicuna_instruct_textinv import Blip2VicunaInstructTextinv
    from transformers import BertTokenizer, LlamaConfig, LlamaTokenizer
    torch.manual_seed(19)
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nis\nthis\nphoto\nreal\nfake\n?\n')
    qt = BertTokenizer(str(vocab))
    qt.add_special_tokens({'bos_token': '[DEC]'})
    qc = BertConfig(vocab_size=len(qt), hidden_size=16, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=24, hidden_dropout_prob=0.,
                    attention_probs_dropout_prob=0.)
    qc.encoder_width, qc.add_cross_attention, qc.cross_attention_freq, qc.query_length = 8, True, 1, 2
    qformer = BertLMHeadModelTextualInversion(qc)
    qformer.cls = None
    qt.add_special_tokens({'additional_special_tokens': ['[*]']})
    lt = LlamaTokenizer.from_pretrained(str(PHASE / 'results/assets/vicuna-7b-v1.3'))
    lt.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '</s>', 'eos_token': '</s>'})
    lc = LlamaConfig(vocab_size=len(lt), hidden_size=32, intermediate_size=48,
                     num_hidden_layers=2, num_attention_heads=4, pad_token_id=lt.pad_token_id)
    lm = LlamaForCausalLMTextualInversion(lc)
    lt.add_special_tokens({'additional_special_tokens': ['[*]']})

    class Tiny(Blip2VicunaInstructTextinv):
        def __init__(self):
            nn.Module.__init__(self)
            self.visual_encoder = nn.Sequential(nn.Flatten(2), nn.Linear(4, 8))
            self.ln_vision = nn.LayerNorm(8)
            self.Qformer, self.tokenizer, self.llm_tokenizer = qformer, qt, lt
            self.query_tokens = nn.Parameter(torch.randn(1, 2, 16))
            self.llm_proj, self.llm_model = nn.Linear(16, 32), lm
            self.max_txt_len, self.qformer_text_input = 128, True

    return Tiny()


@pytest.mark.parametrize('mode', ['lm', 'scalar'])
def test_candidate_permutation_mask_and_frozen_prompt_gradients(tiny_backbone, mode):
    from antifake_model import AntifakeScorer
    scorer = AntifakeScorer(tiny_backbone, mode=mode, gradient_checkpointing=False)
    images = torch.randn(2, 3, 2, 2)
    questions = ['Is this photo real?', 'Is this photo fake?']
    candidates = [['yes', 'no', ''], ['real', 'fake', 'a photo']]
    mask = torch.tensor([[True, True, False], [True, True, True]])
    scorer.eval()
    scores = scorer(images, questions, candidates, mask)
    perm = [1, 0, 2]
    reordered = scorer(images, questions, [[r[i] for i in perm] for r in candidates], mask[:, perm])
    assert torch.allclose(scores[:, perm], reordered, atol=1e-5)
    assert scores.softmax(-1)[0, 2].item() == 0
    scorer.train()
    loss = scorer.training_loss(images, questions, candidates, mask, torch.tensor([0, 1]))
    loss.backward()
    params = dict(scorer.named_parameters())
    for name, param in params.items():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        else:
            assert param.grad is None, name
    for suffix in ['train_word_embeddings.weight', 'train_embed_tokens.weight']:
        grad = next(p.grad for n, p in params.items() if n.endswith(suffix))
        assert grad.abs().sum() > 0
    assert set(scorer.trainable_state_dict()) == {n for n, p in params.items() if p.requires_grad}
    if mode == 'scalar':
        assert not any('lm_head' in n for n in params)


def test_original_upstream_candidate_ranking_is_preserved(tiny_backbone):
    from antifake_model import AntifakeScorer
    scorer = AntifakeScorer(tiny_backbone, mode='lm', gradient_checkpointing=False).eval()
    images = torch.randn(1, 3, 2, 2)
    question = 'Is this photo real?'
    with torch.no_grad():
        official_ranks = tiny_backbone.predict_class(
            {'image': images, 'prompt': question[:-1] + ' [*]?'}, ['yes', 'no'])
        scores = scorer(images, [question], [['yes', 'no']], torch.ones(1, 2, dtype=torch.bool))
    assert torch.equal(scores.argsort(-1, descending=True), official_ranks)


def test_cached_visual_tokens_preserve_scores_and_prompt_gradients(tiny_backbone):
    from antifake_model import AntifakeScorer
    scorer = AntifakeScorer(tiny_backbone, mode='scalar', gradient_checkpointing=False).eval()
    image = torch.randn(1, 3, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    features = scorer.extract_features(image)
    assert not features.requires_grad
    cached = scorer.classify_features(features, ['Is this photo real?'], [['yes', 'no']], mask)
    direct = scorer(image, ['Is this photo real?'], [['yes', 'no']], mask)
    assert torch.equal(cached, direct)
    cached.softmax(-1)[0, 0].backward()
    assert tiny_backbone.Qformer.bert.embeddings.train_word_embeddings.weight.grad.abs().sum() > 0
