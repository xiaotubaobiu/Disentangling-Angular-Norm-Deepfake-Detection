### Task 3: Implement the frozen-CLIP semantic decision transformer

**Files:**
- Create: src/models/clip_semantic_jev.py
- Modify: src/models/factory.py
- Create: tests/models/test_clip_semantic_jev.py

**Interfaces:**
- Consumes: LanguageBank text IDs from Task 1
- Produces: CLIPSemanticVisualJev
- Produces: CLIPSemanticVisualJev.extract_features(images) -> [B,512]
- Produces: CLIPSemanticVisualJev.encode_texts(texts) -> [N,512]
- Produces: CLIPSemanticVisualJev.classify_features(image_features, question_embeddings, candidate_embeddings, candidate_mask) -> [B,N]
- Produces: CLIPSemanticVisualJev.load_h1_visual_checkpoint(path) -> None
- Produces: EncodedTextLookup.build(model, bank, allowed_splits, device)
- Produces: EncodedTextLookup.lookup(question_ids, candidate_ids, candidate_mask)

- [ ] **Step 1: Write model contract tests with a fake CLIP**

~~~python
def test_only_decision_modules_are_trainable(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev("fake", d_model=8, nhead=2, ffn_dim=16)
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(("image_projection.", "question_projection.",
                         "candidate_projection.", "decoder.", "scorer."))
        for name in trainable
    )
    assert not model.clip.training


@pytest.mark.parametrize("candidate_count", [2, 3])
def test_two_and_three_candidate_shapes_and_padding(patched_clip, candidate_count):
    model = patched_clip.CLIPSemanticVisualJev("fake", d_model=8, nhead=2, ffn_dim=16)
    image = torch.randn(4, 4)
    question = torch.randn(4, 4)
    candidates = torch.randn(4, 3, 4)
    mask = torch.arange(3)[None, :] < candidate_count
    mask = mask.expand(4, -1)
    logits = model.classify_features(image, question, candidates, mask)
    assert logits.shape == (4, 3)
    assert torch.isneginf(logits[:, candidate_count:]).all()
    assert torch.isfinite(logits[:, :candidate_count]).all()


def test_all_t5_permutations_are_equivariant(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16, dropout=0.0
    ).eval()
    image = torch.randn(2, 4)
    question = torch.randn(2, 4)
    candidates = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    reference = model.classify_features(image, question, candidates, mask)
    for order in itertools.permutations(range(3)):
        order = list(order)
        actual = model.classify_features(
            image, question, candidates[:, order], mask[:, order]
        )
        torch.testing.assert_close(actual, reference[:, order], atol=1e-6, rtol=0)


def test_padded_candidate_has_no_gradient(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16, dropout=0.0
    )
    image = torch.randn(2, 4)
    question = torch.randn(2, 4)
    candidates = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    logits = model.classify_features(image, question, candidates, mask)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert candidates.grad[:, 2].abs().max().item() == 0.0


def test_h1_checkpoint_loads_only_visual_weights(patched_clip, tmp_path):
    model = patched_clip.CLIPSemanticVisualJev("fake", d_model=8, nhead=2, ffn_dim=16)
    expected = torch.full_like(model.clip.visual.projection.weight, 2.5)
    path = tmp_path / "h1.pth"
    torch.save({"model_state_dict": {
        "visual.projection.weight": expected,
        "scorer.0.weight": torch.randn(2, 2),
    }}, path)
    model.load_h1_visual_checkpoint(path)
    torch.testing.assert_close(model.clip.visual.projection.weight, expected)


def test_train_lookup_rejects_heldout_text_id(patched_clip, language_bank):
    model = patched_clip.CLIPSemanticVisualJev("fake", d_model=8, nhead=2, ffn_dim=16)
    lookup = patched_clip.EncodedTextLookup.build(
        model, language_bank, allowed_splits=("train", "fixed"),
        device=torch.device("cpu"),
    )
    heldout_id = language_bank.text_id("t1_q_heldout")
    with pytest.raises(KeyError, match="not encoded"):
        lookup.lookup(
            torch.tensor([heldout_id]),
            torch.tensor([[language_bank.text_id("t1_real_train"), -1, -1]]),
            torch.tensor([[True, False, False]]),
        )
~~~

- [ ] **Step 2: Run the model tests and see the import failure**

Run:

~~~bash
pytest -q tests/models/test_clip_semantic_jev.py
~~~

Expected: import fails because models.clip_semantic_jev is absent.

- [ ] **Step 3: Implement frozen encoders and H1 visual loading**

Keep the complete CLIP module to retain encode_text, freeze every CLIP
parameter, and force it into evaluation mode:

~~~python
class CLIPSemanticVisualJev(nn.Module):
    def __init__(self, clip_model_name="ViT-B/16", d_model=256, nhead=8,
                 ffn_dim=1024, num_layers=2, dropout=0.1):
        super().__init__()
        self.clip, _ = clip.load(clip_model_name, device="cpu")
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.clip.eval()
        feature_dim = int(self.clip.visual.output_dim)
        self.image_projection = Projection(feature_dim, d_model)
        self.question_projection = Projection(feature_dim, d_model)
        self.candidate_projection = Projection(feature_dim, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.scorer = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def extract_features(self, images):
        with torch.no_grad():
            return self.clip.encode_image(images).float()

    def encode_texts(self, texts):
        tokens = clip.tokenize(list(texts), truncate=True).to(next(self.parameters()).device)
        with torch.no_grad():
            return self.clip.encode_text(tokens).float()
~~~

load_h1_visual_checkpoint extracts state keys beginning with visual. and loads
them strictly into self.clip.visual. It rejects a checkpoint with missing
visual keys.

- [ ] **Step 4: Implement masked permutation-equivariant scoring**

~~~python
def classify_features(self, image_features, question_embeddings,
                      candidate_embeddings, candidate_mask):
    memory = torch.stack((
        self.image_projection(image_features.float()),
        self.question_projection(question_embeddings.float()),
    ), dim=1)
    queries = self.candidate_projection(candidate_embeddings.float())
    decoded = self.decoder(
        tgt=queries,
        memory=memory,
        tgt_key_padding_mask=~candidate_mask,
    )
    logits = self.scorer(decoded).squeeze(-1)
    return logits.masked_fill(~candidate_mask, float("-inf"))

def forward(self, images, question_embeddings, candidate_embeddings,
            candidate_mask):
    return self.classify_features(
        self.extract_features(images),
        question_embeddings,
        candidate_embeddings,
        candidate_mask,
    )
~~~

- [ ] **Step 5: Implement split-scoped encoded text lookup**

EncodedTextLookup builds embeddings only for explicitly allowed splits and
uses a dense text-ID-to-row tensor initialized to -1. lookup raises if an
unmasked ID is absent, which makes accidental held-out access during training
fail loudly.

~~~python
@dataclass
class EncodedTextLookup:
    row_for_text_id: torch.Tensor
    embeddings: torch.Tensor

    def lookup(self, question_ids, candidate_ids, candidate_mask):
        q_rows = self.row_for_text_id[question_ids]
        c_rows = self.row_for_text_id[candidate_ids.clamp_min(0)]
        if (q_rows < 0).any() or (c_rows[candidate_mask] < 0).any():
            raise KeyError("text ID is not encoded in this split-scoped lookup")
        questions = self.embeddings[q_rows]
        candidates = self.embeddings[c_rows.clamp_min(0)]
        candidates = candidates.masked_fill(~candidate_mask[..., None], 0.0)
        return questions, candidates
~~~

- [ ] **Step 6: Register the model factory name**

Add clip_semantic_jev to build_model and pass the five architecture values
from configuration. Do not alter clip_visual_jev.

- [ ] **Step 7: Run model and H1 regression tests**

Run:

~~~bash
pytest -q tests/models/test_clip_semantic_jev.py tests/models/test_clip_visual_jev.py
~~~

Expected: all tests pass.

- [ ] **Step 8: Commit the model**

~~~bash
git add src/models/clip_semantic_jev.py src/models/factory.py tests/models/test_clip_semantic_jev.py
git commit -m "feat: add E2 cross-attention decision model"
~~~

---

