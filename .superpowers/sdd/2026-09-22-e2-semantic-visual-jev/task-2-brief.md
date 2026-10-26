### Task 2: Build deterministic five-task semantic datasets

**Files:**
- Create: src/data/semantic_dataset.py
- Modify: src/data/builders.py
- Create: tests/test_semantic_dataset.py

**Interfaces:**
- Consumes: LanguageBank and SEMANTIC_TO_ID from Task 1
- Produces: method_family(method_id: str, label: int) -> str
- Produces: SemanticTemplateSampler.render(sample: dict, task_id: str | None, split: str, epoch: int, permutation: tuple[int, ...] | None = None) -> dict
- Produces: SemanticDataset(base_dataset, sampler, mode: str, tasks: tuple[str, ...])
- Produces: SemanticDataset.set_epoch(epoch: int) -> None
- Produces: build_semantic_train_loader, build_semantic_val_loader, and build_semantic_eval_loader
- Batch tensors: question_text_id [B], candidate_text_ids [B,3], candidate_semantic_ids [B,3], candidate_mask [B,3], target_index [B], task_index [B], question_yes_means [B]

- [ ] **Step 1: Write truth-table, determinism, and padding tests**

~~~python
import torch

from data.semantic_dataset import SemanticTemplateSampler, method_family
from data.semantic_language import load_language_bank


def _sample(label, method_id, sample_id="sample-1"):
    return {
        "image": torch.zeros(3, 224, 224),
        "label": torch.tensor(label),
        "method_id": method_id,
        "video_id": "001",
        "video_key": f"{method_id}::001",
        "image_path": "/tmp/frame.png",
        "sample_id": sample_id,
    }


def test_method_family_never_guesses_unknown_fake():
    assert method_family("real", 0) == "camera"
    assert method_family("simswap", 1) == "swap"
    assert method_family("sadtalker", 1) == "reenact"
    assert method_family("DFD_fake", 1) == "unknown"


def test_t2_target_flips_with_question_polarity(bank):
    sampler = SemanticTemplateSampler(bank, seed=42)
    real = _sample(0, "real")
    positive = sampler.render(
        real, "T2", "train", 0, force_question="t2_real_q1_train",
        force_candidate_set="answer_train_short", permutation=(0, 1),
    )
    negative = sampler.render(
        real, "T2", "train", 0, force_question="t2_fake_q1_train",
        force_candidate_set="answer_train_short", permutation=(0, 1),
    )
    assert positive["target_semantic"] == "yes"
    assert negative["target_semantic"] == "no"


def test_t4_and_t5_truth_tables(bank):
    sampler = SemanticTemplateSampler(bank, seed=42)
    swap = _sample(1, "simswap")
    assert sampler.render(
        swap, "T4", "train", 0, force_question="t4_swap_q_train"
    )["target_semantic"] == "yes"
    assert sampler.render(
        swap, "T4", "train", 0, force_question="t4_reenact_q_train"
    )["target_semantic"] == "no"
    t5 = sampler.render(swap, "T5", "train", 0)
    assert t5["target_semantic"] == "swap"
    assert t5["candidate_mask"].tolist() == [True, True, True]


def test_binary_candidates_are_padded_without_targeting_padding(bank):
    row = SemanticTemplateSampler(bank, seed=42).render(
        _sample(1, "fsgan"), "T1", "train", 0
    )
    assert row["candidate_mask"].tolist().count(True) == 2
    assert row["candidate_text_ids"].shape == (3,)
    assert row["target_index"] < 2


def test_hashing_is_repeatable_and_epoch_sensitive(bank):
    sampler_a = SemanticTemplateSampler(bank, seed=2048)
    sampler_b = SemanticTemplateSampler(bank, seed=2048)
    item = _sample(1, "fomm", sample_id="stable-id")
    first = sampler_a.render(item, None, "train", 3)
    repeated = sampler_b.render(item, None, "train", 3)
    changed = sampler_a.render(item, None, "train", 4)
    assert first["render_id"] == repeated["render_id"]
    torch.testing.assert_close(
        first["candidate_text_ids"], repeated["candidate_text_ids"]
    )
    assert first["render_id"] != changed["render_id"]


def test_language_render_is_identical_with_zero_or_two_workers(bank, base_dataset):
    zero_worker = SemanticDataset(
        base_dataset, SemanticTemplateSampler(bank, seed=42), mode="train"
    )
    two_workers = SemanticDataset(
        base_dataset, SemanticTemplateSampler(bank, seed=42), mode="train"
    )
    zero_worker.set_epoch(6)
    two_workers.set_epoch(6)
    ids_zero = [
        render_id
        for batch in DataLoader(zero_worker, batch_size=8, num_workers=0)
        for render_id in batch["render_id"]
    ]
    ids_two = [
        render_id
        for batch in DataLoader(two_workers, batch_size=8, num_workers=2)
        for render_id in batch["render_id"]
    ]
    assert ids_zero == ids_two


def test_unknown_fake_supports_t1_to_t3_but_rejects_t4_t5(bank):
    sampler = SemanticTemplateSampler(bank, seed=42)
    dfd = _sample(1, "DFD_fake")
    sampler.render(dfd, "T1", "heldout", 0)
    for task in ("T4", "T5"):
        with pytest.raises(ValueError, match="unknown method family"):
            sampler.render(dfd, task, "heldout", 0)
~~~

- [ ] **Step 2: Verify the tests fail**

Run:

~~~bash
pytest -q tests/test_semantic_dataset.py
~~~

Expected: import fails because data.semantic_dataset is absent.

- [ ] **Step 3: Implement stable hashing and method-family truth**

~~~python
TASKS = ("T1", "T2", "T3", "T4", "T5")
MAX_CANDIDATES = 3


def _stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def method_family(method_id: str, label: int) -> str:
    if int(label) == 0:
        return "camera"
    if method_id in FACE_SWAP_METHODS:
        return "swap"
    if method_id in REENACTMENT_METHODS:
        return "reenact"
    return "unknown"
~~~

Task selection uses _stable_u64(seed, epoch, sample_id) modulo five.
Question, candidate-set, and permutation choices use an independent
random.Random seeded by _stable_u64(seed, epoch, sample_id, task_id).

- [ ] **Step 4: Implement render logic and fixed-size tensors**

The renderer must compute canonical truth before permutation, then permute
text IDs and semantic IDs together:

~~~python
def _pack_candidates(options, target_semantic, order):
    ordered = [options[index] for index in order]
    target_index = next(
        index for index, option in enumerate(ordered)
        if option.semantic == target_semantic
    )
    text_ids = torch.full((MAX_CANDIDATES,), -1, dtype=torch.long)
    semantic_ids = torch.full((MAX_CANDIDATES,), -1, dtype=torch.long)
    mask = torch.zeros(MAX_CANDIDATES, dtype=torch.bool)
    for index, option in enumerate(ordered):
        text_ids[index] = option.text_id
        semantic_ids[index] = SEMANTIC_TO_ID[option.semantic]
        mask[index] = True
    return text_ids, semantic_ids, mask, target_index
~~~

For T2, target yes exactly when question.yes_means equals real/fake visual
truth. For T4, target yes exactly when question.yes_means equals swap/reenact
method family. For T5, target is camera/swap/reenact.

- [ ] **Step 5: Add train and deterministic suite wrappers**

~~~python
class SemanticDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, sampler, mode="train", tasks=TASKS, split="train"):
        self.base_dataset = base_dataset
        self.sampler = sampler
        self.mode = mode
        self.tasks = tuple(tasks)
        self.split = split
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.mode == "train":
            return len(self.base_dataset)
        return len(self.base_dataset) * len(self.tasks)

    def __getitem__(self, index):
        if self.mode == "train":
            base_index, task_id = index, None
        else:
            base_index = index // len(self.tasks)
            task_id = self.tasks[index % len(self.tasks)]
        sample = self.base_dataset[base_index]
        return self.sampler.render(sample, task_id, self.split, self.epoch)
~~~

Suite builders must omit T4/T5 for DFD and preserve the existing H1
DistributedSampler, worker seeding, pinning, and shuffle behavior.

- [ ] **Step 6: Refactor builders without changing old loader outputs**

Extract build_train_dataset, build_val_dataset, and build_eval_dataset from
the existing builders. Existing build_train_loader, build_val_loader, and
build_eval_loader call those functions, so H1 behavior remains byte-for-byte
equivalent at the sample-dictionary boundary. Add semantic loader functions
that wrap the base dataset and use default collation on fixed-shape tensors.

- [ ] **Step 7: Run semantic and legacy data tests**

Run:

~~~bash
pytest -q tests/test_semantic_dataset.py tests/test_sample_eval.py tests/test_video_aggregation.py
~~~

Expected: all tests pass.

- [ ] **Step 8: Commit the semantic data layer**

~~~bash
git add src/data/semantic_dataset.py src/data/builders.py tests/test_semantic_dataset.py
git commit -m "feat: add deterministic E2 semantic sampler"
~~~

---

