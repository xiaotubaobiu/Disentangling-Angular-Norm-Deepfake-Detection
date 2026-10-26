### Task 6: Add permutation, counterfactual, and meaningless-text controls

**Files:**
- Create: src/evaluation/semantic_controls.py
- Modify: experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py
- Create: tests/test_semantic_controls.py

**Interfaces:**
- Produces: enumerate_candidate_permutations(candidate_count: int)
- Produces: permutation_audit(model, image_features, question_embeddings, candidate_embeddings, candidate_mask) -> dict
- Produces: build_counterfactual_requests(task_id, split, sample)
- Produces: build_meaningless_candidate_request(meaningful_request, sample_id)
- Produces: build_meaningless_question_request(meaningful_request, sample_id)

- [ ] **Step 1: Write exhaustive control tests**

~~~python
def test_permutation_enumeration_is_complete():
    assert enumerate_candidate_permutations(2) == ((0, 1), (1, 0))
    assert len(enumerate_candidate_permutations(3)) == 6


def test_nonsense_assignment_is_position_and_label_balanced(bank):
    assignments = [
        build_meaningless_candidate_request(
            meaningful_binary_request(label=index % 2), f"sample-{index}", bank
        )
        for index in range(400)
    ]
    counts = Counter(
        (request.target_index, request.label) for request in assignments
    )
    assert max(counts.values()) - min(counts.values()) <= 20


def test_t2_counterfactual_keeps_implied_fake_probability():
    pair = build_counterfactual_requests("T2", "heldout", fake_sample())
    assert {request.question_yes_means for request in pair} == {"real", "fake"}
    assert {request.target_semantic for request in pair} == {"no", "yes"}


def test_t4_counterfactual_asks_both_families():
    pair = build_counterfactual_requests("T4", "heldout", swap_sample())
    assert {request.question_yes_means for request in pair} == {"swap", "reenact"}
    assert {request.target_semantic for request in pair} == {"yes", "no"}
~~~

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

~~~bash
pytest -q tests/test_semantic_controls.py
~~~

Expected: import fails because evaluation.semantic_controls is absent.

- [ ] **Step 3: Implement exhaustive permutation auditing**

For each audited batch, cache image features once. Evaluate every valid order,
invert the order on output, and compare canonical softmax probabilities with
the identity order.

~~~python
error = (remapped_probabilities - reference_probabilities).abs().max().item()
consistent = (
    remapped_probabilities.argmax(dim=-1)
    == reference_probabilities.argmax(dim=-1)
).all().item()
~~~

Aggregate maximum error and total consistency count across DDP ranks.

- [ ] **Step 4: Implement paired question controls**

T2 pairs one held-out real-polarity question with one held-out fake-polarity
question for the same sample and answer set. T4 pairs swap and reenactment
questions. render_id records the common pair ID so evaluation joins exactly,
never by row order.

- [ ] **Step 5: Implement counterbalanced meaningless controls**

Use stable hashing of control name and sample_id to permute the two or three
pseudowords independently of image label and output position. Retain a hidden
canonical semantic assignment only in evaluation metadata. Meaningless
questions replace the meaningful question while retaining its hidden
yes_means field, so the paired score drop is well-defined but the model cannot
recover polarity from the text.

- [ ] **Step 6: Extend the evaluator with explicit conditions**

Accepted conditions are:

~~~text
fixed_h1 seen heldout permutation counterfactual
meaningless_candidates meaningless_questions
~~~

The permutation condition runs on the first 4,096 deterministically sorted
samples per domain. Every other condition uses the full eligible domain.

- [ ] **Step 7: Run control and model invariance tests**

Run:

~~~bash
pytest -q tests/test_semantic_controls.py tests/models/test_clip_semantic_jev.py
~~~

Expected: all tests pass.

- [ ] **Step 8: Commit controls**

~~~bash
git add src/evaluation/semantic_controls.py tests/test_semantic_controls.py experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py
git commit -m "feat: add E2 semantic control suites"
~~~

---

