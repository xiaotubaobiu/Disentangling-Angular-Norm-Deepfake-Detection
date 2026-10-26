### Task 5: Implement canonical semantic evaluation and video metrics

**Files:**
- Create: src/evaluation/semantic_eval.py
- Create: experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py
- Create: tests/test_semantic_eval.py

**Interfaces:**
- Consumes: semantic eval loaders, CLIPSemanticVisualJev, held-out EncodedTextLookup
- Produces: map_candidate_probabilities(probabilities, semantic_ids, mask) -> [B,7]
- Produces: canonical_fake_probability(row: dict) -> float
- Produces: validate_domain_task(domain: str, task_id: str) -> None
- Produces: frame_semantic_rows(batch, probabilities, condition) -> list[dict]
- Produces: aggregate_semantic_videos(rows) -> pandas.DataFrame
- Produces: compute_task_metrics(video_rows) -> pandas.DataFrame
- Output: results/evaluation/seed_{42,7,2048}/{condition}/{domain}_frames.npz
- Output: results/evaluation/seed_{42,7,2048}/{condition}/{domain}_videos.csv
- Output: results/evaluation/seed_{42,7,2048}/{condition}/{domain}_metrics.json

- [ ] **Step 1: Write canonical-mapping and aggregation tests**

~~~python
def test_t2_fake_probability_depends_on_question_polarity():
    authentic = {"task_id": "T2", "question_yes_means": "real",
                 "p_yes": 0.2, "p_no": 0.8}
    manipulated = {"task_id": "T2", "question_yes_means": "fake",
                   "p_yes": 0.8, "p_no": 0.2}
    assert canonical_fake_probability(authentic) == pytest.approx(0.8)
    assert canonical_fake_probability(manipulated) == pytest.approx(0.8)


def test_t5_fake_probability_is_one_minus_camera():
    row = {"task_id": "T5", "p_camera": 0.15,
           "p_swap": 0.60, "p_reenact": 0.25}
    assert canonical_fake_probability(row) == pytest.approx(0.85)


def test_video_aggregation_keeps_same_id_from_different_methods_separate():
    rows = [
        frame_row("simswap", "001", 1, 0.9),
        frame_row("blendface", "001", 1, 0.2),
        frame_row("real", "001", 0, 0.1),
    ]
    videos = aggregate_semantic_videos(rows)
    assert set(videos.video_key) == {
        "simswap::001", "blendface::001", "real::001",
    }


def test_dfd_metrics_exclude_t4_and_t5():
    with pytest.raises(ValueError, match="DFD supports only T1-T3"):
        validate_domain_task("dfd", "T4")
~~~

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

~~~bash
pytest -q tests/test_semantic_eval.py
~~~

Expected: import fails because evaluation.semantic_eval is absent.

- [ ] **Step 3: Implement canonical mapping**

Scatter each valid candidate probability into a seven-column semantic tensor.
Rows without a semantic receive zero; duplicate semantics in one candidate set
are rejected.

~~~python
def map_candidate_probabilities(probabilities, semantic_ids, mask):
    mapped = probabilities.new_zeros((probabilities.shape[0], 7))
    for column in range(probabilities.shape[1]):
        valid = mask[:, column]
        mapped[valid, semantic_ids[valid, column]] = probabilities[valid, column]
    return mapped
~~~

T1/T3 fake probability is p_fake. T2 uses p_no when yes_means is real and
p_yes when yes_means is fake. T5 uses 1 - p_camera. T4 stores p_yes for each
question and combines swap/reenact pairs after matching condition, seed,
sample_id, and video key.

- [ ] **Step 4: Implement semantic video aggregation and metrics**

Group frame rows by:

~~~python
GROUP_KEYS = [
    "seed", "condition", "task_id", "variant_id", "question_yes_means",
    "method_id", "video_id", "video_key", "label", "method_family",
]
~~~

Average canonical probabilities over frames. Reject duplicate sample IDs
inside one semantic condition. Compute:

- T1/T3 and Fixed-H1 video AUROC, AP, Brier, accuracy, and ECE;
- T2/T4 video balanced accuracy and F1;
- T5 video macro balanced accuracy and macro F1;
- task counts and per-method slices.

- [ ] **Step 5: Implement distributed inference without selection leakage**

evaluate_semantic.py accepts config, checkpoint, seed, domains, conditions,
batch size, and output directory. It loads the selected checkpoint, encodes
only the text splits requested by the condition, gathers rows across ranks,
deduplicates by semantic condition plus sample_id, and writes compressed frame
arrays plus human-readable video and metric files.

The script refuses heldout or control evaluation when checkpoint metadata lacks
the expected train-manifest SHA256.

- [ ] **Step 6: Run evaluation and H1 aggregation regression tests**

Run:

~~~bash
pytest -q tests/test_semantic_eval.py tests/test_sample_eval.py tests/test_video_aggregation.py
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit semantic evaluation**

~~~bash
git add src/evaluation/semantic_eval.py tests/test_semantic_eval.py experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py
git commit -m "feat: add canonical E2 semantic evaluation"
~~~

---

