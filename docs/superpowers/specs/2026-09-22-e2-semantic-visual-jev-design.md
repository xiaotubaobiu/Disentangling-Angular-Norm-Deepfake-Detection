# E2 Semantic Visual-Jev Experiment Design

Date: 2026-09-22

Status: approved in conversation on 2026-09-22

## 1. Objective

E2 tests whether a non-generative Visual-Jev detector can use both a question
and the meanings of a variable candidate set, rather than behaving as a fixed
binary classifier whose two output slots happen to have text labels.

The experiment starts from the completed H1 study:

- H1-Base: CLIP ViT-B/16 image encoder plus a binary classification head.
- H1-Jev / E2-Base: the completed Minimal Visual-Jev with one fixed question
  and two fixed real/fake candidates.
- E2-Semantic: the model specified here.
- E2-Tiny and E2-VLM are explicitly out of scope for this experiment.

The final result must be one of:

- supported: all preregistered gates G1 through G6 pass;
- not supported: at least one gate clearly fails;
- insufficient evidence: point estimates pass but uncertainty cannot establish
  a required non-inferiority or degradation claim.

Thresholds must not be relaxed after target-domain or held-out-language results
are observed.

## 2. Controlled visual protocol

E2 must use the exact H1 visual protocol:

- dataset root: /Dataset/deepfake_detection/DF40_all;
- the same real/fake labels;
- the same training, validation, and test splits;
- the same sixteen manipulation methods;
- the same frame sampling, face crops, image size, augmentations, normalization,
  and video aggregation;
- the same video key: method_id::video_id;
- the same H1 CLIP ViT-B/16 visual weights.

The sixteen H1 methods have two supervised families:

- identity swap: simswap, inswap, blendface, faceswap, fsgan, mobileswap, e4s,
  and facedancer;
- facial reenactment: fomm, facevid2vid, wav2lip, sadtalker, MRAA, pirender,
  tpsm, and lia.

The dataset directory also contains indices for fully generated imagery, but
those indices were not part of H1. They must not be added to E2 because doing
so would break the controlled H1 comparison.

## 3. Canonical semantic labels

Every candidate string maps to one canonical semantic ID:

- authenticity: real or fake;
- boolean response: yes or no;
- provenance: camera, swap, or reenact.

The target is computed from the canonical meaning before candidates are
permuted. target_index is the position of that meaning after permutation.

For every output row, the evaluator retains:

- task_id;
- question_id and question text;
- candidate IDs, candidate text, and canonical candidate meanings;
- target_index;
- real/fake label;
- method_id and method_family;
- sample_id, image_path, video_id, and video_key.

This retained semantic metadata is required to map probabilities back from
candidate positions to canonical meanings.

## 4. Five semantic tasks

### T1: canonical authenticity status

Purpose: preserve the H1 fixed-candidate task and measure basic candidate
semantics.

Example:

- question: What is the authenticity status of this facial image?
- candidates: an authentic facial image; a manipulated deepfake facial image.

The candidates are randomly permuted during training. The exact H1 question
and candidates are also retained as a distinct Fixed-H1 evaluation condition.

### T2: positive and negative polarity

Purpose: force the model to use question meaning.

Training questions include authenticity-positive and manipulation-positive
forms, for example:

- Is this facial image authentic?
- Does this face appear unaltered?
- Is this facial image manipulated?
- Has this face been digitally altered?

Candidates are variants of yes and no. The same visual label has opposite
target positions under authenticity-positive and manipulation-positive
questions.

### T3: synonymous authenticity descriptions

Purpose: prevent the candidates from collapsing to two memorized strings.

Training real descriptions include:

- a real human face photo;
- a genuine facial image;
- an unaltered facial photograph.

Training fake descriptions include:

- a fake face photo;
- a manipulated facial image;
- a forged facial photograph.

Question and candidate paraphrases reserved for held-out evaluation must have
no exact full-string overlap with the training language bank.

### T4: manipulation-family question

Purpose: use the existing method labels and test whether the question changes
the answer for the same fake image.

Questions ask whether the face is identity-swapped or facially reenacted.
Targets are:

| Visual truth | swap question | reenactment question |
| --- | --- | --- |
| real | no | no |
| identity swap | yes | no |
| facial reenactment | no | yes |

For a fake sample, training alternates deterministically between a matching
question and a counterfactual non-matching question. For a real sample, the
queried family is selected uniformly. Evaluation always includes both family
questions. Because this produces a non-uniform yes/no distribution, balanced
accuracy is the primary T4 classification metric.

### T5: coarse provenance

Purpose: test a richer forensic schema with a variable candidate count.

T5 is a three-candidate task:

- camera-captured and unmanipulated;
- identity-swapped;
- facially reenacted.

Targets are:

| Visual truth | T5 target |
| --- | --- |
| real | camera |
| identity swap | swap |
| facial reenactment | reenact |

No H1 sample is labelled as fully synthetically generated.

## 5. Language banks and split discipline

The training language bank contains nine question templates:

- one canonical T1 question;
- four polarity templates for T2;
- one generic description-selection question for T3;
- one swap and one reenactment question for T4;
- one provenance question for T5.

Each task also has several candidate surface forms. Yes/no answer forms and T5
provenance descriptions have multiple training phrasings.

The held-out language bank contains new question and candidate paraphrases for
all five tasks. Held-out full strings:

- are absent from training;
- are absent from source-validation checkpoint selection;
- are encoded only during final semantic evaluation;
- remain fixed before the first target-domain evaluation.

Vocabulary tokens may naturally overlap; the isolation unit is the complete
question or candidate string.

## 6. SemanticTemplateSampler

Each base image is dynamically converted to:

    image, question, candidates, target_index

plus the metadata in Section 3.

Training behavior:

1. Select T1 through T5 with equal probability, approximately 20 percent each.
2. Select a task-compatible question and candidate surface form.
3. Derive the target canonical meaning from label and method_family.
4. Permute candidates.
5. Return the permuted target_index and canonical semantic metadata.

Randomness must be reproducible across runs, DataLoader workers, and DDP ranks.
Task selection is derived from a stable cryptographic hash of:

    global_seed, epoch, sample_id

Task-internal choices are derived from a separate stable cryptographic hash of:

    global_seed, epoch, sample_id, task_id

Python's process-randomized hash function must not be used. The sampler exposes
set_epoch so every sample can receive new but deterministic language at each
epoch.

Validation and test suites are deterministic. They use preregistered template
and candidate assignments rather than runtime random draws. Candidate
permutations are exhaustive in the dedicated invariance audit: two
permutations for binary tasks and six for T5.

## 7. Model

### 7.1 Frozen encoders

- vision: the exact H1 CLIP ViT-B/16 visual encoder and weights;
- text: the matching frozen CLIP text encoder.

Both encoders remain in evaluation mode and have requires_grad disabled.

For an image, question, and candidates:

    v = CLIP.encode_image(image)
    q = CLIP.encode_text(question)
    c_i = CLIP.encode_text(candidate_i)

All pooled embeddings are 512-dimensional. Because the text bank is finite and
the encoder is frozen, unique text embeddings are precomputed once. Held-out
embeddings are computed only when held-out evaluation begins.

### 7.2 Trainable projections

Image, question, and candidate embeddings use separate trainable projection
blocks:

    LayerNorm(512) -> Linear(512, 256)

Separate projections both align modalities and distinguish image from question
memory tokens.

### 7.3 Decision transformer

The decision transformer is a two-layer pre-normalized Transformer Decoder:

- d_model: 256;
- attention heads: 8;
- feed-forward dimension: 1024;
- dropout: 0.1.

The memory sequence is:

    image_token, question_token

Candidate tokens are decoder queries. Candidate self-attention has no
candidate-position embedding, after which every candidate cross-attends to
image and question memory. A padding mask supports two or three candidates.

### 7.4 Shared scorer and loss

Every candidate uses the same scorer:

    LayerNorm(256) -> Linear(256, 1)

Masked candidate logits are trained with cross entropy against target_index.
There is no contrastive loss, method auxiliary head, calibration loss, or
generative objective.

Only the three projections, two decoder layers, and shared scorer are
trainable.

### 7.5 Permutation equivariance

With no candidate position embeddings and shared parameters, permuting
candidate tokens must permute candidate logits in the same way. In evaluation
mode, mapping probabilities back to canonical semantic IDs must produce:

- maximum absolute probability error no greater than 1e-6;
- identical canonical predictions for every tested permutation.

This property is both unit-tested and audited on real evaluation samples.

## 8. Training protocol

Training inherits H1 settings unless a setting is architecture-specific:

- seeds: 42, 7, and 2048;
- epochs: 15;
- learning rate: 1e-4;
- weight decay: 1e-2;
- warmup: one epoch;
- EMA decay: 0.999;
- early stopping patience: five;
- per-GPU batch: 128, subject only to a documented memory reduction;
- mixed precision and four-GPU DDP on GPUs 0 through 3.

The source validation objective is deterministic seen-language cross entropy
over all five tasks. Checkpoint selection may use only FF++ validation and
seen-language strings. It must not inspect CDF, DFD, held-out language, or
meaningless-text controls.

A one-seed smoke run verifies correctness and memory use. It is not an
experiment result. After the protocol is locked, the three final seed runs are
the reported results. Software defects may be fixed, but target results must
not drive architecture or threshold changes.

## 9. Evaluation conditions

Each final seed is evaluated on the same visual records under:

1. Fixed-H1: the exact H1 question and fixed candidates;
2. Seen-language: deterministic T1 through T5 training-language suites;
3. Held-out-language: deterministic T1 through T5 unseen paraphrases;
4. Candidate permutation: exhaustive candidate orders;
5. Question counterfactuals: paired polarity and paired method-family questions;
6. Meaningless candidates: length-matched pseudoword candidates with semantic
   assignment counterbalanced independently of image and candidate position;
7. Meaningless questions: unrelated or semantically empty questions with
   yes/no candidates for T2 and T4.

Primary domains:

- FF++ test;
- CDF external test.

DFD is exploratory for T1 through T3. Its fake records do not have a reliable
swap/reenact family label, so DFD cannot contribute to T4 or T5 gates.

## 10. Semantic probabilities and video aggregation

Candidate probabilities are first mapped from positions to canonical meanings.

Binary fake probabilities:

- T1 and T3: probability assigned to the fake description;
- T2: P(no) for an authenticity-positive question and P(yes) for a
  manipulation-positive question;
- T4: max of P(yes given swap question) and
  P(yes given reenactment question);
- T5: 1 minus P(camera).

Frame probabilities are then averaged per method_id::video_id. Using video_id
alone is forbidden because IDs repeat across manipulation methods.

Primary metrics:

- T1 and T3: video AUROC;
- T2: video balanced accuracy plus polarity consistency;
- T4: video balanced accuracy plus matched-versus-counterfactual ordering;
- T5: video macro balanced accuracy;
- Fixed-H1: video AUROC.

AP, accuracy, F1, Brier score, ECE, frame metrics, and per-method slices are
reported as secondary diagnostics. Brier and ECE are not pass gates because H1
did not establish a calibration advantage.

The E2-Base comparator is the completed H1-Jev checkpoint with the matching
seed and cached predictions where available.

## 11. Uncertainty

Comparisons use paired video-cluster bootstrap with 10,000 resamples,
stratified by label and method where the metric permits. Each resample keeps
the same sampled video keys for the two paired conditions.

The three-seed mean is the primary point estimate. Seed-wise results and seed
standard deviation are also reported. Bootstrap intervals condition on the
three preregistered seeds; seeds are not resampled from a population of three.

Non-inferiority uses a one-sided 95 percent lower bound. Other reported
intervals are two-sided 95 percent intervals.

## 12. Preregistered gates

### G1: fixed-task non-inferiority

For both FF++ and CDF:

- the three-seed mean Fixed-H1 video-AUROC difference
  E2-Semantic minus H1-Jev is at least -0.01;
- the paired video-bootstrap one-sided 95 percent lower bound is at least
  -0.01.

### G2: held-out language generalization

Define each domain's macro semantic score as the unweighted mean of:

- T1 video AUROC;
- T2 video balanced accuracy;
- T3 video AUROC;
- T4 video balanced accuracy;
- T5 video macro balanced accuracy.

For both FF++ and CDF:

- seen minus held-out macro semantic score is no greater than 0.02;
- no individual task drops by more than 0.05.

### G3: candidate-order equivariance

Across all audited binary and T5 permutations:

- maximum canonical probability error is no greater than 1e-6;
- canonical prediction consistency is 100 percent.

### G4: question semantics

On held-out language:

- T2 balanced accuracy is at least 0.80;
- paired opposite-polarity logical consistency is at least 0.95.

The consistency test asks authenticity-positive and manipulation-positive
questions of the same visual item. Their literal yes/no answers must invert
while the implied real/fake decision remains consistent.

### G5: method and provenance semantics

On held-out language:

- T4 balanced accuracy is at least 0.65;
- at least 0.80 of fake video pairs score the true-family question above the
  counterfactual family question;
- T5 macro balanced accuracy is at least 0.60.

These requirements apply to both FF++ and CDF.

### G6: meaningless-text degradation

Relative to the paired meaningful condition:

- meaningless candidates reduce the applicable primary semantic score by at
  least 0.10;
- meaningless questions reduce T2 and T4 primary scores by at least 0.10;
- the paired-bootstrap 95 percent lower bound for each required degradation is
  greater than zero.

The score reduction is computed after canonical remapping and with
counterbalanced hidden assignments, so output-slot preferences cannot satisfy
the control.

## 13. Decision rule

E2 is supported only if G1 through G6 all pass on their specified domains and
conditions.

If a point estimate violates a threshold, E2 is not supported and the failed
gate identifies the failure mode. If all point thresholds pass but a required
confidence bound does not, the conclusion is insufficient evidence.

Secondary metrics cannot override a failed primary gate. Post-hoc analyses may
explain a failure but cannot change the preregistered verdict.

## 14. Alternatives considered

### Token-level visual/text cross-attention

Using CLIP visual patch tokens and text token sequences is more expressive, but
it changes the H1 pooled visual representation, requires custom token
extraction, substantially increases cost, and confounds semantic reasoning
with new spatial modeling. It is deferred.

### Pooled concatenation MLP

Concatenating pooled image, question, and candidate embeddings is cheaper, but
it is too close to H1-Jev and provides weaker evidence for a genuine
candidate-conditioned decision mechanism.

The selected pooled-token decoder preserves the H1 visual representation while
adding an explicit, permutation-equivariant question/candidate decision stage.

## 15. Required outputs

The completed experiment must retain:

- source code and unit/integration tests;
- frozen configuration and language-bank manifests;
- three final checkpoints and run metadata;
- frame- and video-level semantic predictions or reproducible compact caches;
- per-seed and aggregate metric tables;
- bootstrap intervals and gate-status table;
- a final report stating supported, not supported, or insufficient evidence;
- exact commands needed to reproduce training and evaluation.
