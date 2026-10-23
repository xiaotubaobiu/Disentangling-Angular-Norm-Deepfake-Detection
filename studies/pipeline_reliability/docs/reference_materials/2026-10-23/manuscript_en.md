# Pipeline Pitfalls in Deepfake Video Detection: From Preprocessing History to Cross-Domain Decisions

**Working manuscript, version 1.0 — 23 October 2026.** This is a pre-results draft for the newly specified experiments. The methodological text is complete enough to guide execution. New empirical tables and quantitative conclusions remain pending. Previously reported 31-method exploratory results are not relabeled as results of the new 21-method protocol.

## Abstract

The reported performance of a deepfake detector depends not only on its learned representation but also on the history of its inputs, inference-time feature operations, video scoring, and decision thresholds. These factors can change different statistical objects and should not be interpreted as interchangeable improvements in detection ability. We develop a controlled empirical study of pipeline reliability along four stages. A raw-video track uses FaceForensics++ and DeepFakeDetection to compare models trained on directly processed inputs with models trained on inputs that have already undergone another preprocessing pipeline. A complementary track restricts DF40 to 21 face-swapping and face-reenactment methods and compares raw and unit-normalized features, logit and probability aggregation, and correct and deliberately colliding video identities using shared models and frame caches. Analytical controls distinguish common score transformations from sample-dependent normalization, aggregation nonlinearity, and numerical ties. Finally, frozen detectors are evaluated with source thresholds and independently calibrated thresholds based on limited genuine target videos. Target-label optimal operating points are retained only as post-hoc diagnostics. The study is designed to determine which observed changes reflect useful discrimination, which arise from evaluation artifacts, and which can be addressed by threshold adaptation. It does not assume that normalization or repeated preprocessing is universally harmful.

**For the results version:** insert verified effect ranges for preprocessing history, inference normalization, aggregation/identity interaction, and independent target calibration. Do not submit this design-only abstract as an account of completed experiments.

**Keywords:** deepfake video detection; preprocessing history; feature normalization; evaluation reliability; score aggregation; threshold transfer.

## 1. Introduction

Video deepfake detection commonly learns from face frames and combines frame-level predictions into a video decision. FaceForensics++ established a widely used benchmark for facial manipulation, while DF40 broadened manipulation coverage and evaluation protocols [R1,R3]. Pretrained vision encoders provide useful detection representations, but a detector's final measurement still depends on how an image reaches the encoder, how the learned features are used, which frames define one evaluation unit, and how a continuous score is converted into a decision.

These dependencies should not be conflated. A released face crop that is processed again during training may have a different history from a fresh deployment input. Applying L2 normalization only after a raw-feature detector has been trained changes the representation seen by its classifier. Averaging logits and averaging frame probabilities define different video statistics. Combining videos from different manipulation methods because their local identifiers coincide changes the evaluation unit itself. Even after these implementation issues have been resolved, a source-selected threshold need not maintain its false-positive rate on a new genuine-video distribution.

Standardization work already recognizes that inconsistent processing and evaluation can produce unfair comparisons [R4]. Our objective is therefore not to rediscover the importance of consistent protocols. Instead, we specify paired interventions that reveal which conclusions depend on particular pipeline choices. The distinction is consequential: a higher reported AUC can coexist with an invalid evaluation unit, and a probability threshold closer to 0.5 can occur without any improvement in classification.

The study is also motivated by a retrospective audit of the authors' earlier conference work. That work associated inference-time normalization with cross-domain AUC improvements. Subsequent code inspection identified method-agnostic video identities and associated sampling/aggregation problems. The earlier implementation is retained as a traceable artifact rather than silently overwritten. Because the historical checkpoints are no longer fully available, the present experiments test the identified class of failure using new models; they do not claim exact recovery of every historical result. The complete conference citation and affected claims must be supplied in the final version.

We ask four questions. RQ1: does hidden training-input history alter performance under an otherwise fixed deployment preprocessing pipeline? RQ2: does inference-time normalization merely change score scale, or does it change cross-domain ordering? RQ3: how do aggregation, evaluation-unit collisions, and numerical precision affect reported AUC and the relative assessment of raw and normalized inference? RQ4: how much decision performance can a fixed detector recover using limited genuine target calibration data?

Our contributions are an empirical design and auditable evidence, not a new detection architecture. We provide complementary raw-video and multi-manipulation tracks; paired score analyses with explicit analytical negative controls; and an information-budget-aware evaluation of threshold transfer. The specific empirical findings are determined by the new results, rather than assumed in advance.

## 2. Related Work

### 2.1 Benchmarks and processing consistency

FaceForensics++ contains Deepfakes, FaceSwap, Face2Face, and NeuralTextures [R1]. Google's DeepFakeDetection contribution provides another source of manipulated face videos [R2]. DF40 contains a broader collection of generation and manipulation workflows [R3]. Our DF40 track excludes entire-face synthesis and face editing, because its target unit is a natural video manipulated by face swapping or reenactment. The selected DF40-derived FF and CDF pools are not interchangeable with the original four-method FF++ or standard Celeb-DF protocols.

DeepfakeBench addresses consistent data management, detector implementation, and evaluation [R4]. Related generated-image work shows that JPEG and image-size biases can influence both learned detectors and their apparent generalization [R5]. Our preprocessing experiment differs from comparing unrelated image collections: it derives both input histories from the same underlying videos and evaluates both trained models on the same single-pass deployment inputs.

### 2.2 Feature normalization and adaptation

CLIP learns visual representations using image–text supervision [R9]. We use CLIP, DINOv2, and EfficientNet [R9–R11] as concrete pretrained backbones, without treating differences among three models as an identified causal effect of their pretraining objectives. GenD combines LayerNorm adaptation, normalized features, and additional geometric objectives [R6]. That matched training procedure is not equivalent to inserting L2 normalization after raw-feature fine-tuning. Our results consequently cannot establish whether all normalized-training methods are beneficial or harmful.

### 2.3 Calibration and threshold control

Probability calibration concerns the agreement between predicted probabilities and observed frequencies [R7]. Selecting a threshold to constrain false positives is a distinct objective. Neyman–Pearson classification formalizes prioritized error control and supplies order-statistic calibration rules with finite-sample conditions [R8]. We study empirical genuine-only calibration as a practical baseline and use existing statistical controls to interpret its limitations. We do not propose a new risk-control theorem.

Distribution-shift uncertainty evaluation also distinguishes reliable probabilities from in-domain post-processing success [R12].

## 3. Formulation and Analytical Controls

### 3.1 Pipeline and score definitions

Let A denote a publisher's preprocessing operation, B the deployment operation, and f_theta a visual encoder. A linear two-class head produces l = W h + b_vec. Fake is the positive class. The scalar frame margin is z = l_fake − l_real, and its probability representation is sigmoid(z). A single fake-class logit is not substituted for this difference.

Each natural video has an identity containing dataset, domain, manipulation method, and local video ID. Genuine videos are deduplicated by actual source identity rather than duplicated for every fake method. Each evaluated video contains eight different RGB frames. We compare mean-logit S_L = mean_t z_t and mean-probability S_P = mean_t sigmoid(z_t). A collision audit intentionally omits the manipulation component from fake-video identities. The resulting score represents an artificial group, not an actual video.

### 3.2 Inference normalization

Write h = r u with r = ||h||_2 and ||u||_2 = 1. Define w = W_fake − W_real, b = b_fake − b_real, and q = w^T u. Then z_R = r q + b and z_D = q + b. The encoder and classifier are fixed in all inference comparisons.

A common positive scaling preserves ordering; removing sample-specific radii need not. We therefore compare raw features with unit directions and with directions rescaled by a common positive radius estimated from source training data. Under mean-logit aggregation, the two direction-based conditions induce the same ordering. A persistent raw advantage cannot then be explained solely by restoring a common amplitude. It remains a result about the fixed representation and readout, not a proof that all directional classifiers are inadequate.

### 3.3 Aggregation nonlinearity and finite precision

Near zero, sigmoid(z) = 1/2 + z/4 − z^3/48 + O(z^5). If all relevant margins occupy a common approximately affine region, mean-probability can be approximately order-equivalent to mean-logit. L2 normalization alone does not guarantee this condition: classifier weights and biases remain unrestricted.

More generally, the absolute gap between mean(sigmoid(z)) and sigmoid(mean(z)) is bounded by Var(z)/(12 sqrt(3)), using the global maximum absolute second derivative of the sigmoid. This bound describes a potential aggregation discrepancy, not its effect on AUC or the direction of a ranking change. Combining methods introduces a between-method component into the variance of the mixed frame distribution, which can interact with curvature.

Finite precision is analyzed separately. Sigmoid applied after video-logit averaging is strictly increasing in exact arithmetic, but finite representations may collapse different scores into ties. We count exact ties and recompute relevant conditions from cached margins at increasing precision. Recalculation does not restore information previously lost in encoder-level mixed precision.

### 3.4 Threshold proximity is not a performance measure

For a final video margin S, let p_a = sigmoid(aS), a > 0. Equivalent thresholds obey tau_a = sigmoid(a logit(tau_1)). AUC, the maximum attainable empirical balanced accuracy, and decisions at corresponding thresholds are invariant in exact arithmetic, while tau_a approaches 0.5 as a approaches zero. Thus numerical proximity to 0.5 is not evidence of improved calibration or transferable decisions. Actual feature normalization is sample-dependent and can simultaneously change scale and ordering.

### 3.5 Metrics and estimands

Pooled AUC treats every actual video as one entry. Method-macro AUC averages each manipulation's AUC against the same genuine-video pool. Both are binary detection metrics. They differ in fake-method weighting, not in whether the detector predicts a manipulation class. A collision-group AUC is reported only as an audit output; method-macro AUC is undefined after method identities have been merged.

At threshold tau, FPR is the proportion of genuine videos classified as fake, TPR is fake-video sensitivity, and BA = (1 + TPR − FPR)/2. Pooled and method-macro TPR/BA are labeled separately. Target-test maximum BA and low-FPR ROC vertices are optimistic sample-level diagnostics, not deployable results or population upper bounds.

## 4. Data and Locked Protocol

### 4.1 Complementary tracks

Track H trains on original FF++ four-method videos and tests on held-out FF++ and DFD. It compares B(X) with B(A(X)). Pipeline A uses registered five-point alignment and a tighter field of view, storing 256-pixel PNG images; pipeline B uses the registered wider crop and produces 224-pixel inputs. The second pass operates only on the saved first-pass image and obtains annotations in that image's coordinate system. It cannot retrieve cropped context from the original frame.

Track D trains on the FF-domain portion of 21 registered DF40 methods: nine face-swapping and twelve reenactment methods. It evaluates the corresponding FF and CDF video pools. Entire-face synthesis and face editing are excluded. Source official roles, known provenance links, and pixel hashes determine disjoint training, development, calibration, and testing manifests. Unknown ancestry is recorded rather than claimed independent.

Both tracks may share underlying FF++ content and are not presented as independent replications. DFD and CDF were also historically inspected during development; the new protocol is frozen for execution, but the project is not described as wholly blind or preregistered.

<!-- TABLE:T1 Actual source groups, methods, videos, frames, and exclusions for each role. -->

### 4.2 Training and convergence qualification

CLIP ViT-B/16, DINOv2 ViT-B/14, and EfficientNet-B0 use explicit non-L2-normalized feature boundaries and linear two-class heads. Full-parameter fine-tuning uses cross-entropy on raw features. Five formal seeds are 7, 42, 123, 2026, and 3407. Genuine and fake sampling each receive probability 1/2; fake methods are balanced, followed by equal video and frame sampling within a method.

Source-only development determines the fixed training recipe and endpoint before formal target evaluation. DINOv2 receives a bounded convergence check because earlier fixed-budget runs showed variable source performance. This is not a search for target-domain gains or for parity with another backbone. Preprocessing-history pairs share initialization, data identities, sampling plans, augmentation randomness, and training budget. Inference interventions do not require additional training.

<!-- TABLE:T2 Frozen learning rates, update budgets, source development metrics, and successful AMP updates. -->

### 4.3 Calibration and uncertainty

Every correct score pipeline obtains its own source genuine threshold. Separate target genuine videos support nested calibration budgets of 10, 20, 40, and 80 when independent provenance and sample support allow them. All budgets use the same held-out target test pool. Two hundred nested sampling repetitions share calibration IDs across model and score comparisons. Selecting all 80 videos from a fixed 80-video pool does not create 200 independent calibration sets.

We report individual seeds and mean ± sample SD. Paired source-group bootstraps quantify conditional test uncertainty for fixed models; calibration-subset variability is reported separately. Missing methods invalidate a macro-bootstrap replicate rather than changing the estimand. Colliding and correct evaluation units cannot be subjected to an ordinary one-to-one paired bootstrap; resampling must act on their common source groups before evaluation.

## 5. Preprocessing-History Results

<!-- TABLE:T3 Two training histories by two testing histories on FF++ and DFD. -->

The primary contrast compares a directly trained model and a history-trained model on the same single-pass B test inputs. Matched B(A(X)) testing is an auxiliary control. We first report preprocessing coverage and failure rates because condition-specific exclusions could otherwise change the test population.

**Pending empirical paragraph:** For {{H_BACKBONE}} on {{H_DOMAIN}}, the history-training effect is {{H_DELTA}} percentage points with conditional interval {{H_INTERVAL}}. The common eligible cohort covers {{H_COVERAGE}} of the registered videos. The matched-history comparison gives {{H_MATCHED_EFFECT}}. These values must come from new runs. A null or reversed effect is retained; pipeline parameters are not changed after observing it.

## 6. Normalization and Video-Scoring Results

### 6.1 Correct evaluation

<!-- TABLE:T4 Correct-grouping raw/L2 by logit/probability aggregation. -->

We first assess normalization under valid video identities. Common-scale and common-radius controls test whether a scalar score explanation is sufficient. A three-state real–fake pair ledger reconstructs the AUC change and reveals beneficial and harmful ordering changes that could cancel in a net metric.

**Pending empirical paragraph:** The normalization effects across backbones are {{N_EFFECTS}}. Restoring a common source radius changes these results by {{N_RADIUS_CONTROL}}. These statements apply to the trained checkpoints and fixed heads; they do not identify a causal property of contrastive pretraining.

### 6.2 Evaluation-unit collisions

<!-- TABLE:T5 Eight-cell audit and differences in raw/L2 rankings, with unit counts. -->

For the collision audit, we report the number of actual videos replaced by each artificial group and compare raw/L2 differences within the same grouping rule. An apparent normalized-feature advantage under the defective protocol is not described as improved video detection. Where the relative ranking reverses, its scope is limited to the evaluated models and data. Failure to reproduce the historical CLIP sign is not hidden.

### 6.3 Nonlinearity versus numerical ties

<!-- TABLE:T6 Logit ranges, affine residuals, pair transitions, and precision convergence. -->

We test the local-linearity explanation using frame-margin distributions and explicit aggregation residuals rather than AUC similarity alone. High-precision results determine whether an observed gap survives after representational ties have been reduced. The analysis retains opposing pair transitions; a small net difference is not equated with identical ranking.

## 7. Cross-Domain Decision Reliability

### 7.1 Frozen source operating points

Source thresholds and target score distributions are evaluated on the same target test cohort used for all subsequent calibration comparisons. Genuine-score empirical cumulative distributions and marked ROC operating points show the location of a transferred threshold. These distributions diagnose score transfer but do not, by themselves, identify the visual source of domain shift.

### 7.2 Limited genuine target calibration

<!-- TABLE:T7 Default/source/target-real calibration at each budget, with FPR, TPR, BA, and subset spread. -->

A reduction in FPR is interpreted jointly with fake sensitivity. An average BA plateau is not evidence that a sample budget is sufficient for reliable false-positive control. Under independent continuous genuine scores, allowing k = floor(n alpha) empirical false positives and thresholding just above the (n−k)-th order statistic gives expected out-of-sample FPR (k+1)/(n+1). This provides a finite-sample reference for the empirical procedure, not a guarantee for correlated video samples.

An existing nonparametric Neyman–Pearson rule is an optional statistical comparator. Where its independent-sample requirement is not met, the result is marked infeasible or descriptive; it is not replaced by a claim of guaranteed error control.

### 7.3 Limits of threshold recovery

<!-- TABLE:T8 Same-cohort AUC, attainable low-FPR vertices, and target-test maximum BA. -->

Threshold selection moves along the ROC of a fixed score. It cannot recover an ordering already changed by feature or scoring operations. We compare independently selected operating points with sample-level diagnostic ROC vertices to separate limited calibration precision from poor low-FPR discrimination. Actual attained FPR is reported without interpolation when the genuine sample count makes the ROC discrete.

## 8. Discussion and Limitations

Inference normalization is a diagnostic intervention, not a proposed universally useful enhancement. Its interpretation requires distinguishing common scale, sample-specific feature dependence, and classifier compatibility. Likewise, a video-identity collision is a concrete implementation defect, not evidence that the entire literature uses the same defect. The historical case motivates tests but does not authorize extrapolating its prevalence.

The preprocessing track identifies the effect of specified composite histories. It does not independently attribute all differences to alignment, crop size, compression, or irreversible information loss. The multi-method track studies content-domain transfer for methods included in source training, not open-world detection. Three backbones do not isolate the causal role of pretraining or architecture. Limited genuine target videos and unknown provenance constrain both confidence statements and deployment extrapolation.

For reproducible use, we recommend publishing input-history definitions, actual video identities, fixed frame manifests, score definitions, and threshold-data permissions alongside checkpoints. A processed-image benchmark can still support valid comparisons when its versions and downstream processing are controlled. Conversely, raw inputs alone do not ensure fair evaluation without a documented pipeline.

## 9. Conclusion

This study separates input history, representation use, video scoring, and threshold decisions in deepfake detection. Its final empirical conclusions must be limited to the effects supported by the new controlled runs. The intended outcome is not a universal preference for raw or normalized features, or for logit or probability averaging. It is an auditable distinction between changes in useful discrimination, changes in evaluation, and changes in the operating point. Deployment-oriented conclusions are based on independently calibrated target evaluation rather than the numerical proximity of an oracle threshold to 0.5.

## Reproducibility and relation to prior work

The formal citation, DOI, and publication version of the authors' earlier conference paper remain to be supplied: {{PRIOR_CONFERENCE_CITATION}}. The uploaded thesis-style PDF is not assumed to be that publication. Legacy implementation artifacts, known limitations, corrected code, and new results are versioned separately. Missing historical checkpoints prevent an exact retrospective reassessment. Repository notices and a later paper do not replace any formal correction process required by the original publisher.

## References

See `references.md` for verified source metadata. The final bibliography must include the authors' earlier conference paper and receive a final literature check before submission. Unverified references from exploratory conversations are not treated as publication facts.
