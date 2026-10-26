---
bibliography: journal_references.bib
link-citations: true
---

# Evaluation Reliability in Deepfake Detection Pipelines: Preprocessing History, Feature Scoring, Video Identity, and Threshold Adaptation

> Working integrated draft: the D primary matrix, training terminals of all 45 formal models, and H numerical pairing across all three backbones are accepted. H caches, analyses and the four-core source-best sensitivity remain incomplete. The abstract, H results and full-study conclusions retain their three PENDING blocks. This is not a submission manuscript.

## Abstract

<!-- PENDING:ABSTRACT -->

The complete abstract will be written after H results and the final claim-to-evidence checks.

Keywords: deepfake detection; evaluation reliability; input history; feature normalization; video grouping; threshold adaptation

## 1. Introduction

The object evaluated in deepfake detection is an entire pipeline from image construction to a video decision. Face localization and cropping, assignment of frames to videos, conversion of features into scores, and the data used to determine a threshold jointly define the measured system. FaceForensics++ provides a resource for facial-manipulation evaluation, DF40 expands manipulation coverage and evaluation protocols, and the Google/Jigsaw release contributes visual-forgery videos to the FaceForensics framework [@rossler2019ffpp; @yan2024df40; @dufour2019dfd]. Interpreting a comparison therefore requires specifying input history, evaluation units and operating points alongside model weights.

Preprocessing history concerns the relationship between training and deployment. Detecting a face again after alignment or cropping is a different operation from processing the corresponding original video frame directly. An additional operator in the training history can therefore create a processing-chain mismatch when deployment applies only the current operator. Released images alone do not generally identify the effect of each historical step. We instead construct two explicit chains from the same original mother frames: direct cropping B(X), and alignment followed by a new detection and crop B(A(X)). The intervention concerns these constructed operators; it does not recover the unknown history of released images. The training histories share eligible videos, mother frames and random streams. The deployment comparison and the auxiliary common-success cohort are defined separately to retain coverage differences associated with processing failures.

After training, representation and evaluation identity introduce further changes. Expressing a feature as radius times direction separates Raw-feature scoring from scoring with its L2 unit direction. Replacing the input to a fixed head trained on Raw features is an inference intervention, rather than a comparison between algorithms trained to use normalized representations. Averaging frame logits and averaging frame probabilities also need not commute with the nonlinear probability mapping. Common scale, sample-specific radius and finite-precision ties consequently require separate checks. If a video identifier additionally loses its method field, identically named videos from different manipulations can be merged, changing both the evaluation unit and its frame count. Whether this grouping error modifies the normalization effect calls for a same-seed interaction, rather than a comparison of absolute AUCs computed over different video sets.

Ranking and deployment operating points answer distinct questions. AUC describes the ordering of real and fake video scores. False-positive and true-positive rates at a frozen threshold depend on where those distributions lie relative to that threshold. An empirical false-positive constraint on source-real videos does not itself guarantee the same target-domain rate. A small target-real sample defines a threshold-adaptation problem, but neither establishes probability calibration nor turns unresolved ancestry into independent observations. Every calibration budget and post-hoc oracle diagnostic is therefore evaluated on the same target TEST, isolated from the entire candidate calibration pool. Ranking on the full target pool is reported separately.

Two complementary routes implement these comparisons. Route D uses released DF40 images from nine face-swap and twelve face-reenactment methods, applying representation, aggregation and grouping interventions to the same model caches. Route H constructs preprocessing histories from FaceForensics++ videos and evaluates the deployment contrast on the local DeepFakeDetection (DFD) pool. The routes address different interventions; their manipulation mixtures, image construction and test cohorts are not pooled into a single overall effect. CLIP ViT-B/16, DINOv2 ViT-B/14 and EfficientNet-B0 use recipes frozen after source-domain development, with five formal seeds reported separately for each backbone. The questions and primary contrasts are fixed as follows.

- **RQ1: training preprocessing history.** On all B-qualified DFD test videos, compare models trained on B(A(X)) and B(X) using the same deployment input, H10−H00. FaceForensics++ supplies the source-domain check, and the common-success cohort supplies the auxiliary 2×2 comparison.
- **RQ2: inference normalization under a fixed head.** Compare same-seed L2−Raw pooled AUC on CDF-full using correct video grouping and mean-logit aggregation. This is not a comparison between normalization-based training algorithms.
- **RQ3: grouping-by-normalization interaction.** On CDF-full, subtract the correct-grouping L2−Raw effect from the collision-grouping effect, separately for mean-logit and mean-probability. Per-seed reversal counts remain distinct from reversal of the average effect.
- **RQ4: real-only threshold adaptation.** On one fixed calibration-test cohort, compare thresholds based on 20 target-real videos against source-real thresholds in FPR, TPR and pooled balanced accuracy. Budgets 10, 40 and 80 and the default threshold are also reported. Twenty is the prespecified comparison budget, without an assumption of adequacy or optimality; oracle results are post-hoc diagnostics.

The design aims to make the comparison objects associated with input history, evaluation identity, representation scoring and threshold adaptation explicit. Common caches, paired random streams, source and content audits, numerical order-preservation controls and independent metric recomputation constrain interpretation without assuming that any operation improves detection. Sample SD across five training seeds, variation across 200 calibration subsets, and conditional uncertainty from 2,000 TEST resamples describe different levels and are not combined as independent model evidence. The documented scope of source graphs and proxies also limits claims about independence and population false-positive control.

This study extends earlier exploratory work in which the target domains had already been inspected. Freezing comparisons, data roles and selection rules before the new formal evaluation does not make the overall research a prospective blind test or a global preregistration. Conclusions remain conditional on the registered models, data, operators and fixed heads. Null effects, adverse effects, unavailable intervals and missing conditions are retained. Final empirical contributions and effect directions require the complete matrix and independently recomputed results.

## 2. Related work and scope of comparison

### 2.1 Benchmarks, input management and evaluation units

DF40 expands manipulation coverage and evaluation protocols for deepfake detection [@yan2024df40]. DeepfakeBench organizes unified data management, detector implementations and evaluation protocols within a common framework to improve consistency and reproducibility [@yan2023deepfakebench]. These resources support the data and implementation basis of detection experiments. Building on them, we freeze the method registry and data roles before defining two interventions: H constructs training preprocessing histories from the same mother frames, while D changes representation, aggregation and video grouping within common caches. A filename constitutes a study video identity only together with its method and data role; it does not establish independent generative ancestry.

Input construction determines each route's scope. The all-B deployment cohort and the common B/AB-success cohort in H answer different questions, with processing failures and coverage reported for each. Collision grouping in D is an explicit evaluation intervention that changes the unit relative to correct video grouping. Comparisons across those groupings use a common closure of source and collision relations, preserve resampling multiplicity, and remain descriptive when closure support is inadequate. The resulting interval scope depends on documented relations and registered proxies, rather than the number of filenames.

### 2.2 Foundation representations, normalization and magnitude

Ojha et al. study nearest-neighbor classification and linear probing in the feature space of a large pretrained vision-language model [@ojha2023universal]. Their approach makes the acquisition of a representation and its downstream reader joint parts of the comparison. In D, the encoder and two-output head are first adapted on the source domain and then held fixed. Interventions derived from common Raw caches consequently concern this trained encoder and its original head.

GenD combines restricted LayerNorm adaptation with L2 normalization and metric learning on a hypersphere [@yermakov2026gend]. Representation geometry is part of its training objective. Replacing Raw-trained features by unit directions at inference, while retaining the original head, defines a different intervention. An L2−Raw effect under that head therefore cannot establish the relative merit of all normalization-based training methods. We retain both logit and probability aggregation to examine dependence on the video-scoring rule.

The 2026 NormFake preprint studies magnitude and sparsity of pretrained features, including L1 norm and the L1/L2 ratio, for detection without an additional learned classifier [@smeu2026normfake]. It directly addresses the detection value of magnitude. Our object is instead the interaction of L2 radius with directional projection in a fine-tuned two-output head, using common-scale controls and order-transition ledgers to constrain the computational explanation. Results involving different pretrained models, learning procedures, norm statistics and cohorts retain their own scope. Neither a common class direction for magnitude nor a common semantic explanation is assumed across these protocols.

### 2.3 Probability calibration, threshold adaptation and false-positive control

Guo et al. examine agreement between predictive confidence and correctness and evaluate post-hoc methods including temperature scaling [@guo2017calibration]. Ovadia et al. systematically investigate accuracy and uncertainty calibration under dataset shift [@ovadia2019shift]. Probability reliability, real–fake ordering and error rates at a frozen operating point are related but differently defined quantities. Our scale controls are diagnostic; real-video calibration chooses a threshold subject to the specified empirical false-positive condition. Scores near 0.5 or an empirical source FPR alone do not establish reliable target probabilities.

Neyman–Pearson classification targets population type-I error control and distinguishes it from directly restricting empirical error [@tong2018np]. This distinction bounds our threshold interpretation: unresolved source and generation relations, finite real-calibration budgets and reuse of a common calibration pool require disclosure. RQ4 evaluates every budget on one fixed, isolated target TEST; post-hoc oracles use attainable operating points at actual score boundaries. A common TEST makes operating metrics comparable, but does not replace the independent-sampling and other assumptions needed for population control. Five training seeds, calibration-subset variation and conditional TEST intervals are reported separately, rather than multiplied into an independent replicate count.

## 3. Methods

### 3.1 Study design and prespecified comparisons

We study how preprocessing history, feature normalization, video identity, aggregation, and threshold adaptation affect the evaluation of a deepfake detector. Two complementary routes address different interventions. Route D uses released DF40 images from nine face-swap (FS) and twelve face-reenactment (FR) methods [@yan2024df40]. Route H starts from c23-compressed FaceForensics++ (FF++) videos [@rossler2019ffpp], constructs two preprocessing histories from the same mother frames, and evaluates transfer to the local DeepFakeDetection (DFD) video pool. These routes are reported separately because their manipulation sets, image origins, and input construction differ.

The primary comparisons are fixed before new formal target evaluation (Table M1). Each backbone is analyzed separately. The study is a controlled extension of earlier exploratory work, whose target domains have previously been inspected; it is not a prospective blind benchmark or a claim of globally preregistered research.

**Table M1. Questions and primary contrasts.** L denotes mean-logit aggregation; P denotes mean-probability aggregation. Differences use the same seed and the appropriate fixed cohort.

| Question | Route | Primary comparison |
| --- | --- | --- |
| RQ1: preprocessing history | H | H10−H00 on all B-qualified DFD test videos; FF++ is the source-domain check |
| RQ2: inference-time normalization | D | L2−Raw pooled AUC under correct grouping and L on CDF-full |
| RQ3: identity loss and scoring interaction | D | Grouping × normalization interaction on CDF-full, separately for L and P |
| RQ4: real-only threshold adaptation | D | Target-real n=20 versus source-only threshold: FPR, TPR, and pooled balanced accuracy on the same calibration-test cohort |

The remaining registered domains, budgets, scoring cells, and seeds will be reported in full. Neither the direction of an effect nor agreement with earlier findings is a stopping or selection criterion.

### 3.2 Video identity, source groups, and data isolation

The scoring unit is a natural video with UID dataset/domain/method/video_id. Real videos use method=real and enter the negative pool once. A source group is a different object: it is a connected component of documented original-video, donor, target, or driver relations and exact RGB links. Identity of a person alone does not establish identity of an original video, and distinct UIDs do not establish independent ancestry. Unresolved relations remain explicitly marked.

For D, the 21-method manifests were rebuilt from the complete released association indices before fixed-frame selection or canonical merging. Official roles are retained, and the 360 FF training source pairs are partitioned into 306 TRAIN and 54 DEV pairs using seed 20261023. H uses exactly the same pair assignment. Selected cross-role RGB overlap, real/fake label conflicts, and missing registered methods block data acceptance. Byte and decoded-RGB metadata, parent relations, role assignments, and processing versions are retained. The D acceptance reused previously audited image hashes after checking current file metadata for all 1,034,481 indexed paths; this is not a fresh cryptographic rehash of every image.

For D, valid frames are sorted by numeric frame index and then image path, the first occurrence of each decoded RGB is retained, and eight equally spaced positions are selected with floor rounding. Videos with fewer than eight distinct valid RGB frames are excluded. RGB-identical TRAIN inputs are stored canonically while all method contributions to their sampling mass are retained. The accepted TRAIN pool contains 611 real and 12,232 fake videos: 12,843×8=102,744 frame associations, reduced by 519 redundant associations in 493 cross-method RGB groups to 102,225 canonical inputs. Source 281 is absent from the original real indices, explaining 611 rather than the nominal 612 training reals. Eight SadTalker candidates with unresolved FF sources are quarantined; removing them does not authenticate all audio or driver ancestry.

The D CDF pool contains DF40-generated manipulations on CDF-source videos, rather than the unmodified Celeb-DF benchmark protocol [@li2020celebdf]. Two CDF cohorts have separate uses. CDF-full contains 177 real and 12,876 fake videos for RQ2/RQ3. The calibration-test analysis uses 80 real calibration videos and one fixed test set of 97 real and 4,111 fake videos. Video identity, full-candidate exact-RGB closure, and conservative name-token proxies isolate this test set from the entire 80-video calibration candidate pool, not merely a sampled calibration subset. This supports an empirical analysis conditional on those proxies. Complete generated-parent, donor/driver, audio, and identity ancestry is unavailable; complete ancestry independence and a strict population false-positive guarantee are not claimed. AUC and threshold metrics for RQ4 will all be recomputed on its own test set, without importing CDF-full AUC.

### 3.3 Feature decomposition and the eight D scoring conditions

Each encoder produces an unnormalized output h=fθ(x) for a frame x. With a two-output linear head, let w=Wfake−Wreal and b=bfake−breal. Define

$$
r=\|h\|_2,\quad \rho=\max(r,10^{-12}),\quad u=h/\rho,\quad q=w^\top u,
\qquad z_R=w^\top h+b=\rho q+b,\quad z_N=w^\top u+b.
$$

The epsilon convention and counts of zero or sub-epsilon norms will be recorded. Raw refers to the encoder output before any additional output normalization; architecture-internal LayerNorm or BatchNorm remains intact. L2 replaces h with u while keeping the original head and bias fixed. It is an inference intervention on a Raw-trained detector, not a separately trained normalization-aware model.

Formal D models will be forwarded in FP32 to cache Raw features and the head once per role. Radii, projections, margins, sigmoid operations, and aggregation will use FP64 unless the numerical diagnostics require higher precision. For representation j∈{R,N} and the fixed K=8 frames of video v,

$$
S_{j,L}(v)=K^{-1}\sum_{t=1}^{K}z_{j,t},
\qquad
S_{j,P}(v)=K^{-1}\sum_{t=1}^{K}\sigma(z_{j,t}).
$$

L ranks and thresholds the mean margin directly. Applying sigmoid after L is a diagnostic control and is distinct from P.

**Table M2. Eight cache-derived D conditions.** The real pool is unchanged across grouping conditions.

| Cells | Video grouping | Representation | Aggregation |
| --- | --- | --- | --- |
| 1–2 | Complete video UID | Raw | L / P |
| 3–4 | Complete video UID | L2 | L / P |
| 5–6 | UID with fake-method identity omitted | Raw | L / P |
| 7–8 | UID with fake-method identity omitted | L2 | L / P |

When omission of method merges k fake videos sharing an ID, the collision group pools their complete 8k frames with denominator 8k. It does not resample eight frames from the merged group. Cross-label ID conflicts block scoring. This deliberately simulates an evaluation identity error; it does not reintroduce earlier training or validation sampling errors or claim exact reproduction of an earlier 32×k-frame test.

Pooled AUC is primary and gives each correctly identified video unit equal weight. Correct grouping additionally reports method-macro AUC, averaging the fixed method-specific AUCs with the same real pool. Method-macro AUC under collision grouping is undefined and will be marked NA. For aggregation a, define

$$
\Delta_{g,a}=\mathrm{AUC}_{g,N,a}-\mathrm{AUC}_{g,R,a},
\qquad I_a=\Delta_{\mathrm{collision},a}-\Delta_{\mathrm{correct},a}.
$$

Both deltas and the interaction will be retained per seed. Reversal requires opposite nonzero signs within the same seed, using tolerance 10⁻¹⁰; a reversal in the across-seed mean is distinguished from the number of seeds that reverse. Fifteen D formal models yield 120 AUC cells per domain and 240 across FF and CDF without additional training.

Mandatory diagnostics apply positive scales a∈{0.1,1} to the complete frame margin, including its bias, before aggregation; substitute c·u for h, where c is the weighted median TRAIN radius; and compute sigmoid after mean-logit. They retain local affine residuals, Jensen gaps, frame- and method-level variance, exact probability endpoints, and ties. Raw→L2 and L→P transitions are decomposed into wrong/tied/correct real–fake ordering states. Define Tij as the normalized transition proportion: the raw transition count (or weighted count) divided by the total real–fake pair mass. With credits g=(0,0.5,1), the ledger must reconstruct ΔAUC=ΣijTij(gj−gi) within 10⁻¹⁰.

Probability ties/endpoints or a failed ordering-preserving control trigger recomputation from the same logits at 80 and 160 decimal digits, then 320 if needed. Sorting remains at that precision; unresolved ordering is reported as unavailable rather than accepted after conversion to FP64. No matched-head refitting is required in v2, so interpretation remains conditional on the original fixed head.

### 3.4 H preprocessing histories and deployment cohorts

H selects mother frames before face processing. A uniform policy decodes at most 32 uniformly spaced time candidates, removes duplicate RGB content, and selects eight positions by rounded equally spaced indices. If fewer than eight distinct frames remain, a chronological full-video fallback determines eligibility. Selection is independent of detector success, labels, and detector scores. B and AB share the same mother-frame identities; failed frames are not replaced separately for either history.

The accepted geometry uses the local five-landmark SCRFD detector only. It has a fixed 640×640 input, score threshold 0.5, and NMS IoU 0.4. Resizing uses scale min(1,640/width,640/height), floor-rounded dimensions, top-left placement, and zero padding, with exact inverse x/y scales. This no-upscale rule was fixed using source TRAIN/DEV feasibility checks before target preprocessing. Detector resizing uses OpenCV bilinear interpolation and RGB normalization (pixel−127.5)/128. NMS uses inclusive box areas (+1) and a stable score/coordinate order. Multi-face selection prioritizes bounding-box area, then confidence, then lexicographic box and landmark coordinates. For DFD this selects the largest detected face and does not certify that it is the manipulated face.

A(X) is an orientation-preserving least-squares five-point similarity alignment without RANSAC, saved as a 256×256 lossless PNG. The 112-coordinate template is [(38.2946,51.6963), (73.5318,51.5014), (56.0252,71.7366), (41.5493,92.3655), (70.7299,92.2041)]. Its target coordinates are

$$
T_{256}=128+\{(256/112)T_{112}-128\}/1.2.
$$

Thus margin 1.2 has an explicit geometric meaning. B(X) is an unrotated square crop of side 1.5·max(box width,box height), resized to 224×224, with the box center mapped to pixel center (111.5,111.5). AB=B(A(X)) independently redetects the disk-decoded A PNG and applies the same B operation; it never reuses mother-frame coordinates. OpenCV 4.11.0 bicubic interpolation and constant-zero borders are fixed. Valid-pixel masks use nearest-neighbor mapping, and AB inherits A's valid region.

A version is ineligible if any selected frame cannot be decoded or processed, has no qualified face, a zero-area box or nonfinite/degenerate landmarks, or if its eight final images do not have distinct RGB content. Padding is reported without a target-dependent exclusion threshold. Failure rates, multi-face rates, padding, and class/method coverage will accompany all H estimates.

H-B and H-AB training and DEV use the same natural videos qualified for both versions, with paired mother frames, initialization, sample streams, and augmentation values. The primary deployment cohort contains all B-qualified test videos. Its H10−H00 compares AB-trained and B-trained models on B inputs. A separate common B/AB-qualified test cohort supplies the complete 2×2 matrix (Table M3), including a newly computed H10−H00 on that subset. Estimates from different cohorts are not subtracted.

**Table M3. H history matrix on the common qualified cohort.**

| Training input | Test B(X) | Test B(A(X)) |
| --- | --- | --- |
| B(X) | H00 | H01 |
| B(A(X)) | H10 | H11 |

Raw L is the primary H score; P is a same-cache sensitivity analysis. Each model obtains its source threshold from B(SOURCE_CAL) reals, freezes it, and applies it to B(TEST). Any displayed AB operating point uses the same B deployment threshold and is identified accordingly.

### 3.5 Threshold policies and operating-point metrics

D threshold analyses use correct grouping only. Every model and scoring flow receives its own threshold; calibration IDs are common across flows within a repeat. The inclusive decision is score≥τ→Fake. Defaults are τ=0 for L and τ=0.5 for P. A source-only threshold is the smallest feasible threshold with empirical real SOURCE_CAL FPR≤0.05. Threshold boundaries use nextafter toward +∞ to handle ties, including probability thresholds above one when all observations must be classified real.

Target-real budgets are n∈{10,20,40,80}. Two hundred common random permutations provide nested subsets without video duplication; unavailable budgets are NA. Every budget uses the same fixed TARGET_TEST. At n=80, the complete 80-video pool provides one unique calibration set, not 200 independent experiments. Target-test oracle thresholds are post-hoc diagnostics: maximum pooled balanced accuracy over all attainable score boundaries and actual ROC vertices satisfying FPR≤5% or ≤10%.

Each row will include FPR, TPR, pooled balanced accuracy BA=(1−FPR+TPR)/2, the actual false-positive count, pooled and method-macro AUC, and method-macro TPR/BA. Real negatives are counted once. With 97 test reals, FPR≤5% permits at most four false positives (4/97≈4.12%); an interpolated 5% point will not replace the attainable point.

An order-statistic reference at α=δ=0.05 requires at least ceiling[log(δ)/log(1−α)]=59 independent units under the relevant independence and threshold assumptions; n=10,20,40 cannot support that nontrivial reference guarantee. The CDF proxy isolation does not certify those assumptions, including for n=80. RQ4 evaluates threshold adaptation and does not establish calibrated conditional probabilities.

### 3.6 Uncertainty and reporting

For each backbone, formal outcomes will be reported per seed and as the five-seed mean±sample SD. The five trained models, 200 calibration subsets, and 2,000 test bootstrap replicates describe different variation sources and will not be pooled as independent trained detectors.

Test bootstraps share sampled source groups across paired conditions. Known source relations define the units; CDF/DFD proxy units are explicitly identified as conditional approximations. For cross-grouping interactions, known source edges and collision-group edges first form a common connected closure. Each replicate resamples closure groups and reconstructs both grouping conditions. Repeated group draws retain a draw ID or exact multiplicity weights. A method-macro replicate missing any fixed method is invalid and its frequency will be reported. If closure leaves too few large groups for stable uncertainty estimation, the cross-grouping interaction will receive five-seed descriptive reporting without an ordinary paired interval.

Intervals are pointwise 95% conditional intervals and do not jointly cover training randomness, unknown ancestry, or all tested comparisons. Effects are expressed in percentage points with complete registered tables. A 4/5 or 5/5 sign pattern is descriptive, not automatic significance. No p-value or exclusion of a zero-crossing interval selects a finding; any added formal testing must first register its comparison family and multiplicity correction.

## 4. Experimental setup

### 4.1 Data and registered methods

**Table E1. Frozen eligible video counts for D and H.** Every selected video has eight distinct valid RGB frames. H TRAIN and DEV use the B/AB common-qualified cohort. Test reporting retains both the all-B-qualified and common-qualified cohorts; estimates from different cohorts are not directly subtracted. Counts are conditional on the frozen source, exact-RGB, and detection eligibility scope and do not certify complete manipulated-video ancestry independence.

| Route/status | Role | Real | Fake |
| --- | --- | ---: | ---: |
| D / accepted | TRAIN | 611 | 12,232 |
| D / accepted | DEV | 108 | 2,175 |
| D / accepted | SOURCE_CAL | 140 | 0 |
| D / accepted | SOURCE_TEST (FF) | 140 | 2,884 |
| D / accepted | TARGET_FULL (CDF) | 177 | 12,876 |
| D / accepted proxy scope | TARGET_CAL (CDF) | 80 | 0 |
| D / accepted proxy scope | TARGET_TEST (CDF) | 97 | 4,111 |
| H / data frozen | TRAIN (B/AB common) | 611 | 2,442 |
| H / data frozen | DEV (B/AB common) | 107 | 428 |
| H / data frozen | SOURCE_CAL (B) | 140 | 0 |
| H / data frozen | SOURCE_TEST (FF++, all B-qualified) | 140 | 560 |
| H / data frozen | SOURCE_TEST (FF++, common-qualified) | 139 | 557 |
| H / data frozen | TARGET_TEST (DFD, all B-qualified) | 293 | 3,027 |
| H / data frozen | TARGET_TEST (DFD, common-qualified) | 259 | 2,831 |

D retains FS9: fsgan, faceswap, simswap, inswap, blendface, uniface, mobileswap, e4s, facedancer; and FR12: fomm, facevid2vid, wav2lip, MRAA, one_shot_free, pirender, tpsm, lia, danet, sadtalker, mcnet, hyperreenact. It excludes e4e, other facial-editing methods, and entire-face synthesis. This is a joint 21-method source-training design, not an unseen-method DF40 protocol.

H uses Deepfakes, Face2Face, FaceSwap, and NeuralTextures at c23 compression. Its 560 official-validation fake candidates are unused for training or recipe choice but remain in the full 8,431-video graph for complete conflict auditing. Source-video bytes, mother images, derived PNGs, and masks underwent full artifact verification, followed by fresh-process validation and independent review before freezing. Table E1 reports the final frozen eligible cohorts. The fixed largest-face rule in DFD does not certify the selected face as the manipulated object; filename proxies and exact-RGB closure do not certify complete generative ancestry. Data and model qualifications are recorded separately: a data freeze alone does not authorize H GPU training.

Preprocessing coverage uses the fixed candidate pool as its denominator. Among 3,431 DFD candidates, 3,320 are B-eligible (96.7648%) and 3,090 are jointly B/AB-eligible (90.0612%); the common cohort retains 93.0723% of all-B videos. B and common coverage are 293/363 and 259/363 for real videos, versus 3,027/3,068 and 2,831/3,068 for fake videos. In FF++, common eligibility is 3,053/3,060 for TRAIN, 535/540 for DEV, and 696/700 for SOURCE_TEST. TRAIN and DEV use only common-eligible videos, and processing success does not override the administrative exclusion of SOURCE_UNUSED. Class-specific coverage and processing-history selection must remain distinct from detection metrics; coverage rates do not identify detector performance or counterfactual outcomes for missing videos.

Across all 8,431 candidates, 67,400 of 67,448 prescribed mother slots are selected. Six videos lacking eight distinct mothers account for 48 unselected slots. A and B each have 114 ineligible videos (six mother-selection failures and 108 with no-qualified-face failures); AB has 360 (six and 354). These totals include SOURCE_UNUSED. Failed videos, failed frames, and potentially overlapping failure reasons have distinct denominators. Original and A-redetection multiface-frame rates are 10,922/67,248 (16.2414%) and 146/66,745 (0.2187%), conditional on valid recorded detections. Missing metadata is reported separately from recorded failures; padding is stratified by processing history and eligibility scope. These multiface rates do not certify manipulated-face identity. Full stratified coverage and exclusions are provided in T1-H and supplementary material.

### 4.2 Models, sampling, and optimization

The visual encoders are CLIP ViT-B/16 [@radford2021clip], DINOv2 ViT-B/14 [@oquab2024dinov2], and EfficientNet-B0 [@tan2019efficientnet], with respective 512-dimensional projected, 768-dimensional CLS, and 1,280-dimensional pooled features. Each receives a newly initialized two-output linear head. All visual parameters are fine-tuned with cross-entropy (Real=0, Fake=1); the CLIP text tower is excluded. Initialization uses hash-verified local public pretrained weights, never earlier fine-tuned detectors.

**Table E2. Common recipe and backbone-specific candidates.** D microbatch 32 is source-qualified. All three H backbones passed independent AMP and FP32 source qualification at microbatch 32; both histories use the same microbatch and effective batch size of 128.

| Setting | CLIP ViT-B/16 | DINOv2 ViT-B/14 | EfficientNet-B0 |
| --- | --- | --- | --- |
| Base learning rate | 10⁻⁵ | {10⁻⁵,3×10⁻⁵,5×10⁻⁵} | 10⁻⁴ |
| Input / effective batch | 224×224 / 128 | 224×224 / 128 | 224×224 / 128 |
| Initial budget | 30 epochs | 30 epochs | 30 epochs |
| Source-qualified D microbatch | 32 | 32 | 32 |
| Source-qualified H microbatch (AMP / FP32) | 32 / 32 | 32 / 32 | 32 / 32 |
| Optimizer / weight decay | AdamW / 5×10⁻⁴ | AdamW / 5×10⁻⁴ | AdamW / 5×10⁻⁴ |
| Gradient clipping / initial AMP scale | 1.0 / 128 | 1.0 / 128 | 1.0 / 128 |
| Schedule | One-epoch linear warmup; cosine decay to 5% of base learning rate | Same | Same |

AdamW [@loshchilov2019adamw] uses the pinned PyTorch defaults for other optimizer parameters. Microbatches accumulate to effective batch 128 and stay fixed within a backbone/route and across paired H histories; this also controls EfficientNet BatchNorm exposure. The candidate microbatch sequence is 32,16,8, choosing the largest source-qualified value.

Real examples receive sampling mass 0.5; fake methods share the other 0.5 equally (D: 0.5/21; H: 0.5/4). Videos are equally weighted within method and frames within video. Canonical inputs sum all associated method contributions. Sampling is weighted with replacement, so these are expected masses rather than exact per-epoch quotas.

D draws 128·ceiling(102,225/128)=102,272 inputs per epoch, giving 799 optimizer attempts. The 47 above the canonical count are budget rounding, not appended images; all draws use the same replacement sampler. H histories define their shared training length from 24,424 common mother-frame identities, drawing 128·ceiling(24,424/128)=24,448 inputs and making 191 attempts per epoch. The additional 24 draws are likewise budget rounding, not additional mother frames. This length is unaffected by version-specific pixel duplication. A 30-epoch budget therefore contains 23,970 attempts for D and 5,730 for H; successful updates are counted separately. Per-epoch sample indices, assigned method contributions, actual method counts, and keyed augmentation values are saved.

Images are decoded as RGB, resized to 224×224 by Pillow bicubic interpolation, and augmented with horizontal flip p=0.5 and a brightness/contrast branch p=0.5, both factors sampled in [0.9,1.1]. Brightness precedes contrast. No additional random crop, rotation, blur, or JPEG augmentation is used. CLIP normalization has mean (0.48145466,0.45762756,0.40821073) and SD (0.26862954,0.26130258,0.27577711); the other encoders use mean (0.485,0.456,0.406) and SD (0.229,0.224,0.225). Evaluation uses deterministic resizing/normalization only. Differences from the earlier 31-method study therefore cannot be attributed solely to removing EFS.

### 4.3 Source-only development and formal budget

Development seeds 1007 and 1042 are disjoint from formal seeds 7,42,123,2026,3407. Source DEV is evaluated after every epoch using correct grouping and Raw L method-macro AUC. A candidate is scored by the last five epochs averaged across development seeds. H averages both histories and both seeds to choose one shared recipe; independent best recipes for B and AB are prohibited. Differences≤0.001 are ties, resolved first toward the shorter budget, then the smaller learning rate. Numerical or recovery qualification failures are excluded with reasons, without consulting target outcomes.

DINOv2 first selects its learning rate at 30 epochs. If any development seed, or either H history, has mean(epoch26–30)−mean(epoch21–25)>0.002, the selected rate is rerun from the original pretrained initialization for 45 epochs. The same rule using epoch41–45 versus epoch36–40 can trigger a fresh 60-epoch run. The learning-rate grid does not reopen, and the longer schedule is not resumed from the shorter run. Sixty epochs is the ceiling. After a numerically qualified extension is triggered, its final developed budget is retained without target-based reversion. This assesses budget sensitivity, not global convergence.

The D source-development rule freezes learning rates of 10⁻⁵ for CLIP, 10⁻⁵ for DINOv2, and 10⁻⁴ for EfficientNet, with 30-epoch endpoints for all three; DINOv2 did not trigger the 45-epoch extension. This selection uses no target outcomes. After completing its own 20 source-development models, H selected learning rates of 10⁻⁵ for CLIP and DINOv2 and 10⁻⁴ for EfficientNet using the joint rule across both histories and both development seeds. Both histories share the selected recipe, with a 30-epoch endpoint for each backbone. For the selected H DINOv2 recipe, the largest adjacent terminal five-epoch mean gain across the four development histories was 0.000376, below 0.002; the 45/60-epoch extension was therefore not triggered. The H recipe is determined from its own source evidence rather than inferred from D or target-domain performance.

**Table E3. Registered training counts.** These are planned counts, not completed models.

| Stage | D | H (both histories) | Total |
| --- | ---: | ---: | ---: |
| Standard 30-epoch development | 10 | 20 | 30 |
| Maximum additional DINO 45/60-epoch development | 4 | 8 | 12 |
| Formal Raw models, three backbones×five seeds per history | 15 | 30 | 45 |

Development models cannot be promoted into formal results. Formal recipes freeze the epoch count, draws per epoch, and attempted optimizer updates; the main checkpoint is the scheduled terminal last, with source-best only a prespecified sensitivity analysis. No additional L2-trained models are registered. Qualification tasks and any conditional FP32 reruns are separately accounted for.

Source-best sensitivity selects the earliest epoch attaining the maximum source-DEV method-macro AUC within each formal trajectory, retains all five formal seeds, and adds no training. The original plan specified the sensitivity concept; its reporting scope was finalized after D primary results were available and is restricted to four core comparisons: H10−H00 on native B inputs; L2−Raw for correctly grouped CDF videos under mean-logit aggregation; the grouping×L2 interaction under both aggregations; and FPR, TPR and pooled BA under target-real calibration at n=20 relative to source-real calibration on the fixed TEST. D retains six required roles; H uses only B/SOURCE_CAL, B/SOURCE_TEST and B/TARGET_TEST for both training histories. The applicable CAL and TEST resampling counts (200 and 2000, respectively), numerical controls, NA rules and unfavorable results are retained. Independent source-only selection may give different epochs and successful-update counts, so this additional analysis is interpreted as descriptive robustness evidence without the primary equal-update-budget interpretation.

### 4.4 Numerical pairing and reproducibility

The scheduler advances by attempted optimizer updates, including AMP skips. Loss, gradient norm, learning rate, loss scale, and success/skip indicators are retained per attempt; successful updates are counted separately. Equal epochs or attempts do not imply equal successful updates. Gradient clipping uses an FP64 norm calculation, and eight consecutive AMP skips trigger source-only numerical review, with the counter preserved across epochs and recovery. Nonfinite losses stop the run.

Before H target-score export, all five history pairs for a backbone must pass numerical-ledger review. If the histories have asymmetric AMP skips, all five pairs for that backbone will be rerun uniformly in FP32 with the same budget and source-qualified common microbatch; earlier AMP runs remain numerical-failure/sensitivity artifacts. If microbatch must decrease, it decreases uniformly across the backbone's H runs. No selective catch-up updates are added. Nonfinite FP32 losses or gradients suspend that backbone for a documented source-only correction and protocol revision.

The locked runtime includes Python 3.11.5, PyTorch 2.6.0+cu124, torchvision 0.21.0, timm 1.0.26, mpmath 1.3.0, OpenCV 4.11.0, and ONNX Runtime 1.20.1 for detection. Computation is restricted to four Tesla V100-SXM2-32GB devices (GPU0–3), with at most four single-GPU jobs and initially four data-loading workers per training job. TF32 and cuDNN benchmarking are disabled and strict deterministic algorithms are enabled [@pytorch26reproducibility]. DINOv2 retains its learned 37×37 positional grid and differentiable deterministic CPU interpolation at the 224-pixel input size.

D source qualification covers feature dimensions, full-parameter gradients, FP32 export, byte-exact continuous-versus-resumed training, and rejection of changed data, configuration, code or weight identities. Checkpoints bind those identities and store optimizer, scheduler-clock, scaler, RNG and sampling states. H passed separate source-only GPU qualification for all six combinations of three backbones and AMP or FP32. The two histories share initialization, sampling and augmentation streams, with microbatch 32 and effective batch size 128. Two continuous epochs and one epoch followed by recovery to epoch two were checked for byte equality of parameters, optimizer, scaler, RNG and Raw FP32 exports. Data acceptance and model qualification are recorded separately. These checks support reproducibility of training and recovery; their implications for cross-domain performance require the complete formal experiments.

## 5. Results

Every backbone is summarized separately over seeds 7, 42, 123, 2026 and 3407. AUC is reported on the 0–1 scale; effects and interactions are percentage points. The ± values are sample SD across five training seeds, distinct from conditional TEST confidence intervals. CAL repeats are reduced within each model before aggregation across models. source_test, target_full and the fixed RQ4 target_test remain separate; the 15 models are not pooled as an IID population.

### 5.1 Processing history: RQ1

<!-- PENDING:H_RESULTS -->

H results require independent acceptance of all formal endpoints, five-pair ledgers and effective precision, complete caches and paired analyses.

### 5.2 Feature scoring under the fixed head: RQ2

| Backbone | Representation | Aggregation | Source TEST macro AUC | CDF target_full macro AUC | CDF target_full pooled AUC |
|---|---|---|---:|---:|---:|
| CLIP ViT-B/16 | raw | mean_logit | 0.999582 ± 0.000159 | 0.291299 ± 0.036949 | 0.296297 ± 0.039619 |
| CLIP ViT-B/16 | raw | mean_probability | 0.999516 ± 0.000149 | 0.298783 ± 0.036427 | 0.303797 ± 0.038961 |
| CLIP ViT-B/16 | l2 | mean_logit | 0.999555 ± 0.000143 | 0.405114 ± 0.115057 | 0.409049 ± 0.117439 |
| CLIP ViT-B/16 | l2 | mean_probability | 0.999552 ± 0.000143 | 0.405144 ± 0.115042 | 0.409080 ± 0.117423 |
| DINOv2 ViT-B/14 | raw | mean_logit | 0.999656 ± 0.000095 | 0.707721 ± 0.080862 | 0.709934 ± 0.084645 |
| DINOv2 ViT-B/14 | raw | mean_probability | 0.999592 ± 0.000093 | 0.714017 ± 0.075237 | 0.716150 ± 0.078786 |
| DINOv2 ViT-B/14 | l2 | mean_logit | 0.999654 ± 0.000089 | 0.742525 ± 0.035583 | 0.745415 ± 0.038028 |
| DINOv2 ViT-B/14 | l2 | mean_probability | 0.999655 ± 0.000086 | 0.742532 ± 0.035569 | 0.745422 ± 0.038013 |
| EfficientNet-B0 | raw | mean_logit | 0.999415 ± 0.000094 | 0.707038 ± 0.035017 | 0.702243 ± 0.034976 |
| EfficientNet-B0 | raw | mean_probability | 0.999357 ± 0.000107 | 0.710660 ± 0.031007 | 0.708642 ± 0.030845 |
| EfficientNet-B0 | l2 | mean_logit | 0.999319 ± 0.000252 | 0.319945 ± 0.036527 | 0.325260 ± 0.036418 |
| EfficientNet-B0 | l2 | mean_probability | 0.999316 ± 0.000261 | 0.320316 ± 0.036504 | 0.325646 ± 0.036386 |

Under natural-video grouping and mean-logit aggregation, the average effect of inference-time L2 normalization depended on the backbone configuration. CDF target_full macro AUC changed from 0.291299 ± 0.036949 to 0.405114 ± 0.115057 for CLIP, and from 0.707721 ± 0.080862 to 0.742525 ± 0.035583 for DINOv2. For EfficientNet-B0, it decreased from 0.707038 ± 0.035017 to 0.319945 ± 0.036527. The corresponding pooled-AUC differences were +11.275 ± 11.244,+3.548 ± 4.734 and −37.698 ± 7.070 percentage points. An average gain does not imply improvement for every seed; low and below-0.5 outcomes were retained without reversing score direction.

### 5.3 Video grouping and normalization interaction: RQ3

These interactions concern the CDF target_full ranking cohort only.

Δ is the L2-minus-Raw pooled AUC difference; interaction is Δcollision−Δcorrect. Reversals are evaluated within the same seed at the frozen 1e-10 tolerance, rather than from the sign of an averaged effect. Effects below are percentage points.

| Backbone | Aggregation | Δ correct (pp) | Δ collision (pp) | Interaction (pp) | Within-seed reversals |
|---|---|---:|---:|---:|---:|
| CLIP ViT-B/16 | mean_logit | 11.275 ± 11.244 | 6.576 ± 9.277 | -4.699 ± 6.635 | 0/5 |
| CLIP ViT-B/16 | mean_probability | 10.528 ± 11.574 | 6.592 ± 9.508 | -3.936 ± 6.660 | 0/5 |
| DINOv2 ViT-B/14 | mean_logit | 3.548 ± 4.734 | 2.454 ± 4.719 | -1.094 ± 1.547 | 0/5 |
| DINOv2 ViT-B/14 | mean_probability | 2.927 ± 4.213 | 5.114 ± 3.661 | 2.186 ± 1.471 | 2/5 |
| EfficientNet-B0 | mean_logit | -37.698 ± 7.070 | -42.351 ± 5.722 | -4.653 ± 2.505 | 0/5 |
| EfficientNet-B0 | mean_probability | -38.300 ± 6.605 | -30.926 ± 5.820 | 7.373 ± 1.689 | 0/5 |

Collision-grouping method-macro AUC lacks the required single-method identity; all24 corresponding summary cells remain NA. The full eight-cell conditions, source controls, numerical diagnostics and per-seed values remain in the 12 complete summary tables.

With mean-logit aggregation, no within-seed sign reversal induced by grouping was observed for any backbone (0/5 each). With mean-probability aggregation, DINOv2 seeds 7 and 2026 reversed sign: effects of −1.097 and −2.039 percentage points under correct grouping became+1.167 and +1.811 under collision grouping. This condition therefore yielded 2/5 reversals even though the mean effects across training seeds were positive under both grouping rules. CLIP and EfficientNet-B0 remained at 0/5 with mean-probability aggregation. Aggregation, grouping and per-seed effects consequently need to be reported separately.

### 5.4 Real-video threshold adaptation on the fixed TEST: RQ4

The primary comparison uses mean-logit, both representations, the source-real threshold(n=140) and prespecified target-real budget(n=20). FPR, TPR and pooled BA are percentages. All four budgets are retained in the complete tables and mean-logit calibration figures. Mean-probability sensitivity is retained in the complete tables. The table reports pooled BA.

| Backbone | Representation | Calibration | TEST FPR (%) | TEST TPR (%) | TEST pooled BA (%) |
|---|---|---|---:|---:|---:|
| CLIP ViT-B/16 | raw | source_real n=140 | 100.000 ± 0.000 | 100.000 ± 0.000 | 50.000 ± 0.000 |
| CLIP ViT-B/16 | raw | target_real n=20 | 9.197 ± 3.111 | 3.613 ± 1.129 | 47.208 ± 1.199 |
| CLIP ViT-B/16 | l2 | source_real n=140 | 100.000 ± 0.000 | 100.000 ± 0.000 | 50.000 ± 0.000 |
| CLIP ViT-B/16 | l2 | target_real n=20 | 6.537 ± 2.850 | 7.372 ± 4.378 | 50.417 ± 1.934 |
| DINOv2 ViT-B/14 | raw | source_real n=140 | 100.000 ± 0.000 | 100.000 ± 0.000 | 50.000 ± 0.000 |
| DINOv2 ViT-B/14 | raw | target_real n=20 | 7.381 ± 4.103 | 29.528 ± 15.690 | 61.073 ± 7.834 |
| DINOv2 ViT-B/14 | l2 | source_real n=140 | 100.000 ± 0.000 | 100.000 ± 0.000 | 50.000 ± 0.000 |
| DINOv2 ViT-B/14 | l2 | target_real n=20 | 8.809 ± 1.958 | 36.801 ± 9.099 | 63.996 ± 4.550 |
| EfficientNet-B0 | raw | source_real n=140 | 100.000 ± 0.000 | 100.000 ± 0.000 | 50.000 ± 0.000 |
| EfficientNet-B0 | raw | target_real n=20 | 8.235 ± 1.566 | 26.496 ± 4.954 | 59.130 ± 2.099 |
| EfficientNet-B0 | l2 | source_real n=140 | 100.000 ± 0.000 | 100.000 ± 0.000 | 50.000 ± 0.000 |
| EfficientNet-B0 | l2 | target_real n=20 | 7.626 ± 1.151 | 4.420 ± 1.591 | 48.397 ± 0.667 |

On the same fixed RQ4 TEST cohort, the Raw mean-logit source-real thresholds yielded 100% FPR, 100% TPR and 50% pooled BA for all three backbones. With n=20 target-real calibration, mean FPR was 9.197%, 7.381% and 8.235%; TPR was 3.613%, 29.528% and 26.496%; and pooled BA was 47.208%, 61.073% and 59.130%, respectively. The operating points changed substantially, but these observations do not establish restored ranking or a population-level5% false-positive guarantee.

## 6. Discussion

### 6.1 Scope of the fixed-head inference intervention

For the fixed original head, removing feature radius changes the margin from r·a+b to a+b. The results do not support L2 normalization as a universal cross-domain improvement. The average CLIP gain still left target_full macro AUC below 0.5, whereas average ranking performance fell markedly for EfficientNet-B0. Because a refitted matched directional head was not added, the conclusion concerns the fixed-head inference intervention and does not establish that directional features are insufficient for arbitrary classifiers. Backbone configurations also differ in initialization and training recipe; their contrast does not isolate architecture as a causal factor.

### 6.2 Grouping, aggregation and seed-level effects

Grouping errors and nonlinear aggregation can jointly alter the direction or magnitude of an evaluated effect. DINOv2's 2/5 seed-level reversals were hidden by averaging. The result does not establish universal reversal, while a non-reversing mean does not negate the per-seed observations. Conditional intervals retain their registered source-group or proxy-closure scope. Missing method identities and numerically unreportable cells remain NA; pointwise intervals are not a simultaneous guarantee for the whole table.

### 6.3 Empirical threshold adaptation and uncertainty

Target-real calibration uses overlapping, nested finite subsets; n=80 reuses the entire 80-video pool and has only one distinct calibration set. CAL variation, training-seed SD and conditional TEST bootstrap uncertainty remain separate. The n=20 budget also does not meet the 59-independent-unit condition discussed in Methods for a nontrivial NP guarantee, and independence of the CDF ancestry proxy is uncertified. These are empirical operating-point results, without a claim of probability calibration or strict population risk control. H's matched preprocessing-history evaluation remains pending; the D results do not substitute for that causal comparison.

### 6.4 Processing history and the deployment population

The H branch separates training history from test-input history. Its primary deployment comparison evaluates H10−H00 on all B-eligible test videos and therefore changes the training history while retaining B deployment inputs. The complete 2-by-2 history matrix instead conditions on joint B/AB eligibility. Recomputing H10−H00 on the common cohort clarifies the comparison population, but directly subtracting the all-B and common-cohort effects cannot identify a counterfactual effect for missing videos.

Coverage, failure categories, multiple faces and padding distributions form part of the interpretation. Processing success is not a random sample, and a finding conditional on common eligibility does not automatically generalize to AB failures. The frozen largest-face rule makes input construction reproducible without certifying that the selected DFD face is manipulated. Accordingly, the analysis addresses the registered processing policy and its eligible population, rather than an average effect over every face or video. D and H differ in data provenance, manipulation sets and input histories. They should be interpreted separately before assessing whether their evidence is compatible, rather than pooled as independent replications of one experiment.

## 7. Limitations

### 7.1 Relationships and external validity

Natural-video identity and source ancestry are different levels of description. Recorded relationships, exact-RGB links and conservative filename proxies constrain known leakage without certifying unrecorded donor, driver, audio or person relationships. The CDF-source pool in this study carries DF40-generated manipulations and is not the original Celeb-DF benchmark protocol. DFD source proxies are likewise not a complete ancestry graph. Group-bootstrap intervals and calibration/test separation must therefore be scoped to known or proxy relationships rather than interpreted as proof of population independence. Because the research history has already involved related target domains, the controlled extension should not be described as a prospectively blinded study as a whole.

### 7.2 A finite real calibration pool and discrete operating points

An empirical FPR constraint is not a population false-positive guarantee. Under the relevant independence and threshold assumptions, the alpha=delta=0.05 order-statistic reference requires at least 59 independent units. Budgets n=10, 20 and 40 do not meet that nontrivial reference, and n=80 cannot repair uncertified ancestry independence. With 97 real test videos, FPR changes in integer false-positive increments: a 5% limit admits at most four false positives. Interpolation cannot create an attainable operating point at exactly 5%. Threshold adaptation should therefore be reported as an empirical comparison on the fixed pool, together with its conditional uncertainty.

### 7.3 Numerical behavior, training budgets and scope

A fixed attempt budget does not ensure identical numbers of successful optimizer updates. AMP overflow follows the frozen skip and scaler rules; attempts, successful updates and skips must be reported separately. Post hoc makeup steps would change the comparison.

High-precision ordering and independent recomputation can expose numerical artifacts but cannot remove limitations introduced by data selection or model specification. Unresolved ranking must remain NA. Missing registered methods in macro-bootstrap draws or inadequate connected closures must not yield fabricated intervals. The eventual interpretation is bounded by the fixed heads, three registered backbones, five formal seeds, accepted processing policy and corresponding eligible cohorts.

## 8. Conclusions

<!-- PENDING:CONCLUSIONS -->

Full-study conclusions require completed H results and final verification of both branches.

## Figures and captions

### Figure F1. Paired processing histories

![Paired processing histories](figures/paired_histories/paired_mother_processing_histories.png)

The five rows cover all H source method labels and were fixed by lexicographic (video_uid, mother_frame_index, mother_id) selection within frozen common TRAIN, without consulting deepfake model-performance scores or adding a new geometry-confidence ranking. Saved SCRFD geometry metadata are retained and frozen common eligibility is unchanged. Columns retain the full saved mother, direct B crop, A intermediate, AB crop after A redetection, and AB validity mask (white valid, black padding). Image aspect ratios are retained; saved pixels are displayed with nearest interpolation, without new face detection, alignment, cropping or retouching. Padding fractions are recomputed from the saved masks. A/B/AB byte, RGB, mask and parent-chain identities are verified. Examples may share source people or ancestry and are illustrative, not independent evidence of image quality, manipulated-face identity, or model accuracy. No target TEST performance or H GPU qualification is claimed.

### Figure F2. H input-processing coverage

![H input-processing coverage](figures/pre_result/H_preprocessing_coverage.png)

**H preprocessing coverage.** Bars show B and joint B/AB processing eligibility among all frozen candidate natural videos, by role and DFD class. Labels are numerator/denominator counts. All SOURCE_UNUSED candidates are administratively excluded from training, calibration and testing even when processing-eligible. TRAIN/DEV exposure uses the common cohort. This is input coverage, without detector accuracy or H GPU-qualification claims. Overlapping failure categories are not added.

### Figure F3. D source-development curves

![D source-development curves](figures/pre_result/D_source_development_curves.png)

**D source development curves.** Every registered development run is shown: two seeds per fixed learning-rate candidate. Points are independently accepted source DEV method-macro AUC; shared displayed limits retain every observed point. Each epoch has 799 attempted updates and 102272 sampled inputs. Successes and AMP skips are retained separately in plot data. Source DEV participates in the frozen recipe selection, and does not estimate formal TEST performance or training-seed confidence intervals.

### Figure F4. Source-TEST score distributions

![Source-TEST score distributions](figures/D_accepted_results/ECDF_source_test.png)

**ECDF.** Saved FP64 mean-logit margins (score field), correct natural-video grouping only. Source source_test and target target_full are separate figures; the target_full RQ2/RQ3 ranking cohort is not the fixed RQ4 target_test cohort and is never reconstructed as a CAL+TEST union. Real/fake colours and the five registered seed line styles remain separate in each backbone and raw/L2 panel. The ECDFs use the registered saved role tables unchanged. No CAL resampling, CAL-derived threshold, rank_score_decimal substitution, model/seed pooling or score-based example selection is applied; target_full membership need not be disjoint from the separately registered CAL and TEST roles.

### Figure F5. Scores in the full registered CDF ranking cohort

![Scores in the full registered CDF ranking cohort](figures/D_accepted_results/ECDF_target_full.png)

**ECDF.** Saved FP64 mean-logit margins (score field), correct natural-video grouping only. Source source_test and target target_full are separate figures; the target_full RQ2/RQ3 ranking cohort is not the fixed RQ4 target_test cohort and is never reconstructed as a CAL+TEST union. Real/fake colours and the five registered seed line styles remain separate in each backbone and raw/L2 panel. The ECDFs use the registered saved role tables unchanged. No CAL resampling, CAL-derived threshold, rank_score_decimal substitution, model/seed pooling or score-based example selection is applied; target_full membership need not be disjoint from the separately registered CAL and TEST roles.

### Figure F6. CLIP target-real calibration variation

![CLIP target-real calibration variation](figures/D_accepted_results/CAL_variation_clip_vit_b16.png)

**Calibration variation.** Each paired raw/L2 box shows 200 saved common nested target-real CAL repetitions for one model on its fixed target_test ID hash, at budgets 10/20/40/80. Boxes show median, quartiles and full min–max range, not TEST confidence intervals or across-training-model SD. Five seed positions remain distinct within each backbone; no 15-model estimator is formed. Labels u=raw/L2 report actual distinct calibration sets; a full 80 pool has u=1 despite 200 repeat rows. Legitimate unavailable budgets retain all200 NA rows and the seed position. No thresholds, predictions or inferential statistics are re-estimated. Accepted twelve tables are concatenated with exact fields/rows/NA and backbone identity, preserving default/source-real operating points, mean-probability sensitivity, oracle score-domain labels and original conditional CI references. ECDF target_full and RQ4 fixed target_test retain their separate roles. Training-seed mean±sampleSD, CAL spread and 2000 TEST bootstrap pointwise conditional intervals remain distinct.

### Figure F7. DINOv2 target-real calibration variation

![DINOv2 target-real calibration variation](figures/D_accepted_results/CAL_variation_dinov2_vit_b14.png)

**Calibration variation.** Each paired raw/L2 box shows 200 saved common nested target-real CAL repetitions for one model on its fixed target_test ID hash, at budgets 10/20/40/80. Boxes show median, quartiles and full min–max range, not TEST confidence intervals or across-training-model SD. Five seed positions remain distinct within each backbone; no 15-model estimator is formed. Labels u=raw/L2 report actual distinct calibration sets; a full 80 pool has u=1 despite 200 repeat rows. Legitimate unavailable budgets retain all200 NA rows and the seed position. No thresholds, predictions or inferential statistics are re-estimated. Accepted twelve tables are concatenated with exact fields/rows/NA and backbone identity, preserving default/source-real operating points, mean-probability sensitivity, oracle score-domain labels and original conditional CI references. ECDF target_full and RQ4 fixed target_test retain their separate roles. Training-seed mean±sampleSD, CAL spread and 2000 TEST bootstrap pointwise conditional intervals remain distinct.

### Figure F8. EfficientNet-B0 target-real calibration variation

![EfficientNet-B0 target-real calibration variation](figures/D_accepted_results/CAL_variation_efficientnet_b0.png)

**Calibration variation.** Each paired raw/L2 box shows 200 saved common nested target-real CAL repetitions for one model on its fixed target_test ID hash, at budgets 10/20/40/80. Boxes show median, quartiles and full min–max range, not TEST confidence intervals or across-training-model SD. Five seed positions remain distinct within each backbone; no 15-model estimator is formed. Labels u=raw/L2 report actual distinct calibration sets; a full 80 pool has u=1 despite 200 repeat rows. Legitimate unavailable budgets retain all200 NA rows and the seed position. No thresholds, predictions or inferential statistics are re-estimated. Accepted twelve tables are concatenated with exact fields/rows/NA and backbone identity, preserving default/source-real operating points, mean-probability sensitivity, oracle score-domain labels and original conditional CI references. ECDF target_full and RQ4 fixed target_test retain their separate roles. Training-seed mean±sampleSD, CAL spread and 2000 TEST bootstrap pointwise conditional intervals remain distinct.

The twelve complete D supporting tables retain their original fields, NA cells, default/source-real operating points, other calibration budgets, mean-probability sensitivity and oracle score domains; see the [table entry point](figures/D_accepted_results/tables/five_seed_metrics.csv) and companion files.

## Appendix A. Reproducibility protocol and counting conventions

### A.1 Videos, canonical RGB inputs and draw budgets

A natural-video identity includes dataset, domain, method and video ID. Real videos use the method label real and occur once in the negative pool. Source groups encode known or proxy ancestry and are not interchangeable with video UIDs or person identities. Exact-RGB links constrain observed content duplication. Eight-frame natural videos used for scoring and canonical RGB training inputs are different levels of description.

D TRAIN contains 611 real and 12,232 fake videos: 12,843 videos and 102,744 eight-frame associations. Merging 519 redundant associations within 493 cross-method RGB groups yields 102,225 canonical RGB inputs while retaining the sampling mass contributed by each method. Missing real source 281 explains 611 rather than the nominal 612; images are not replicated to fill this gap. Each epoch samples with replacement using the frozen masses and rounds the total draw budget up to a multiple of 128, giving 102,272 draws. The extra 47 is a budget-rounding difference, not a fixed set of appended images or additional unique RGB inputs.

The two H training histories share 3,053 common-eligible natural videos and 24,424 logical mother-frame slots. B and AB are two histories of the same mother frame; versions and support masks do not create additional natural videos. The per-model replacement draw budget is 24,448, with the extra 24 again arising from rounding. Failed eligibility is not repaired with another history's image or an unselected frame. SOURCE_UNUSED contributes to the complete relationship/RGB graph and never enters training, development, calibration or testing.

### A.2 Attempts, successful updates and precision

The effective batch is fixed at 128. D and H both use a microbatch of 32 with gradient accumulation to maintain an effective batch of 128. Each of the three H backbones passed source-domain GPU qualification at microbatch 32 under both AMP and FP32, giving six qualification runs. This choice comes from actual qualification runs rather than CPU tests or a data freeze. D and H have 799 and 191 optimizer attempts per epoch, respectively, or 23,970 and 5,730 over the base 30 epochs. Any registered extension uses its actual frozen epoch count.

Let A denote attempts and S AMP skips. Successful optimizer updates satisfy U=A−S. The learning-rate clock advances by attempts, whereas Adam state advances only on successful updates. Finite-loss, nonfinite-gradient events follow the frozen scaler and stopping rules. Events, subsequent recovery and cumulative counts are retained without makeup steps, seed changes, threshold reselection or test-pool changes. Final tables report A, U and S separately; equal epoch counts do not prove equal successful updates. If H triggers its registered uniform FP32 fallback, only effective registered paired runs can supply final caches; retained AMP artifacts are not substituted.

### A.3 Artifact identity and independent recomputation

Each run preserves source TRAIN/DEV manifest hashes, configuration, initialization weights, environment lock, qualification and training-code identities, together with seeds, epoch-level draw/augmentation records, optimizer, RNG, scaler and recovery state. The primary endpoint is the frozen last epoch. Source-best weights are retained separately for registered sensitivity analyses and cannot replace the primary endpoint according to target outcomes.

Seven formal inference-role caches bind their own manifests and the same original FP32 head. Head rows and biases are promoted to FP64 before subtraction. Ranking follows the registered precision rules, while operating decisions remain in the fixed FP64 threshold space. Saved scores are reread for independent AUC and integer operating-count recomputation. CDF-full differs from the fixed TEST used by calibration analyses, and CAL scores never enter TEST AUC. H all-B and common cohorts retain separate estimands and threshold provenance.

Five training seeds, 200 calibration subsets and 2,000 test bootstrap draws describe distinct variation sources. Unresolved ordering, missing methods, unavailable budgets and inadequate connected closures retain NA and its cause. Unfavorable seeds are not dropped before aggregation. A machine-readable asset index records manifests and version identities. These identities support recomputation without certifying unrecorded generative ancestry or a population false-positive guarantee.

## Appendix B. Scoring identities and numerical interpretation

### B.1 Radius, epsilon and the fixed head

For a saved frame feature h and the original two-output head, let w=W_fake−W_real and b=b_fake−b_real. Define

$$
r=\|h\|_2,\quad \rho=\max(r,\epsilon),\quad u=h/\rho,\quad q=w^\top u,\quad \epsilon=10^{-12}.
$$

Then

$$
z_R=\rho q+b,\qquad z_U=q+b.
$$

For r≥epsilon, rho=r and u is a unit direction. For r<epsilon, u need not be unit length. The implementation field zero_norm flags r<epsilon, including near-zero features rather than exact zeros alone. Writing the identity with rho preserves this boundary; replacing rho by r for every frame would be incorrect. Removing sample-specific radius while keeping the head generally does not give one common positive affine transformation of all scores.

The common-radius control c is the radius median weighted by source-TRAIN sampling mass. When c>0,

$$
z_{cu}=cq+b=c z_U+(1-c)b.
$$

Averaging these scores within each video preserves the same common positive affine relationship and hence exact ordering. The condition c>0 matters: at c=0 the control is constant, and its actual result must be retained without asserting strict order preservation. Target scores do not determine c.

### B.2 Whole-margin scaling and the two aggregations

For m frames define L(z)=mean(z_t) and P(z)=mean(sigmoid(z_t)). Correct videos have m=8; a collision group merging k methods uses all 8k frames. If z'_t=a z_t+d with a>0 and a common d for every frame, then

$$
L(z')=aL(z)+d.
$$

Video ordering is preserved. Transforming the threshold to aτ+d also preserves every video decision. The registered positive_scale control uses a=0.1 or 1 on the entire margin, including bias, before aggregating frames.

There is no corresponding general affine relationship for P. Applying sigmoid before averaging and applying it after averaging define different functions. The latter strictly preserves order in exact arithmetic, whereas the former can change ordering with the frame distribution and common scale. Because sigmoid curvature varies over its input domain, the Jensen gap P(z)−sigmoid(L(z)) has no universal fixed sign. Descriptions of radius, frame variance or Jensen gap alone do not establish the direction of a real–fake ordering change.

FP64 sigmoid endpoints and ties can break strict real-arithmetic order preservation. Reaggregation at 80/160/320 decimal digits is conditional on the saved FP64 frame margins. It cannot recover information rounded during the encoder's FP32 forward pass or earlier margin construction. Unresolved ranking remains NA under the frozen rules; a monotonicity theorem does not replace numerical acceptance.

### B.3 AUC ledgers and grouping interaction

Between two scoring conditions with identical labels, video sets and nonnegative weights, encode a fake–real pair as 0 for incorrect order, 1 for a tie and 2 for correct order. Let C_ij be the weighted pair mass transitioning from state i to state j, and M the total fake–real pair mass. Then

$$
\Delta\mathrm{AUC}=M^{-1}\sum_{i=0}^{2}\sum_{j=0}^{2}C_{ij}(j-i)/2.
$$

This is the AUC identity assigning half credit to ties. Pair mass includes multiplicity weights from repeated resampling. Method-macro summaries must retain the prescribed comparison set for each method. The code requires reconstruction error at most 1e-10. The ledger explains pair-order changes conditional on the caches; it is not a semantic causal decomposition of forgery mechanisms.

Correct grouping C and collision grouping K have different evaluation units, so their score rows cannot be paired directly. For every seed and aggregation, first define Δ_C=AUC_C,U−AUC_C,R and Δ_K=AUC_K,U−AUC_K,R, then I=Δ_K−Δ_C. Raw-to-L2 and logit-to-probability ledgers are reconstructed within each fixed grouping. Resampling I across groupings uses the registered common source–collision closure; inadequate closure support yields no ordinary paired interval.

A reversal requires two finite same-seed effects with opposite signs outside tolerance 1e-10. Effects within tolerance have sign zero; a nonfinite effect invalidates that seed comparison. The valid-seed denominator for reversal counts is distinct from reversal of the five-seed mean effect. If any registered estimate is unavailable, the complete five-seed mean and sample SD remain NA rather than becoming a valid-subset estimate.

### B.4 Empirical thresholds, ties and the oracle score space

Write n finite FP64 real calibration scores in ascending order as x_(1),…,x_(n), and let k=floor(αn) be the allowed false-positive count. With score≥τ predicting fake, the protocol boundary is

$$
\tau=\operatorname{nextafter}(x_{(n-k)},+\infty).
$$

The implementation adds a 1e-12 numerical guard to αn before floor. For frozen α=0.05 and the registered sample sizes this retains the stated integer budget. Every score tied at the boundary is excluded, so the realized count can be below k. A probability threshold may exceed one to classify all finite test scores as real; it is not clipped to [0,1]. The empirical budget is distinct from population false-positive control, whose independent-sampling assumptions are not certified by source proxies or deduplication.

Ordinary calibration and deployment operating metrics use the FP64 score field and score≥τ. AUC and pair-order ledgers retain usable rank_score values. oracle_operating_points also preferentially enumerates attainable vertices in rank_score space and labels them post-hoc diagnostics. If high-precision rank_score resolves ties in FP64 score, that oracle concerns the ranking score space; it does not prove attainability by the deployed FP64 threshold. Tables must preserve this distinction rather than substitute a high-precision oracle for a source- or target-calibrated threshold policy. All comparisons remain on the registered fixed TEST cohort.

## References

References use the accompanying [BibTeX bibliography](journal_references.bib); citation keys resolve to the existing verified sources.
