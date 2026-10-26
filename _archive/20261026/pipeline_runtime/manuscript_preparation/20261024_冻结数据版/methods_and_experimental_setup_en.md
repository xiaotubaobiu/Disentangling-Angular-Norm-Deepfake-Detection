# DeepfakeDetector v2: Methods and experimental setup

*Preparation note (24 October 2026). This text contains Methods and experimental settings only. Data acceptance is complete for D and H, while the formal model matrix and analyses remain incomplete. H still requires separate source-only GPU numerical and recovery qualification. This revision updates frozen cohorts and training budgets; it reports no performance outcome or contribution claim.*

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

### 4.2 Models, sampling, and optimization

The visual encoders are CLIP ViT-B/16 [@radford2021clip], DINOv2 ViT-B/14 [@oquab2024dinov2], and EfficientNet-B0 [@tan2019efficientnet], with respective 512-dimensional projected, 768-dimensional CLS, and 1,280-dimensional pooled features. Each receives a newly initialized two-output linear head. All visual parameters are fine-tuned with cross-entropy (Real=0, Fake=1); the CLIP text tower is excluded. Initialization uses hash-verified local public pretrained weights, never earlier fine-tuned detectors.

**Table E2. Common recipe and backbone-specific candidates.** D microbatch 32 is source-qualified; H must freeze one matched microbatch per backbone through its own qualification.

| Setting | CLIP ViT-B/16 | DINOv2 ViT-B/14 | EfficientNet-B0 |
| --- | --- | --- | --- |
| Base learning rate | 10⁻⁵ | {10⁻⁵,3×10⁻⁵,5×10⁻⁵} | 10⁻⁴ |
| Input / effective batch | 224×224 / 128 | 224×224 / 128 | 224×224 / 128 |
| Initial budget | 30 epochs | 30 epochs | 30 epochs |
| Source-qualified D microbatch | 32 | 32 | 32 |
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

The D source-development rule freezes learning rates of 10⁻⁵ for CLIP, 10⁻⁵ for DINOv2, and 10⁻⁴ for EfficientNet, with 30-epoch endpoints for all three; DINOv2 did not trigger the 45-epoch extension. This selection uses no target outcomes. The H common recipe still requires its separate source-development process and is not inferred from D.

**Table E3. Registered training counts.** These are planned counts, not completed models.

| Stage | D | H (both histories) | Total |
| --- | ---: | ---: | ---: |
| Standard 30-epoch development | 10 | 20 | 30 |
| Maximum additional DINO 45/60-epoch development | 4 | 8 | 12 |
| Formal Raw models, three backbones×five seeds per history | 15 | 30 | 45 |

Development models cannot be promoted into formal results. Formal recipes freeze the epoch count, draws per epoch, and attempted optimizer updates; the main checkpoint is the scheduled terminal last, with source-best only a prespecified sensitivity analysis. No additional L2-trained models are registered. Qualification tasks and any conditional FP32 reruns are separately accounted for.

### 4.4 Numerical pairing and reproducibility

The scheduler advances by attempted optimizer updates, including AMP skips. Loss, gradient norm, learning rate, loss scale, and success/skip indicators are retained per attempt; successful updates are counted separately. Equal epochs or attempts do not imply equal successful updates. Gradient clipping uses an FP64 norm calculation, and eight consecutive AMP skips trigger source-only numerical review, with the counter preserved across epochs and recovery. Nonfinite losses stop the run.

Before H target-score export, all five history pairs for a backbone must pass numerical-ledger review. If the histories have asymmetric AMP skips, all five pairs for that backbone will be rerun uniformly in FP32 with the same budget and source-qualified common microbatch; earlier AMP runs remain numerical-failure/sensitivity artifacts. If microbatch must decrease, it decreases uniformly across the backbone's H runs. No selective catch-up updates are added. Nonfinite FP32 losses or gradients suspend that backbone for a documented source-only correction and protocol revision.

The locked runtime includes Python 3.11.5, PyTorch 2.6.0+cu124, torchvision 0.21.0, timm 1.0.26, mpmath 1.3.0, OpenCV 4.11.0, and ONNX Runtime 1.20.1 for detection. Computation is restricted to four Tesla V100-SXM2-32GB devices (GPU0–3), with at most four single-GPU jobs and initially four data-loading workers per training job. TF32 and cuDNN benchmarking are disabled and strict deterministic algorithms are enabled [@pytorch26reproducibility]. DINOv2 retains its learned 37×37 positional grid and differentiable deterministic CPU interpolation at the 224-pixel input size.

D source qualification has accepted feature dimensions, full-parameter gradients, FP32 export, byte-exact continuous-versus-resumed training, and rejection of changed data/configuration/code/weight identities. Checkpoints bind those identities and store optimizer, scheduler-clock, scaler, RNG, and sampling states. H data acceptance has passed against its frozen manifests, while its separate source-only GPU numerical and recovery qualification remains pending. H development and formal training have not started. Data acceptance and model qualification are recorded separately. No formal target-domain performance, calibration benefit, or completed 45-model study is asserted in this preparation.
