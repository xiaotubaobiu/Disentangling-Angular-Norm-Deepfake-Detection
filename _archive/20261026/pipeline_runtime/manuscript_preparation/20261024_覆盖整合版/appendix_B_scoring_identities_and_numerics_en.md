# Appendix B. Scoring identities and numerical interpretation

*Editorial note. These are algebraic explanations of the frozen scoring definitions and their implementation. They add no experiment, protocol change or detector result. Equalities first concern exact real arithmetic; cache precision and FP64 rounding are addressed separately by the numerical audits.*

## B.1 Radius, epsilon and the fixed head

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

## B.2 Whole-margin scaling and the two aggregations

For m frames define L(z)=mean(z_t) and P(z)=mean(sigmoid(z_t)). Correct videos have m=8; a collision group merging k methods uses all 8k frames. If z'_t=a z_t+d with a>0 and a common d for every frame, then

$$
L(z')=aL(z)+d.
$$

Video ordering is preserved. Transforming the threshold to aτ+d also preserves every video decision. The registered positive_scale control uses a=0.1 or 1 on the entire margin, including bias, before aggregating frames.

There is no corresponding general affine relationship for P. Applying sigmoid before averaging and applying it after averaging define different functions. The latter strictly preserves order in exact arithmetic, whereas the former can change ordering with the frame distribution and common scale. Because sigmoid curvature varies over its input domain, the Jensen gap P(z)−sigmoid(L(z)) has no universal fixed sign. Descriptions of radius, frame variance or Jensen gap alone do not establish the direction of a real–fake ordering change.

FP64 sigmoid endpoints and ties can break strict real-arithmetic order preservation. Reaggregation at 80/160/320 decimal digits is conditional on the saved FP64 frame margins. It cannot recover information rounded during the encoder's FP32 forward pass or earlier margin construction. Unresolved ranking remains NA under the frozen rules; a monotonicity theorem does not replace numerical acceptance.

## B.3 AUC ledgers and grouping interaction

Between two scoring conditions with identical labels, video sets and nonnegative weights, encode a fake–real pair as 0 for incorrect order, 1 for a tie and 2 for correct order. Let C_ij be the weighted pair mass transitioning from state i to state j, and M the total fake–real pair mass. Then

$$
\Delta\mathrm{AUC}=M^{-1}\sum_{i=0}^{2}\sum_{j=0}^{2}C_{ij}(j-i)/2.
$$

This is the AUC identity assigning half credit to ties. Pair mass includes multiplicity weights from repeated resampling. Method-macro summaries must retain the prescribed comparison set for each method. The code requires reconstruction error at most 1e-10. The ledger explains pair-order changes conditional on the caches; it is not a semantic causal decomposition of forgery mechanisms.

Correct grouping C and collision grouping K have different evaluation units, so their score rows cannot be paired directly. For every seed and aggregation, first define Δ_C=AUC_C,U−AUC_C,R and Δ_K=AUC_K,U−AUC_K,R, then I=Δ_K−Δ_C. Raw-to-L2 and logit-to-probability ledgers are reconstructed within each fixed grouping. Resampling I across groupings uses the registered common source–collision closure; inadequate closure support yields no ordinary paired interval.

A reversal requires two finite same-seed effects with opposite signs outside tolerance 1e-10. Effects within tolerance have sign zero; a nonfinite effect invalidates that seed comparison. The valid-seed denominator for reversal counts is distinct from reversal of the five-seed mean effect. If any registered estimate is unavailable, the complete five-seed mean and sample SD remain NA rather than becoming a valid-subset estimate.

## B.4 Empirical thresholds, ties and the oracle score space

Write n finite FP64 real calibration scores in ascending order as x_(1),…,x_(n), and let k=floor(αn) be the allowed false-positive count. With score≥τ predicting fake, the protocol boundary is

$$
\tau=\operatorname{nextafter}(x_{(n-k)},+\infty).
$$

The implementation adds a 1e-12 numerical guard to αn before floor. For frozen α=0.05 and the registered sample sizes this retains the stated integer budget. Every score tied at the boundary is excluded, so the realized count can be below k. A probability threshold may exceed one to classify all finite test scores as real; it is not clipped to [0,1]. The empirical budget is distinct from population false-positive control, whose independent-sampling assumptions are not certified by source proxies or deduplication.

Ordinary calibration and deployment operating metrics use the FP64 score field and score≥τ. AUC and pair-order ledgers retain usable rank_score values. oracle_operating_points also preferentially enumerates attainable vertices in rank_score space and labels them post-hoc diagnostics. If high-precision rank_score resolves ties in FP64 score, that oracle concerns the ranking score space; it does not prove attainability by the deployed FP64 threshold. Tables must preserve this distinction rather than substitute a high-precision oracle for a source- or target-calibrated threshold policy. All comparisons remain on the registered fixed TEST cohort.
