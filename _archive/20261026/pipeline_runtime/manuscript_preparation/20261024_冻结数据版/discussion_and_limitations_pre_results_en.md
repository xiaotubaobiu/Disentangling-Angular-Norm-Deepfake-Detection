# Discussion and limitations: preparation before completed results

*This draft follows the frozen methods and contains no performance observations from unfinished models. Directions, effect sizes and contribution claims require all registered analyses and independent recomputation. The interpretation principles below do not select which results are retained: all registered controls, including null and negative findings, must be reported.*

## 6. Discussion

### 6.1 Representation and aggregation under a fixed head

The D branch compares feature scale, video identity and aggregation using the same feature caches. Its L2 condition retains the classifier head and bias learned during Raw training. It therefore estimates the response of an existing detector to this inference intervention. It does not estimate the performance attainable by retraining a normalized model, nor does it fully separate feature direction from compatibility with the original head. The registered comparison under correct grouping and mean-logit should precede interpretation of other aggregation conditions and numerical diagnostics.

The deliberately incorrect grouping omits the manipulation-method identity and changes both the scoring unit and the number of frames per group. A grouping-by-normalization interaction can test whether an evaluation identity error changes the conclusion, but effects under the two groupings are not simple replications on the same natural-video population. Method-macro AUC is undefined for collision groups; discarded method labels cannot be reassigned to manufacture this estimand. The transition ledger decomposes AUC differences into changes in real–fake pair ordering. This provides an arithmetic account of scoring changes, rather than a causal account of generation or learning mechanisms. A sigmoid applied after mean-logit must also remain distinct from mean-probability, so numerical endpoints and aggregation nonlinearity are not directly equated with detector capability.

### 6.2 Processing history and the deployment population

The H branch separates training history from test-input history. Its primary deployment comparison evaluates H10−H00 on all B-eligible test videos and therefore changes the training history while retaining B deployment inputs. The complete 2-by-2 history matrix instead conditions on joint B/AB eligibility. Recomputing H10−H00 on the common cohort clarifies the comparison population, but directly subtracting the all-B and common-cohort effects cannot identify a counterfactual effect for missing videos.

Coverage, failure categories, multiple faces and padding distributions form part of the interpretation. Processing success is not a random sample, and a finding conditional on common eligibility does not automatically generalize to AB failures. The frozen largest-face rule makes input construction reproducible without certifying that the selected DFD face is manipulated. Accordingly, the analysis addresses the registered processing policy and its eligible population, rather than an average effect over every face or video. D and H differ in data provenance, manipulation sets and input histories. They should be interpreted separately before assessing whether their evidence is compatible, rather than pooled as independent replications of one experiment.

### 6.3 Threshold adaptation and distinct uncertainty sources

Real-only threshold adaptation separates ranking from a deployment operating point. Each flow obtains a threshold from its own calibration scores, while the corresponding comparisons share calibration IDs and a fixed test cohort. Any eventual improvement must therefore be described jointly by FPR, TPR, balanced accuracy and the integer false-positive count. A change in operating performance does not establish conditional-probability calibration, and an oracle using test labels is not a deployable procedure.

Five training seeds, 200 calibration subsets and 2,000 test bootstrap draws represent different sources of variation. Test intervals condition on the fixed model and recorded relationships; calibration-subset spread is reported separately, and the sample SD across seeds does not replace either quantity. At n=80, the entire 80-video calibration pool provides only one distinct subset. Treating repeated draws as additional independent observations, or interpreting a count of same-direction seeds as significance, would exceed the design.

## 7. Limitations

### 7.1 Relationships and external validity

Natural-video identity and source ancestry are different levels of description. Recorded relationships, exact-RGB links and conservative filename proxies constrain known leakage without certifying unrecorded donor, driver, audio or person relationships. The CDF-source pool in this study carries DF40-generated manipulations and is not the original Celeb-DF benchmark protocol. DFD source proxies are likewise not a complete ancestry graph. Group-bootstrap intervals and calibration/test separation must therefore be scoped to known or proxy relationships rather than interpreted as proof of population independence. Because the research history has already involved related target domains, the controlled extension should not be described as a prospectively blinded study as a whole.

### 7.2 A finite real calibration pool and discrete operating points

An empirical FPR constraint is not a population false-positive guarantee. Under the relevant independence and threshold assumptions, the alpha=delta=0.05 order-statistic reference requires at least 59 independent units. Budgets n=10, 20 and 40 do not meet that nontrivial reference, and n=80 cannot repair uncertified ancestry independence. With 97 real test videos, FPR changes in integer false-positive increments: a 5% limit admits at most four false positives. Interpolation cannot create an attainable operating point at exactly 5%. Threshold adaptation should therefore be reported as an empirical comparison on the fixed pool, together with its conditional uncertainty.

### 7.3 Numerical behavior, training budgets and scope

A fixed attempt budget does not ensure identical numbers of successful optimizer updates. AMP overflow follows the frozen skip and scaler rules; attempts, successful updates and skips must be reported separately. Post hoc makeup steps would change the comparison. H still requires the planned paired numerical qualification, any necessary uniform precision fallback, and registration of the effective models. CPU implementation tests or accepted data cannot substitute for actual GPU qualification.

High-precision ordering and independent recomputation can expose numerical artifacts but cannot remove limitations introduced by data selection or model specification. Unresolved ranking must remain NA. Missing registered methods in macro-bootstrap draws or inadequate connected closures must not yield fabricated intervals. The eventual interpretation is bounded by the fixed heads, three registered backbones, five formal seeds, accepted processing policy and corresponding eligible cohorts. The existence, stability and direction of any effect must be established by completed results, rather than by these preparatory paragraphs.
