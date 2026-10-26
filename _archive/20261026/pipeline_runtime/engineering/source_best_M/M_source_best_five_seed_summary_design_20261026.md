# M source-best five-seed summary design

Scope fixed by frozen plan b018b019 §9 and delegated four-core M contract. One backbone per request, exactly D/H x seeds7,42,123,2026,3407. No LAST/DEV/F/common cohorts, model/source revalidation, training, bootstrap regeneration or production run.

## Approach
Reuse accepted single-job audit72f28181 and its read-only authenticated artifacts, plus H6cd3bff4/D a7b07996. Do not create another numeric oracle. A pure endpoint extractor and independent Fraction/statistics five-seed calculation are the only new math. Existing route summaries cannot consume the source-best D/H accepted audit contract, so a separate two-file tool is the smallest isolated change.

## Interfaces
- main([]) returns PREPARE_ONLY with request/schema/dependency requirements, no slot/output.
- execute_summary(request_path,request_sha256,slot) requires exact10 audit refs plus original audit-request refs, job/selected-prefix declarations, one registered backbone, current fixed dependencies, explicit output identity and SHA-bound resource budget.
- Each accepted report must be SELECTED_CORE_METADATA_AND_NUMERIC_ACCEPTED, accepted_for_selected_core true, scientific_final_acceptance false, current consumed report/request/producer/code/result pins and all six role digests fixed. Non-consumed original source weight/image pins remain historical accepted evidence and are not freshly rehashed. Do not call metadata_chain/audit_numeric or load tensors. Stream only authenticated digests and read accepted metrics/bootpoint.
- Extract RQ1 H10-H00 all_B targetDFD/sourceFF, both original thresholds and both aggs, original7 metrics; RQ2 target_full correct mean-logit pooledAUC L2-Raw; RQ3 target_full interaction each agg; RQ4 n20 target-real minus source-real fpr/tpr/pooledBA for raw/l2 and both aggs.71 endpoint rows perseed,355 perbackbone.
- per_seed_core.csv preserves negative/NA, prefix JSON and fixed logical CI/CAL references. five_seed_core.csv has exact registered n5, finite count, mean/sampleSD(ddof1) only if all5 finite. Missing audit/seed is an error; numeric NA makes aggregate unavailable,never a4-seed average. No mean bootstrap,p-values,CI endpoint averaging or15-IID pooling.
- CI references remain perseed TEST2000 conditional. RQ3 interaction CI uses only saved common closure; gated/no-valid/numeric NA status and fixed source reference retained. RQ2 uses only the explicitly labelled downstream arithmetic on aligned frozen normal draws described below, never closure substitution. RQ4 CAL200 subset distributions are separate references. No claim that the original producer saved a normal Raw/L2 difference CI.
- One append-only identity namespace; small bounded CSVs+manifest published only after all pins/math rechecks. Original results read-only. Partial publication preserved/refused,completed identical publication may be reused after rechecks.

## Runtime
Actual execution requires original shared CPU slot0/1 for lifetime,emptyCUDA,threads1,address-space/peak bounds and external SHA-bound summary resource budget referencing fresh joint remaining plan and50GiB floor. Current engineering work only new small synthetic tests,threads1,CPU,no real slot or output. No permission issuer/scheduler.

RQ2 correction accepted by root: normal Raw/L2 frozen draws must align exactly (groups/order, integer multiplicities, seed/repeats/repeat index) before downstream per-draw subtraction and type-7 pointwise percentile CI. Original group gate and numeric NA remain unavailable. Closure CI never substitutes RQ2. Label: downstream arithmetic from aligned frozen normal draws. No new random draw.
