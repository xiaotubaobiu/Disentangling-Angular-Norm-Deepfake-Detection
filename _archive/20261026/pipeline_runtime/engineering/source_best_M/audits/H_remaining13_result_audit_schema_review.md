# Independent schema review: remaining13 original H LAST pair-result audits

Read-only schema/source review, not an implementation or result audit. C = original parallel pair-cache admission candidate; A = independently accepted metadata audit of C; L = original primary-statistics launch; P = producer execution completion; Q = actual accepted CPU qualification; I = saved pair identity.json; G = saved pair complete.json. No new statistics, model/weight/PNG validation, CPU lease or production/F change was performed.

## Exact provenance mapping

| Object | Shape/status and binding |
|---|---|
| C | No top-level `schema`. `status=PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS`; typed seed/models2/roles14, correct backbone, `identity`, `pinned_sha256`, list14 `fresh_ledger_invocations`, typed integer `all_source_completions_fresh=140`, no reuse/scoring/GPU/owner modification. Each ledger row has typed `fresh_source_completions=10`, correct backbone/precision, canonical accepted `reference_sha256`, literal true exact-original flag and false reuse. |
| A | `schema=INDEPENDENT_COMPLETED_H_PARALLEL_PAIR_ADMISSION_METADATA_AUDIT_V1`; `status=PASS_COMPLETED_QUALIFIED_ORIGINAL_PAIR_ADMISSION_METADATA_HASH_REBIND`. Identity key is `candidate_identity`. `canonical_shared107_pin_sha256` equals C.pinned_sha256 exactly. `bracketed_readonly_evidence_sha256` is a separate135-entry auxiliary audit map for the two accepted CLIP reports. |
| L | No top-level `schema`. New123/2026 status is `LAUNCHED_ORIGINAL_PRIMARY_H_PAIR_STATISTICS_AFTER_INDEPENDENT_CACHE_ADMISSION`; fields `pid`, `stdout`, `time_v`, `CPU_slot`, canonical `entry`/`entry_sha256`, `independent_cache_audit`/SHA, `admission_sha256`, `qualification_sha256`, repeats2000, GPUs_allowed false, original_F_unchanged true, M_statistics_started false. |
| P | No top-level `schema`. `status=ORIGINAL_PRIMARY_H_ANALYSIS_EXECUTION_COMPLETE_REQUIRES_INDEPENDENT_METRIC_BOOTSTRAP_AUDIT`. `source_qualification_sha256`, `pair_admission_sha256`, `original_analysis_complete_sha256`, `original_analysis_identity`, `pinned_sha256`, `CPU_slot`, list `fresh_ledger_invocations`. GPU_launch_attempted/primary_cache_owner_modified/M_results_generated/scientific_final_acceptance are all false by construction; these false values are expected scope boundaries. |
| I/G | I.schema is `H_PAIR_ANALYSIS_IDENTITY_V1`. G.schema is `H_PAIR_ANALYSIS_COMPLETE_V1`; status is `ANALYSIS_COMPLETE` or legitimate `ANALYSIS_COMPLETE_WITH_NUMERIC_NA`, independent_recompute `PASSED_FOR_ALL_REPORTABLE_CELLS`. G.identity exactly equals I; G.backbone/seed/effective_precision equal I. G.artifact_sha256 has the exact72-name inventory plus no extra files, freshly hash-checked before/after. |

Required equalities, with raw-file SHAs supplied/pinned externally:

- `C.identity == A.candidate_identity == P.original_analysis_identity == I == G.identity`.
- `L.independent_cache_audit_sha256 == SHA(A)`; its path fixes A. Do not compare this SHA to P.pair_admission_sha256.
- `L.admission_sha256 == P.pair_admission_sha256 == SHA(C)`; P pins the canonical C path directly.
- `L.qualification_sha256 == P.source_qualification_sha256 == SHA(Q)`; the external spec must supply canonical Q path as well as SHA, because new L has no qualification path field.
- `P.original_analysis_complete_sha256 == SHA(G)`; external root-observed terminal SHA must also match.
- `C.pinned_sha256 == A.canonical_shared107_pin_sha256` (exact keys and values). P.pinned_sha256 is exactly the same backbone base107 union `{Q_path:SHA(Q), C_path:SHA(C), producer_entry_path:SHA(entry)}` under current c081 source, normally110 entries. A itself is not inserted by the producer. Bind A separately in the result-auditor spec.
- The14 A headers form a unique key `(history,role)` and `header.complete_sha256 == C.identity.cache_completion_sha256[history+'/'+role] == SHA(current canonical cache complete.json)`. Preserve keys such as `B/B/SOURCE_CAL`; the first B is model history and the second B is input processing version.

A.role_marker_checks has exactly these11 fields in both accepted reports: `history, run_id, role, rows, accepted_videos, chunk_count, original_head_sha256, source_manifest_path, source_manifest_sha256, input_validation, complete_sha256`. It has no dimension/checkpoint-path field; those live in the SHA-bound raw marker identity. B/AB histories must use the expected two effective runs; each history's7roles retain its one original head. Role source-manifest paths/SHAs/row metadata remain bound to frozen accepted inputs. Footer metadata is not a new full image or model validation.

## Exact Q contract and producer source pins

Q.schema = `H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1`.
Q.status = `PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE`.
`allowed_backbones` is a list containing this backbone; `allowed_use` is a list containing `primary_pair_analysis`. Current CLIP Q lists exactly `['independent_role_admission','primary_pair_analysis']`. `original_role_checks` must be strict int14 and `fresh_default_source_checks` strict int140. `all_original_ledger_checks_exact_except_utc` and `independent_review_pass` are literal true; `production_GPU_allowed` is literal false.

Q also binds `candidate_report`/`candidate_report_sha256`, `independent_actual_audit`/`independent_actual_audit_sha256`, `original_source_manifest_sha256`, and `primary_entry_review`/`primary_entry_review_sha256`. For DINO/Eff, use their actually accepted Q and same-backbone actual candidate/independent qualification evidence; the present CLIP-only0ae Q is not transferable merely because code supports three backbones. This follows the existing F validate_qualification contract rather than adding a new scientific condition.

Current Q.implementation_sha256 has four keys and exact bytes:

- parallel_source.py: a6133c824913ab3bf361095b415eefc9ef75be39ab32fb49ab62e2656dc2624a
- cpu_ledger_adapter.py: c6b98535e5a28123baae595ab1f4b208f1feab43ff4018e740be94366d044287
- cpu_ledger_admission.py: bd202e85a4990f9246b96a7081815d07cf364041e5d3498f49bfe41d8779b213
- cpu_primary_analysis.py: c081c6ffeb14dc6bb0219851fd3fd11df25b28d2eed26ac4b7e06ef1e6e4c6ce

Point-check each `SHA(current F/name) == Q.implementation_sha256[name] == P.pinned_sha256[canonical F/name]`; main entry also equals L.entry_sha256 and the externally accepted producer-source SHA. The producer explicitly checks Q's first three dependencies at line68 and appends Q/C/current entry to its pin map at lines71–72. Q.original_source_manifest_sha256 is977431237b901d5ac88172f98bc2591d3cd7992c35be50546cda34ad2276a747; its20 canonical source module hashes must agree with base107/P and I.code_sha256's8 corresponding source names. Current Q.primary_entry_review SHA is67c9588c6960b9c5f29e513ae22bb8ba57d10b8e201b4bf1d190f93a74d8fc23.

Normal first execution requires42 fresh ledger rows, each typed10/exact accepted reference hash/backbone/precision/no reuse (420 ledger-contained source checks). c081 line98 permits14 only when result complete.json existed before producer launch. For the planned remaining first executions, retain strict42. Do not silently accept14 without a separately bound pre-launch already-complete terminal proof. Producer execution schemas above were read from SHA-pinned source; this review did not read or accept a completed123/2026 execution map.

## Old42 assumptions that do not generalize

- Old42 paths/seed/run IDs, golden identity/cache-marker/head/CAL-threshold hashes and launch fields `wrapper_pid/log/independent_pair_admission/qualification` do not fit new L's `pid/stdout/independent_cache_audit/qualification_sha256` shape. Use explicit schema routes and canonical external paths, not permissive alias guessing.
- Base107 is shared across seeds within one backbone; DINO/Eff replace that backbone's10 run-metadata paths/values. Reconstruct their own canonical base107; never reuse the CLIP107 map or CLIP A/Q as proof for another backbone. Auxiliary135 evidence count and old result-audit132/139 pin totals are provenance counts, not universal result invariants.
- CLIP dimension512 does not generalize: original cache_h dimensions are CLIP512,DINO768,Eff1280. Numerical saved-score kernels are dimension-independent; current raw cache markers and accepted FP32/head proof supply model metadata. A headers themselves do not include dimensions.
- Expected B/AB IDs come from final accepted registry plus frozen config for `(backbone,seed,history,effective_precision)`. Do not string-relabel another pair's I/C/A. Current final registry93d9336f... pins AMP for all3; keep it bound.
- AdmissionA producer_PIDs_exited refers to the earlier cache-admission process, not the primary statistics wrapper/child in L. P has no PID field. Bind root-observed primary wrapper/child exit and original time_v exit0 separately; do not guess child PID by adding1. The independent audit may use either free shared slot; it need not equal L.CPU_slot or A.audit_CPU_slot.
- Seed42's existing four snapshot artifact hashes and identity apply only to that pair. No other pair inherits that snapshot eligibility; compression/promotion stays separate and no audit re-encodes/resamples.
- Preserve original LAST final.pt and accepted data/cache/GPU provenance. M source_best/lease/authorization is outside this audit. Do not add global210-cache/owner-exit or fresh model/PNG/tensor revalidation requirements to an already authorized original CPU pair-result saved-score audit.

Reuse the preserved independent numeric module by exact SHA and unchanged pure kernels, not its hard-coded main/_execute_owned control path or monkey-patched globals. Source methods remain the four frozen FF methods; target method remains DeepFakeDetection; both aggregations, all-B/common cohorts, fixed real-only B SOURCE_CAL per model, shared2000 group draws/seed20261023, signed effects and NA rules remain unchanged. Legitimate numeric NA is retained; pair scope is still one training seed and not the five-seed summary. Frame-to-video/feature-dot and fresh geometry coverage reconstruction remain stated scope limitations, not additional experiments or main-plan hard gates.

## Evidence inspected

Accepted A123 add7f231dbdbac81c7554cf75f3c8ec6b4089edd8ce5f21b8b4d780fb4eb7844; A2026 9b4c16812d81d5a8d47601221c0dfbb4ef06b4571f29aed8a809f2cc4a6fc965. C123 bb06224e8efbd39794af19c89edb8d3690d9fdc78c143b941ddfcf2e0d199636; C2026 b308e4e85e690a3f8f7dd6dc700202dc76b0247f40d82134e6ee287a6dad3c36. L12303680bdcc039e9cd1beb8645440bf4a1dee6aaf752ac49bb630abf1642b8fb55; L2026f123688c16aded23a2300b7cabf8925bd7ddcdbe1a6804a0fbc416c464fb93ee. Q0ae309b3366de860cbd77f539c24a05aae3ec1180dbbd84bbbff95ed3111d7f7; producer source c081c6ffeb14dc6bb0219851fd3fd11df25b28d2eed26ac4b7e06ef1e6e4c6ce. No execution pin map, saved-score tables or bootstrap payload was re-audited for this schema review.
