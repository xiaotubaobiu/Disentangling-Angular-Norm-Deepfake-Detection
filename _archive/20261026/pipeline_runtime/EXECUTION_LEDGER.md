# Execution ledger

2026-10-23T08:57:12.228387+00:00

P1 COMPLETE for D/source runtime; P2 D development launched. H geometry qualified, fullpixels running, Htraining pending.

# Deepfake Detector v2 execution handoff

Status: P2 D source development running; H full preprocessing running. The full experiment and journal manuscript task is NOT complete. No formal scientific results exist yet.

Binding plan: /home/z/obsidian/paper/deepfake-detector/新实验设计/20261023/新实验计划_本机适配版.md
Plan SHA256 b018b01915a62c645fc3718134970cd49dc0651ecfd02792220f3a0086c6594d.
New artifact root: /home/z/workspace/deepfake_pipeline_reliability_v2_20261023
New code: /home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2
Git branch codex/pipeline-reliability-plan-20261023, local implementation commit8c4651b313fe384e20043351bd39da3c0c279758 (not pushed).
Frozen training hash336f3097434edf903a5ff402162e29afeca2442ca50f9c7d9883ca60136b8335.
Independent environment: artifactroot/environment/bin/python, exact clone of prior locked runtime plus pinned official PyPI onnxruntime1.20.1; requirements.execution.lock.txt and pip check passed.

## Useful results / counts

D full original42-index rebuild:1,034,481 associations, current stat agreement with audited bytes/RGB metadata for every path. Metadata freshness does not prove fresh cryptographic rehash of all images. The graph is built on full associations before fixed8 and canonical merging.
D TRAIN611real+12232fake; DEV108+2175; SOURCE_CAL140real; SOURCE_TEST140+2884; TARGET_FULL177+12876. Every selected natural video has8uniqueRGB. Train102225 canonical RGB,102744 associations,102272 draws/epoch (=799attempts). All21methods survive. Real mass0.5; eachfake0.5/21; method_contributions maps preserve merged contributions.
Entirely unselected videos391:290source-role mismatches,8unknownSadTalker,93insufficientRGB. Association-level rejected aliases are not all removed videos. Official source281 absent from original indices explains611vs612 realTRAIN.
D freeze SHA6f30a13cc60778b77653493f78be0535a928879ec93226d139f724eab3c23a67. All21artifact hashes and role/RGB/source-graph invariants independently verified.
CDF calibration scope:80realCAL,97realTEST,4111fakeTEST. Full RGB/name-proxy closure excluded21 more fakes than historical4132. PROXY_SCOPE_EMPIRICAL_ONLY; complete ancestor independence and strict NP population guarantees unavailable.
H metadata matches plan TRAIN612/2448, DEV108/432, CAL140real, SOURCE_TEST140/560, DFD363/3068. 560officialvalfakes remainunused. Both D/H share exact360pair306/54 partition.

## Runtime qualifications / evidence

91new CPU tests passed9.69s logs/p1_final_all_tests.log. Existing old67tests are not counted as new evidence.
All3backbones qualifiedmicro32/effective128 with fullparametergradients, expected512/768/1280dimensions, FP32export and100attempts timing. Latest qualification/gpu_gate.json matches frozen traininghash.
Actual weighted-sampling/keyed-augmentation/4worker2epoch vs1+resume comparison passed for all3, byteexact sampling, methodcounts, augmentationvalues, decodedbatches, parameters, optimizer, scaler, Python/NumPy/Torch/CUDA RNG and FP32features; all4 identity changes rejected. logs/pipeline_resume_complete.json and pipeline_resume_verified.json (57artifacts) preserve evidence/checkpoints.
Final saved sourceDEV2283video scores and realCAL140 scores independently recomputed; methodmacroAUC error0 and7FP/140minimal nextafter thresholds. These are qualification, not scientific results. qualification/source_scores_independent_recompute.json.
Independent review reports/training_independent_review.json covers important numerical/recovery/ownership findings; final minor corrections (persistent skipcounter/registry status) tested afterward. Source-best state is stored in checkpoint for recovery; history is repaired from committed checkpoint.
Source H no-upscale geometry accepted10/10 videos (80motherframes), repeated240A/B/ABoutputs/masks/matrices byte-identical. qualification/H_geometry_frozen.json, reports/h_geometry_source_acceptance.json.
Workspace free448.5GiB >200GiBestimatedworkingset+50GiBreserve. Dataset free2898.3GiB. H A/B/AB+mask projection14.2GB excludes full-resolution motherPNG/manifests; actual storage must be measured from progress.

## Live jobs

D queue ownerPID3942476, launched2026-10-23T08:53:30Z. Dlaunch receipt reports/D_launch_receipt.json. queue_status.json and D_PROGRESS.md show activechildren; logs/D_queue_supervisor.log and logs/D_dev_*.log show progress.
Queue performs10D30epochdevelopmentmodels, source-onlyDINO fresh45/60promotion if registeredruletriggers, frozen15Dformal models, then formal-onlyFP32caches with7roles. It then stops ANALYSIS_PENDING. Models/config/arrays/logs stayunderroot/runs. Formalregistry has45models; H30 remainunstarted.
Immutable D gate qualification/D_gate.json SHAc2a9a09d208d2671b7a123df7ee29e83b8abedcac08d9221d5ad26f3948700aa. D configs/execution.json belongs ONLYtoD and must not be changed byH. Never modify trainer/models while D runs; changed codehash correctly rejects recovery/extraction. Future H uses its own gate/config.
H preprocessing ownerPID3915880,4CPUworkers xORT2threads, CUDAempty. data/H/H_preprocessing_progress.json and logs/h_geometry_full_preprocessing.log, reports/h_geometry_handoff.json. All8431videos includingunusedvalfakes processed for audit. Source geometry frozen before any targetdecode. Hdata remains UNFROZEN until full closure/coverage audit. Images at /Dataset/deepfake_detection/derived/pipeline_reliability_v2_20261023/H. Initial ETA4-5hours, updatefromactualprogress.

## Remaining work (do not claim complete)

1. Confirm D first full epochs/checkpoints and update measuredwalltime estimate. HistoricalfourGPU estimate~75h excluded preprocessing/DEV/cache/possibleextensions; recent100stepGPUpure timing is only a lower bound.
2. Independently audit Hdatafreeze_candidate after fullprep, strictsamecommontrainingmother IDs/weights/order and allBdeployment vs common2x2cohorts. Implement and test Hsource qualifications and shared recipe selection (20development), fivepairednumericledgers/AMPfallback and30formalmodels. Htrainingorchestration is NOT implemented yet; avoid treating geometrycompletion as Hmodelcompletion.
3. Danalysis driver still needs integration using tested cache_d/scoring_d APIs; preserve allrawfeature/head identities, eightcells, highprecision controls,3x3ledgers, source/collisionclosure2000bootstrap, calibration200commonnestedsets. Existing reports/d_scoring* are SYNTHETIC_ONLY, not experimental results.
4. H formalFP32caches/2testcohorts and uncertainty tables. H asymmetricAMPskips: beforetargetexport uniformFP32rerunall5pairs ofthatbackbone perplan, do notselectivelypatchsteps.
5. Produce T1-T8/newfigures, independentfinalmetricrecompute and CN/ENjournalpapers afterallnewresults. Existing masters /home/z/obsidian/paper/deepfake-detector/deepfakedetector-10.10.md and 英文稿.md notyetmodified; inspectactualmasters and backupbeforewriting. Use verified参考文献 assets, no inventedcitations/numbers. Fullpaper task remainspending, scientificwriting notrunreport.

## Safety / rulings

User authorized allsteps of latestplan; no renewedapproval needed for safe specifiedwork. OnlyGPU0-3, max4singleGPUjobs globally; bothqueue/workercheckUUID,currentcomputeapps,memory and lifetimeGPUlock. Run/extract locks preventduplicates. Do nottouchGPU4-7VLLM orJevfiles/processes. Never runlegacyrun.pyexecute/LLMcontainerlogic.
One qualification launch was denied byautomaticreview because an upfrontfreshGPUcheck was missing while olderGPU1evidence lookedbusy. Deniedactiondidnotexecute. Resolvedoncewithnew2026-10-23T08:43:35Zfreshidle/app/PIDevidence, supervisorupfrontcheck+workerlifetimeguards, andsuccessfulapprovedrelaunchPID3929862. No pendingpermissionblock.
Source-onlyHfeasibilityruling: fixed640letterboxscale=min(1,640/w,640/h), no-upscale, after systematicABredetectionfailurewith256->640enlargement. Originalfailurespreserved; identicalruleallroles/domains.
Numericalruling:8consecutiveAMPskips triggerssource-onlyreview, persistsacrossepoch/resume. FP64normforclip1.0 preventsfinitegradientnormoverflow. Neither decision consultedtargetmetrics.
Do notpermanentlydelete data oroldresearch; no unknownsoftware/paidservices; no oldfine-tunedweights/predictions reused asnewresults. Mainpaperclaims mustrespectunknownCDF/DFDancestry/face-selectionscope and allnegative/missingoutcomes.


## Update:2026-10-23 09:06UTC real throughput and count reconciliation

User additionally authorized autonomous use of all4allowedGPUs toshortenruntime, withoutrestartingnormaljobs orchangingfrozenprotocol. GPU0/1CLIPseed1007/1042; GPU2/3DINOseed1007/1042. Snapshotutilization97/98/78/85percent. Firstepochall4completed799successfulattempts,0AMPskips. CLIP512.87/520.06secondsperfullepoch, DINO648.80/655.52seconds (includesdrawgeneration,trainingandfullsourceDEV; checkpointwritefollows). Approx198/157inputspersecond end-to-end. CPUbusy24.9percent,IOwait0.084percent;Dataset~2.3MiB/sread+23.6MiB/swrite,~14.8percentIObusy over2s. CurrentbottleneckGPUcompute, noaddedGPUprocesses orworkerchanges warranted. Allnormaljobscontinue. NeedEfficientNet/Hactualepochs beforefirmall-projectETA; old75hremainsahistoricalestimate, notpromise.

Countidentity:12843videos×8=102744associationrows.493exactRGBgroupsspanmultiplefakemethods,0single-methodcollisiongroups;canonicalmergingremoves519extraassociationrows to102225uniqueRGBs. Examplesinswap|simswap376extrarows,fsgan|mobileswap74,others69. Nomissingframesamongselectedvideos;93insufficient8RGBvideoswerealreadyexcludedbeforethesevideocounts. ceil102225/128×128=102272draws=799attempts;difference47isbudgetrounding,not47appendedfiles. All102272drawwithreplacement. Same-seedactiveCLIP/DINOtrainmanifest,samplearray,methodarray,augmentationarraySHAsexactmatch. Evidence reports/D_training_count_reconciliation.json, D_RGB_duplicate_groups.parquet,D_RGB_duplicate_method_sets.csv and performance_snapshot.json.

## Pending continuation authorization / important H final audit

Creating hourly persistentAIcontinuationautomation named Deepfakev2实验与论文续接 was DENIED byautomaticapprovalreview: ongoingjoblaunchesandpapereditsneedexplicitpersistentautomationauthorization. Noautomationwascreated; noalternativecron/watcherwascreated. Auserapprovalquestionispending. Thisdoesnotstopthealready-authorizedfiniteDqueueorHpreprocessing. Parentmustmanuallyresume/checkthisthreadifapprovalnotprovided; donotclaimhourlyfollowupisset.

IndependentHreview found2importantauditissues (reports/h_independent_review.json): SOURCE_UNUSEDvalfakeRGBedgeswereomittedfromfullgraph, andresumeacceptedmissing/corruptedderivedPNGs/masks. Agenth_geometryisfixingaudit/resumevalidationwithtests; pixelprotocolandfrozentrainerunchanged. RunningHowner3915880alreadyimportedoldmodule: continuepixeljob, thenMANDATORYfreshre-finalizationusingupdatedgeometry_h audit-existing andH_processed_manifest_paths.json. DoNOTfreezeHdatafromoldautomaticfinalizeoutput. Requireaudit_schema=FULL_CANDIDATE_GRAPH_INCLUDING_SOURCE_UNUSED_V2, freshH_full_candidate_audit_receipt.json andfullimageintegrityPASSbeforeHsource/modelgate. NoHtrainingisyetallowed. Explicitfreshcommandwillbeinh_geometryauditfixhandoff. Allpixeloutputsremainreusableafterintegrityvalidationbecausegeometryprotocolhash85a865f...ae03unchanged.


## Final session handoff: H audit fixes verified

H audit/integrity fixes are complete and committed locally asae7c24feaf353f91efa671ee79b76492d3f70f52 after99fullv2testsPASS12.13seconds (logs/final_auditfix_all_tests.log). Trainingcodebundle remains336f3097434edf903a5ff402162e29afeca2442ca50f9c7d9883ca60136b8335; runningDjobs andimmutableDgateunchanged. Pixel/detector/geometry/protocol/preprocessfunctionhashesunchanged; Hpixeljobwasnotrestarted. reports/h_geometry_fresh_audit_handoff.json containsverifiedtest/smokeevidence.

AfterHpixeljobfinishes, aFRESHprocessMUSTrun:
cd /home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/environment/bin/python -m reliability_v2.geometry_h audit-existing --manifest-paths /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/data/H/H_processed_manifest_paths.json --artifacts /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/data/H
RequirefreshH_full_candidate_audit_receipt.json schemaFULL_CANDIDATE_GRAPH_INCLUDING_SOURCE_UNUSED_V2,candidate8431,artifact_integrity_statusPASSED,integrityreceipt+alloutputparquethashesbound. Reviewanyquarantines/remainingconflicts andindependentlyverifyselectedcohorts beforegrantingHdata/modelgate. Oldrunningautomaticfinalizerisinsufficientbecauseprocessloadedoldcode. NoHtrainingorformalresultclaimuntilthisreview.

Thefullstudyremainsunfinished: D10standarddevelopmentthenconditionalDINOextensions+15formal+cachesrunningautomatically; H20development+30formalorchestrationstilltobeimplemented/qualified, plusD/Hanalysisdriver/fullstatistics/figuresandCN/ENjournalpapers. ExistingmasterpapershaveNOTbeenoverwritten. Noformalnewmodelresultshavebeenclaimed. Parentcanresumeallremainingauthorizedresearchfromthishandoff; persistenthourlyAIautomationwasDENIEDandnotcreated,pendingexplicituserapprovalquestion. Noalternativepersistentmonitorwascreated.


## 2026-10-23T09:45:32.502655+00:00 analysis/methods round

114CPU testsPASS; independentreviewPASS; localcommitb2a47da0a4ed84e3c946205c768754feb13cd72b. Ddev0/10complete,4active; Hpixels1780/8431. Formalcaches0/no scientificresults. CN/ENmethods+34claimmap saved. Hourlyauthorizationreceived; one retry after0configs failedunsupportedcloudthread. No taskcreated, parentmanualcontinuation,nextsuggested10:30UTC.


## 2026-10-23T10:38:19.600241+00:00 — 10:30healthcheck

PASS D0/10complete,4active,6pending; checkpointepochs11/11/9/9,31960committedupdates,0skips. GPUmean95.3/92.8/82.4/80.2percent,throughput198.6/194.6/157.1/155.5inputs/s. H2830/8431,ETA3.91h;partialallfailuresnoqualifiedface. Capacityhealthy. No repairs/restarts/configchanges/automationretry. Nextmanualcheck11:30UTC; Hqualification/queue implementation remains parentcoordination. See reports/check_20261023T1030Z.json.
