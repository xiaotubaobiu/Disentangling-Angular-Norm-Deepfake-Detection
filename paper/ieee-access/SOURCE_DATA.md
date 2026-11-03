# Manuscript source data

`source_data/tables/` contains the accepted33-model publication tables. D consists of15 retained scoring models; H consists of six retained CLIP models and twelve newly trained DINOv2/EfficientNet models. All H backbones have a ten-epoch cap. Mean/SD values describe training-seed variation on the frozen video cohorts.

`source_data/provenance/` records table and manuscript acceptance, figure inputs, source-root mappings, and the explicit fixed-budget revisions after the initial H results. Original larger-budget H outputs remain in the separate immutable evidence archive and are excluded from current main results. D CSV bytes are unchanged from their original accepted versions.

`source_data/development/` provides the observed source/CDF development epochs and selected maxima used for the development-curve figure. These development trajectories are distinct from the formal-test results.

The source data include summaries and verification metadata. Original media, checkpoints, features, per-video scores, and full processing ledgers are retained in the experiment evidence roots identified by the provenance records. The Overleaf source bundle contains the document dependencies only; no experiment environment or model files are required to compile it.
