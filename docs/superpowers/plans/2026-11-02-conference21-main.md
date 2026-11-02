# Conference training / DF40 augmentation / 21-method main implementation plan

User specification: 2026-11-02 request (repeated three times); implementation and GitHub consolidation explicitly authorized.

Goal: preserve conference-original exactly; consolidate eight other histories into main; implement the requested combined protocol without starting training.

Architecture: retain latest pipeline tree and all eight ancestor histories, tag every old branch tip before removing branch refs. Add studies/conference21 as the default active package; retain previous scientific implementations and results as historical evidence. Reuse v2 D manifests, sampling, eight-cell scoring, calibration and uncertainty code. Do not combine unrelated JEV runtime files into detector imports.

Decisions: three backbones x seeds 7/42/123/2026/3407. Full visual fine-tuning, Raw CE, AdamW WD5e-4, actual per-GPU batch128 (no silent accumulation fallback), LRs 1e-5/5e-5/1e-4, 15 maximum epochs, epoch-level conference scheduler, parameter-only EMA .999 with live buffers as in conference code. Update EMA after successful optimizer updates; AMP-skipped updates do not advance EMA. Source DEV pooled natural-video mean-probability AUC selects EMA best_model.pth; five consecutive non-improving epochs stop. Evaluation accepts only that selected checkpoint. No target-based selection.

Use official OpenAI CLIP B/16 projection512, official Facebook DINOv2 B/14 CLS768, torchvision ImageNet EfficientNet B0 pooled1280. Official DF40 Albumentations1.3.1 transformation implementation and probabilities, OpenCV cubic224 input, then each backbone's pretrained normalization. Do not adopt DF40 CLIP-L/14 architecture or its .5 normalization.

Data: frozen latest FS9/FR12 manifests, fixed8 distinct RGB per natural video, canonical RGB training pool, real mass.5 and fake method mass.5/21, replacement draws with deterministic seed, full associations retained. Preserve correct/wrong x raw/L2 x mean-logit/probability tests and source/target calibration cohorts. H and 31-method historical experiments remain available, not silently rerun under a different protocol.

- [x] Preserve eight branch SHAs/tags, verified full bundle; merge all histories.
- [x] Write failing tests: epoch LR, EMA+BN buffers, skip updates, early stop/resume, source mean-prob AUC, best-EMA extraction contract, official augmentation equivalence.
- [x] Implement trainer, official model adapter and seeded DF40 augmentation; freeze config and artifact namespace.
- [x] Integrate frozen-manifest preparation, cache selection, eight-cell scoring/calibration and CLI.
- [x] Run CPU tests plus architecture forward/backward checks; no full training or target inference.
- [ ] Review diff and provenance; publish main/tags, set main default, retire eight branches with expected-SHA guards, verify conference SHA unchanged and only two branch heads remain.

Review risks: stale old caches/checkpoints must fail identity checks; resume must preserve EMA/best/wait/scheduler; selected best need not be last; BN buffers must follow conference semantics; deterministic official DINO interpolation backward must work; no claim that batch128 fits GPUs before qualification.
