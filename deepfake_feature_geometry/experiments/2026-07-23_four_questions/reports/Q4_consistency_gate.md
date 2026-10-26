# Q4 Feature-Extraction Consistency Gate

- Spec: §7.1 step 2 (de-risking before full Q4 extraction).
- Checkpoint: `clip_raw_s42`  (`clip/raw/seed_42`)
- Domain: `ffpp`
- Frames checked: **1000**
- Tolerance (tol): `0.001`

## Verdict

### **PASS**

## Max errors

| Quantity | max abs err | max rel err |
|---|---|---|
| feature_norm (\|\|z\|\|) | 1.14441e-05 | 1.01153e-06 |
| raw signed_logit (z.w+b) | 3.43323e-05 | 4.7497e-06 |
| direction signed_logit (z/\|\|z\|\|.w+b) | 3.24845e-06 | 3.24845e-06 |

## Method

Re-extraction mirrors `scripts/frame_midt_diagnose.py` exactly:
`build_model(cfg)` → `load_state_dict(best_model.pth, strict=True)` → `.eval()`,
`build_eval_loader(cfg, domain, distributed=False)` (shuffle=False), then
`extract_raw_features`. The signed head is `w_signed = w_fake - w_real`,
`b_signed = b_fake - b_real` (real_logit = logits[:,0], fake_logit = logits[:,1]).

- Re-extracted feature dim D = `512`
- b_signed = `0.031485`

## Interpretation

- All quantities match within tol; the extraction wiring (model factory, checkpoint load, eval loader, feature layer) is correct. Q4's full extraction can proceed.
