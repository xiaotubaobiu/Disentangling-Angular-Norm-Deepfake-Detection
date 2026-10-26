# E2 global constraints

- Dataset root is /Dataset/deepfake_detection/DF40_all.
- Reuse H1 real/fake labels, train/validation/test splits, sixteen methods, frame sampling, crops, augmentations, normalization, and method_id::video_id aggregation.
- The visual encoder is the exact frozen H1 CLIP ViT-B/16 visual tower.
- The CLIP text encoder is frozen; held-out text is not encoded before final evaluation.
- Only image/question/candidate projections, the two-layer decision transformer, and the shared scalar scorer are trainable.
- d_model is 256, attention heads are 8, feed-forward dimension is 1024, dropout is 0.1, and there are exactly two decoder layers.
- Loss is candidate cross entropy only.
- Candidate sets contain two items for T1 through T4 and three items for T5; padded candidates never affect logits or loss.
- Training tasks T1 through T5 are sampled uniformly in expectation.
- Final seeds are 42, 7, and 2048; GPUs 0 through 3 are the only authorized devices.
- Checkpoint selection uses deterministic FF++ validation seen-language cross entropy only.
- FF++ test, CDF, DFD, held-out text, and meaningless controls are forbidden inputs to model selection.
- G1 through G6 and their numeric thresholds are immutable after final evaluation starts.
