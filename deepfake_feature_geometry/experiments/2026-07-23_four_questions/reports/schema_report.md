# Schema Report

## frame parquet: `/home/z/project/deepfake_detection/experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet`

### columns

| column_name        | column_type   |
|:-------------------|:--------------|
| checkpoint_id      | VARCHAR       |
| backbone           | VARCHAR       |
| seed               | BIGINT        |
| train_mode         | VARCHAR       |
| infer_mode         | VARCHAR       |
| domain             | VARCHAR       |
| method_id          | VARCHAR       |
| label              | BIGINT        |
| video_id           | VARCHAR       |
| image_path         | VARCHAR       |
| feature_norm       | DOUBLE        |
| real_logit         | DOUBLE        |
| fake_logit         | DOUBLE        |
| signed_logit       | DOUBLE        |
| prob_fake          | DOUBLE        |
| prediction_correct | TINYINT       |

total rows: 39830640

### counts by backbone

| backbone     |        n |
|:-------------|---------:|
| dinov2       | 11210100 |
| clip         | 11210100 |
| efficientnet | 11210100 |

### counts by seed

|   seed |           n |
|-------:|------------:|
|      7 | 6.72606e+06 |
|     42 | 6.72606e+06 |
|    123 | 6.72606e+06 |
|    999 | 6.72606e+06 |
|   2048 | 6.72606e+06 |

### counts by train_mode

| train_mode   |        n |
|:-------------|---------:|
| raw          | 25222725 |
| normtrain    |  8407575 |

### counts by infer_mode

| infer_mode     |       n |
|:---------------|--------:|
| direction      | 8407575 |
| normtrain_norm | 8407575 |
| raw            | 8407575 |
| length         | 8407575 |

### counts by domain

| domain   |        n |
|:---------|---------:|
| cdf      | 23777040 |
| ffpp     |  9853260 |

### counts by method_id

| method_id   |       n |
|:------------|--------:|
| real        | 9695040 |
| mobileswap  | 2851320 |
| sadtalker   | 1524120 |
| wav2lip     | 1511580 |
| tpsm        | 1501800 |
| blendface   | 1487400 |
| simswap     | 1487400 |
| fomm        | 1483980 |
| MRAA        | 1483740 |
| facedancer  | 1482600 |
| pirender    | 1479840 |
| faceswap    | 1478040 |
| facevid2vid | 1461480 |
| lia         | 1455240 |
| fsgan       | 1443360 |
| inswap      | 1031700 |
| e4s         |  771660 |

### counts by label

|   label |           n |
|--------:|------------:|
|       1 | 2.39353e+07 |
|       0 | 9.69504e+06 |

### real duplication (clip/seed42/raw/direction/ffpp)

real rows: 71664, duplicates dropped by dedup: 67185

### unique video_id per method (clip/seed42/ffpp)

| method_id   |   n_videos |
|:------------|-----------:|
| MRAA        |        140 |
| blendface   |        140 |
| e4s         |        115 |
| facedancer  |        140 |
| faceswap    |        140 |
| facevid2vid |        140 |
| fomm        |        140 |
| fsgan       |        140 |
| inswap      |        118 |
| lia         |        140 |
| mobileswap  |        858 |
| pirender    |        140 |
| real        |        140 |
| sadtalker   |        138 |
| simswap     |        140 |
| tpsm        |        140 |
| wav2lip     |        140 |


## FF++ split audit (必查项 A)

sample image_paths:
- `/Dataset/deepfake_detection/DF40_all/FaceForensics++/original_sequences/youtube/c23/frames/000/254.png`
- `/Dataset/deepfake_detection/DF40_all/FaceForensics++/original_sequences/youtube/c23/frames/000/050.png`
- `/Dataset/deepfake_detection/DF40_all/FaceForensics++/original_sequences/youtube/c23/frames/000/101.png`
- `/Dataset/deepfake_detection/DF40_all/FaceForensics++/original_sequences/youtube/c23/frames/000/305.png`
- `/Dataset/deepfake_detection/DF40_all/FaceForensics++/original_sequences/youtube/c23/frames/000/152.png`
- `/Dataset/deepfake_detection/DF40_all/FaceForensics++/original_sequences/youtube/c23/frames/000/114.png`
- `/Dataset/deepfake_detection/DF40_all/FaceForensics++/original_sequences/youtube/c23/frames/000/293.png`
- `/Dataset/deepfake_detection/DF40_all/FaceForensics++/original_sequences/youtube/c23/frames/000/140.png`
- `/Dataset/deepfake_detection/DF40_all/FaceForensics++/original_sequences/youtube/c23/frames/000/369.png`
- `/Dataset/deepfake_detection/DF40_all/FaceForensics++/original_sequences/youtube/c23/frames/000/216.png`

path encodes split token? False

distinct ffpp video_ids (raw): 1138, distinct frames: 97036

**Decision rule:** if FF++ manifest (comp/list) can be located, tag each ffpp video_id with split; else fall back to FF++ **train** for length orientation and label Q3 accordingly. See spec §3, §6.1.

path segments observed: ['000', '000.png', '012.png', '038.png', '050.png', '089.png', '101.png', '114.png', '140.png', '152.png', '178.png', '203.png', '216.png', '229.png', '254.png', '267.png', '280.png', '293.png', '305.png', '318.png', '369.png', 'DF40_all', 'Dataset', 'FaceForensics++', 'c23']
