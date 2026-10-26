# Q4 配对审计(硬门)

## ffpp:候选同帧组 4479;抽样 300(含FS 300, FR 281)

- group ('ffpp', '190', '117'):methods=['MRAA', 'blendface', 'facedancer', 'faceswap', 'facevid2vid', 'fsgan', 'inswap', 'lia', 'mobileswap', 'real', 'simswap']
- group ('ffpp', '029', '107'):methods=['MRAA', 'blendface', 'e4s', 'facedancer', 'faceswap', 'facevid2vid', 'fsgan', 'lia', 'mobileswap', 'real', 'simswap']
- group ('ffpp', '462', '263'):methods=['MRAA', 'blendface', 'fsgan', 'inswap', 'lia', 'mobileswap', 'real', 'simswap']
- group ('ffpp', '421', '468'):methods=['MRAA', 'blendface', 'e4s', 'facedancer', 'faceswap', 'facevid2vid', 'fsgan', 'inswap', 'lia', 'mobileswap', 'real', 'simswap']
- group ('ffpp', '389', '040'):methods=['MRAA', 'blendface', 'e4s', 'facedancer', 'faceswap', 'facevid2vid', 'fsgan', 'inswap', 'lia', 'mobileswap', 'real', 'simswap']
- group ('ffpp', '255', '151'):methods=['MRAA', 'blendface', 'facedancer', 'facevid2vid', 'fsgan', 'inswap', 'lia', 'mobileswap', 'real', 'simswap']
- group ('ffpp', '176', '205'):methods=['MRAA', 'blendface', 'facedancer', 'facevid2vid', 'fsgan', 'inswap', 'lia', 'mobileswap', 'real', 'simswap']
- group ('ffpp', '995', '088'):methods=['MRAA', 'blendface', 'e4s', 'facedancer', 'faceswap', 'facevid2vid', 'fsgan', 'inswap', 'lia', 'mobileswap', 'real', 'simswap']

- 组内 real 存在 & >=2 方法的违规:0/300
- 同时含 FS 和 FR 方法的组:281/300(若高,FS/FR 必须分开分析)
- AUTO GATE (ffpp):PASS

## cdf:候选同帧组 2560;抽样 300(含FS 294, FR 299)

- group ('cdf', 'id31', '405'):methods=['blendface', 'facedancer', 'faceswap', 'facevid2vid', 'fomm', 'fsgan', 'mobileswap', 'pirender', 'real', 'simswap', 'tpsm']
- group ('cdf', 'id6', '095'):methods=['MRAA', 'blendface', 'e4s', 'facedancer', 'faceswap', 'facevid2vid', 'fsgan', 'inswap', 'lia', 'mobileswap', 'real', 'sadtalker', 'simswap', 'tpsm', 'wav2lip']
- group ('cdf', 'id28', '154'):methods=['MRAA', 'blendface', 'e4s', 'faceswap', 'facevid2vid', 'fsgan', 'inswap', 'lia', 'mobileswap', 'real', 'simswap', 'tpsm', 'wav2lip']
- group ('cdf', 'id33', '470'):methods=['MRAA', 'blendface', 'faceswap', 'lia', 'mobileswap', 'real', 'simswap', 'tpsm']
- group ('cdf', 'id28', '270'):methods=['MRAA', 'blendface', 'e4s', 'facedancer', 'faceswap', 'facevid2vid', 'fomm', 'fsgan', 'inswap', 'lia', 'mobileswap', 'pirender', 'real', 'simswap', 'tpsm']
- group ('cdf', 'id2', '054'):methods=['MRAA', 'blendface', 'e4s', 'faceswap', 'facevid2vid', 'fsgan', 'inswap', 'lia', 'mobileswap', 'real', 'sadtalker', 'simswap', 'tpsm', 'wav2lip']
- group ('cdf', 'id50', '315'):methods=['e4s', 'facevid2vid', 'real']
- group ('cdf', 'id49', '000'):methods=['MRAA', 'blendface', 'e4s', 'facedancer', 'faceswap', 'facevid2vid', 'fomm', 'fsgan', 'inswap', 'lia', 'mobileswap', 'pirender', 'real', 'sadtalker', 'simswap', 'tpsm', 'wav2lip']

- 组内 real 存在 & >=2 方法的违规:0/300
- 同时含 FS 和 FR 方法的组:293/300(若高,FS/FR 必须分开分析)
- AUTO GATE (cdf):PASS

## 配对强度判定

- 强配对:real + 多 fake 共享相同 frame token 且 source identity 一致(FS)/driver 一致(FR)。
- 本审计为**自动化代理**(未人工逐帧核对),判定为 **weak pairing** 除非路径结构证明强对齐。
- FS 与 FR 位移**不得**混入同一公共方向。


## OVERALL AUTO GATE: PASS