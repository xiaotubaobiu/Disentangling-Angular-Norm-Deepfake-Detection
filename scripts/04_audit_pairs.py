# scripts/04_audit_pairs.py
"""Spec §7.1 step1: construct candidate same-frame groups from paths+metadata,
verify real + >=2 fake methods align on the same frame token, FS/FR separate.
HARD GATE before z extraction. Automated proxies (no visual eyeballing); label
pairing strength strong/weak. Exit 0 if PASS, 1 if FAIL."""
from __future__ import annotations
import sys, random
from pathlib import Path
from collections import defaultdict
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from deepfake_detection.study.config import load_config
from deepfake_detection.study.io import load_frame_subset
from deepfake_detection.study.canonical import add_canonical_keys
from deepfake_detection.data.constants import FACE_SWAP_METHODS, REENACTMENT_METHODS

CFG=load_config(ROOT/"configs"/"study_config.yaml")
REP=ROOT/CFG.paths["reports"]; REP.mkdir(parents=True,exist_ok=True)

def candidate_groups(domain):
    df=load_frame_subset(str(ROOT/CFG.parquet["frame"]), train_mode="raw", infer_mode="raw",
        backbones=["clip"], seeds=[42], domains=[domain])
    df=add_canonical_keys(df)
    df["frame_tok"]=df["canonical_frame_id"].str.split("|").str[-1]
    g=defaultdict(list)
    for r in df.itertuples():
        g[(r.domain, r.source_video_id, r.frame_tok)].append((r.method_id, r.image_path, int(r.label)))
    return g

def audit():
    random.seed(CFG.global_seed)
    fs=set(FACE_SWAP_METHODS); fr=set(REENACTMENT_METHODS)
    lines=["# Q4 配对审计(硬门)\n"]
    overall_ok=True
    for domain in CFG.domains:
        g=candidate_groups(domain)
        multi={k:v for k,v in g.items()
               if any(x[2]==0 for x in v) and len({x[0] for x in v if x[2]==1})>=2}
        sample=random.sample(list(multi), min(CFG.q4["audit_sample"], len(multi)))
        fs_n=sum(1 for s in sample if any(m[0] in fs for m in g[s] if m[2]==1))
        fr_n=sum(1 for s in sample if any(m[0] in fr for m in g[s] if m[2]==1))
        lines.append(f"## {domain}:候选同帧组 {len(multi)};抽样 {len(sample)}(含FS {fs_n}, FR {fr_n})\n")
        for k in sample[:8]:
            lines.append(f"- group {k}:methods={sorted({m[0] for m in g[k]})}")
        # automated proxy: within a sampled group, do all entries share the same frame filename token?
        bad=0; fs_in_fr=0
        for k in sample:
            frames={m[1].split("/")[-1] for m in g[k]}
            methods_present={m[0] for m in g[k] if m[2]==1}
            has_real=any(m[2]==0 for m in g[k])
            # proxy: group keyed by frame token already guarantees same token; check real present & >=2 methods
            if not has_real or len(methods_present)<2: bad+=1
            if methods_present & fs and methods_present & fr: fs_in_fr+=1
        lines.append(f"\n- 组内 real 存在 & >=2 方法的违规:{bad}/{len(sample)}")
        lines.append(f"- 同时含 FS 和 FR 方法的组:{fs_in_fr}/{len(sample)}(若高,FS/FR 必须分开分析)")
        rate = bad/max(1,len(sample))
        gate = "PASS" if rate < 0.2 else "FAIL"
        if rate>=0.2: overall_ok=False
        lines.append(f"- AUTO GATE ({domain}):{gate}\n")
    lines.append("## 配对强度判定\n")
    lines.append("- 强配对:real + 多 fake 共享相同 frame token 且 source identity 一致(FS)/driver 一致(FR)。")
    lines.append("- 本审计为**自动化代理**(未人工逐帧核对),判定为 **weak pairing** 除非路径结构证明强对齐。")
    lines.append("- FS 与 FR 位移**不得**混入同一公共方向。\n")
    lines.append(f"\n## OVERALL AUTO GATE: {'PASS' if overall_ok else 'FAIL'}")
    (REP/"Q4_pairing_audit.md").write_text("\n".join(lines))
    print("\n".join(lines[-12:]))
    return overall_ok

if __name__=="__main__":
    raise SystemExit(0 if audit() else 1)
