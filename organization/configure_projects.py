"""Configure the already copied projects; no scientific source edits in Visual-Jev."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "projects/feature_geometry"
NEW = ROOT / "projects/visual_jev"
E2 = "experiments/2026-09-22_semantic_visual_jev_e2"
H1 = "experiments/2026-09-21_visual_jev_h1"


def write(project, path, text):
    dest = project / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)


REQUIREMENTS = """# Python 3.11; install in this project's own environment.
torch==2.6.0
torchvision==0.21.0
numpy>=1.24.4,<2
scipy>=1.11,<2
scikit-learn>=1.3,<2
pandas>=2,<3
pyyaml>=6
pillow>=10
opencv-python-headless>=4.9,<4.12
albumentations==2.0.8
tqdm>=4.65
ftfy>=6
regex
openai-clip==1.0.1
timm>=1.0.26,<2
duckdb>=1.1,<2
pyarrow>=14
matplotlib>=3.7,<4
seaborn>=0.12,<1
pytest>=7,<9
"""
ENV_DOC = """# 环境与运行

先进入本项目根目录。两个项目有历史遗留的同名平铺包（`data`、`models` 等），请分别创建环境，不要装在同一环境。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/check_environment.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

也可在本项目目录运行 `conda env create -f environment.yml`。
PyTorch 固定为 2.6.0 / torchvision 0.21.0；GPU 运行需适配驱动。原实验使用 CUDA 12.4 构建，可以先按 PyTorch 官方安装说明安装对应构建，再安装其余依赖。

本次迁移使用主机现有 Python 3.11 科学计算依赖做验证，完整版本记录在 `provenance/environment_observed.json`；它是观察记录，不是经过依赖解析的 lockfile。原 `requirements.txt` 中的 Albumentations 1.4.18 与实际已验证的 2.0.8 不同，新环境按代码当前所用的 2.0.8 声明。没有在本次整理中重训或声称新装环境能逐位复现 GPU 运算。

外部图像数据统一登记在 `provenance/datasets.json`，默认 `/Dataset/deepfake_detection/DF40_all`。只回放已有结果不读取图像；重新训练/推理需要这些数据及 CLIP 预训练权重（首次可能下载到用户缓存）。换数据位置时另存配置，保留冻结配置用于审计。迁移数据会改变原缓存记录的文件状态，重新提取缓存后应作为新实验登记。

Git 不保存大型模型、特征缓存及逐帧预测。当前本地副本包含这些文件；仅克隆源码时须另行传输 `experiments/*/results/` 和所需 `figures/`。从 `provenance/import_manifest.json` 可查每个原始文件的大小、SHA256 和来源。
"""


def main():
    for project, name in ((OLD, "deepfake-feature-geometry"), (NEW, "visual-jev")):
        write(project, "requirements.txt", REQUIREMENTS)
        write(project, "pyproject.toml", f'''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.11"
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {{file = ["requirements.txt"]}}

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools]
py-modules = ["config_io"]

[tool.pytest.ini_options]
addopts = "--import-mode=importlib"
pythonpath = ["src"]
testpaths = ["tests", "experiments"]
''')
        write(project, "environment.yml", f'''name: {name}
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - -r requirements.txt
      - -e .
''')
        write(project, ".gitignore", """.venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
build/
dist/
runs/
*.pth
*.pt
*.ckpt
*.onnx
*.dat
*.parquet
*.npz
experiments/*/results/
experiments/*/figures/
experiments/_runs/
""")
        write(project, "docs/ENVIRONMENT.md", ENV_DOC)
        write(project, "provenance/datasets.json", json.dumps({
            "dataset_root": "/Dataset/deepfake_detection/DF40_all",
            "primary_domains": ["ffpp", "cdf"], "external_read_only": True,
            "note": "Dataset files are external. DFD historical assets are retained only for provenance; no DFD evaluation is requested."}, indent=2) + "\n")
        write(project, "scripts/check_environment.py", (ROOT / "organization/check_environment.py.template").read_text())
    write(OLD, "scripts/replay_threshold.py", (ROOT / "organization/replay_threshold.py.template").read_text())
    for name in ("relocate_artifacts.py", "replay_e2.py"):
        write(NEW, "scripts/" + name, (ROOT / "organization" / (name + ".template")).read_text())
    # Give audit/report utilities stable source locations outside ignored results.
    for name in ("audit_final_artifacts.py", "export_report_tables.py", "plot_g1_result.py"):
        text = (NEW / E2 / "results/progress" / name).read_text()
        if name == "audit_final_artifacts.py":
            text = text.replace("E2 = Path(__file__).resolve().parents[2]", "E2 = Path(__file__).resolve().parents[1]")
        elif name == "export_report_tables.py":
            text = text.replace("result_root = Path(__file__).resolve().parents[1]", 'result_root = Path(__file__).resolve().parents[1] / "results"')
        else:
            text = text.replace("ROOT = Path(__file__).resolve().parents[2]", "ROOT = Path(__file__).resolve().parents[1]")
        write(NEW, E2 + "/scripts/" + name, text)
    # Keep shared data regression coverage without importing old-paper diagnosis.
    source = (ROOT / "tests/test_audit_fixes.py").read_text()
    import ast
    nodes = ast.parse(source)
    selected = [ast.get_source_segment(source, node) for node in nodes.body if isinstance(node, ast.FunctionDef)
                and node.name in {"test_crop_region_degenerate_box", "test_shared_spatial_augment_not_normalized", "test_shared_transform_pair_varies_across_calls"}]
    write(NEW, "tests/test_shared_data_regressions.py", "import numpy as np\nfrom data.crops import crop_region\nfrom data.transforms import apply_shared_transform_pair, build_shared_spatial_augment\n\n" + "\n\n".join(selected) + "\n")
    for phase in (H1, E2):
        path = NEW / phase / "README.md"
        original = path.read_text()
        notice = "> 独立项目入口：请先阅读本项目 [README](../../README.md)。本页下方保留原实验操作记录；既有结果回放请用 `python scripts/replay_e2.py`，新输出写入 `runs/`。历史训练和评估命令中的结果路径不要用于覆盖冻结结果。\n\n"
        if phase == H1:
            notice = "> 独立项目入口：请先阅读本项目 [README](../../README.md)。H1 训练入口已改为本项目自己的 `scripts/train_h1.py`；下方记录保留原实验口径。\n\n"
        original = original.replace("Run from `/home/z/project/deepfake_detection`.", "Run from this standalone project's root.")
        path.write_text(notice + original)
    baseline = "experiments/2026-05-05_paper_feature_geometry"
    write(NEW, baseline + "/README.md", """# H1 引用的历史基线资产

此目录只含从旧论文复制的 CLIP raw seed 42 检查点及其配置、训练记录，供冻结的 H1 配置按原相对路径加载。没有旧论文的分析源码或其他 29 个检查点。保留目录名是为了保持冻结配置的字节与哈希一致。

该资产已在 `provenance/import_manifest.json` 登记，H1 的 `baseline_selection.csv` 记录验证集选择依据。它是新项目的本地资产，不在运行时访问旧项目。
""")
    write(OLD, "README.md", """# 论文一：特征几何、池化与阈值迁移

研究问题：检测器的特征方向、特征长度、视频聚合和决策阈值分别怎样影响跨域结果。本项目含自己的代码、配置、测试及五阶段完整本地实验产物，可以整体搬走单独运行。

## 先读什么

| 顺序 | 内容 | 入口 |
|---|---|---|
| 1 | 当前结论：CDF 中方向优势的解释及阈值失效 | [最终阈值迁移报告](experiments/2026-07-30_threshold_transfer/reports/threshold_transfer_final_report.md) |
| 2 | 池化与 sigmoid 饱和效应 | [池化报告](experiments/2026-07-27_pooling_saturation/reports/pooling_saturation_final_report.md) |
| 3 | 特征假设与同帧位移审查 | [四个问题报告](experiments/2026-07-23_four_questions/reports/four_questions_final_report.md) |
| 4 | 分组口径修正与最初实验 | [原始结果修正](experiments/2026-05-05_paper_feature_geometry/reports/final_results_notes.md) |
| 5 | 实验全目录与原有使用说明 | [阶段索引](experiments/README.md) · [历史说明](docs/legacy_workspace_readme.md) |

旧结果中按裸 `video_id` 分组的数字已被后续审查修正。写论文时以带方法标识的视频分组和后续阶段结论为准；不能继续把历史 CDF 方向优势写成已成立的特征几何发现。

## 使用

[安装与环境](docs/ENVIRONMENT.md)。在本项目根目录：

```bash
python scripts/check_environment.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python scripts/replay_threshold.py
```

最后一条会实际执行七步 CPU 小规模回放（CLIP、seed 42、两种伪造方法），输出到新的 `runs/threshold_*/`，包含表格、图和报告。它验证管线，不代替冻结的全规模论文结果。原五阶段脚本仍按自身相对路径工作。

重新训练入口为 `experiments/2026-05-05_paper_feature_geometry/scripts/train.py --config <配置>`，用 `torchrun` 启动多卡任务；配置和输出目录应为新实验另存。

## 目录和边界

- `src/diagnosis`、`src/study`、`src/study_pooling`、`src/study_threshold`：本论文的分析。
- `src/data`、`src/models`、`src/engine`、`src/evaluation`：本地训练评估基础模块。
- `experiments/`：2026-05-05 至 2026-07-30 的五阶段代码、配置、报告和数据产物。
- `provenance/`：导入文件哈希、共同代码版本、外部数据登记和原环境记录。
- `runs/`：本项目的新回放输出，可与冻结结果分开阅读。

H1/E2 不属于本论文。共同模块在本项目内有自己的快照，运行不依赖另一个项目或父工作区。不要在同一 Python 环境安装两个项目。
""")
    write(NEW, "README.md", f"""# 论文二：Visual-Jev 假设验证

研究问题：在冻结视觉表征下，候选条件共享评分器能否保持检测能力（H1），进一步加入问句、候选和多任务训练后能否形成可迁移的语义决策能力（E2）。本项目可独立安装和运行，只汇报 FF++、CDF。

## 先读什么

1. [E2 最新中文汇报]({E2}/reports/e2_validation_summary_zh.md)：三种子 × 两主域 × 七条件，共 42 组，冻结结论 `not_supported`；G3 通过，G1/G2/G4/G5/G6 未通过。
2. [协议审查及局限]({E2}/reports/protocol_audit_zh.md)：区分否定当前实现与否定整个研究方向。
3. [H1 验证报告]({H1}/reports/h1_validation_result.md)：支持固定候选接口的非劣替换，不构成开放语义推理证明。
4. [H1 设计](docs/superpowers/specs/2026-09-21-visual-jev-h1-design.md) · [E2 设计](docs/superpowers/specs/2026-09-22-e2-semantic-visual-jev-design.md)。

E2 固定问答检测 AUROC：FF++ H1/E2 = 0.998531 / 0.998487；CDF = 0.911267 / 0.822181。研究写作必须保留这个跨域退化，不能仅凭候选置换通过宣称获得语义推理能力。

## 使用

[安装与环境](docs/ENVIRONMENT.md)。在本项目根目录：

```bash
python scripts/check_environment.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python scripts/replay_e2.py
```

回放只使用 CPU 和已有模型/预测文件：先验证输入字节并重定位路径元数据，再按冻结配置执行 10,000 次 bootstrap 门槛评估及 FF++/CDF 独立审计，输出到新的 `runs/e2_replay_*/`。完整指标、置信区间和结论应与冻结报告一致。

项目整体移动后也执行同一命令。原始绝对路径元数据保存在 `provenance/path_metadata_original/`，迁移日志在 `provenance/path_relocations.json`。模型、预测、语言表、锁定配置和 E2 评估源码不因迁移改变。

H1 训练入口为 `scripts/train_h1.py`，例如在为新实验另存配置与输出路径后运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \\
  scripts/train_h1.py --config <新实验的H1配置.yaml>
```

E2 训练/图像推理入口在 [{E2}/scripts/]({E2}/scripts)，原步骤见 [实验记录]({E2}/README.md)。新训练须另存配置和输出目录；复制的冻结检查点无需重训。历史配置仍登记 DFD，三域缓存清单是原来源指纹的一部分；本项目默认回放只裁决 FF++、CDF，不运行 DFD 评估。

## 目录和边界

- `src/models/clip_visual_jev.py` 与 `clip_semantic_jev.py`：H1、E2 模型。
- `src/data/semantic_*`、`src/engine/semantic_trainer.py`、`src/evaluation/semantic_*` 和 `e2_*`：语言数据、训练、对照和裁决。
- `{H1}`、`{E2}`：正式实验与历史尝试。最终结果入口固定为 E2 `results/evaluation_primary_v2` 和 `results/summary`，旧 `evaluation`、smoke 目录只作历史追溯。
- `experiments/2026-05-05_paper_feature_geometry`：仅为 H1 复制的单个基线资产，[用途说明](experiments/2026-05-05_paper_feature_geometry/README.md)。保留原相对路径以保持配置哈希。
- `provenance/`：输入哈希、共同模块来源、环境、数据及路径迁移记录。
- `runs/`：新的回放输出。

基础模块是本项目自己的代码副本，没有运行时跨项目导入。两个项目使用不同 Python 环境。H1 的 `export_cached_baseline.py` 是历史缓存转换工具，需要显式提供逐帧 Parquet；日常评估可直接使用本地基线检查点与 H1 `evaluate_model.py`，已有基线预测也已随项目复制。
""")
    write(NEW, "experiments/README.md", f"""# Visual-Jev 实验索引

| 实验 | 研究职责 | 结果入口 |
|---|---|---|
| [H1](2026-09-21_visual_jev_h1/README.md) | 固定问句与候选的共享评分器 | [H1 报告](2026-09-21_visual_jev_h1/reports/h1_validation_result.md) |
| [E2](2026-09-22_semantic_visual_jev_e2/README.md) | 多任务语义决策与六道门槛 | [最新中文汇报](2026-09-22_semantic_visual_jev_e2/reports/e2_validation_summary_zh.md) |
| [引用基线](2026-05-05_paper_feature_geometry/README.md) | H1 初始化资产 | 只含选中的 CLIP 检查点及来源文件 |

新操作从项目根 README 开始。阶段 README 中保留的旧命令和中间报告描述当时状态，最终结果以 E2 `evaluation_primary_v2` 为准。
""")
    attachment = Path("/home/z/.codex/attachments/97491804-3ed5-4584-ad1a-c8ac25ac9f07/已粘贴的文本.txt")
    if attachment.is_file():
        shutil.copy2(attachment, NEW / "docs/research_direction_original.txt")
        write(NEW, "docs/RESEARCH_NOTES.md", """# 研究方向原稿

[用户原稿](research_direction_original.txt) 按原字节保存，用于追溯研究意图。文中的实验进度和因果解释是当时记录，含未完成阶段，不应当作最新验证结论。

当前结果请从 [项目 README](../README.md) 进入 H1/E2 正式报告。
""")


if __name__ == "__main__":
    main()
