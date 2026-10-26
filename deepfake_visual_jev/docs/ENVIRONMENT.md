# 环境与运行

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

搬家后请重新创建 `.venv`；虚拟环境本身含绝对路径，不属于可迁移的实验资产。

也可在本项目目录运行 `conda env create -f environment.yml`。
PyTorch 固定为 2.6.0 / torchvision 0.21.0；GPU 运行需适配驱动。原实验使用 CUDA 12.4 构建，可以先按 PyTorch 官方安装说明安装对应构建，再安装其余依赖。

本次迁移使用主机现有 Python 3.11 科学计算依赖做验证，完整版本记录在 `provenance/environment_observed.json`；它是观察记录，不是经过依赖解析的 lockfile。原 `requirements.txt` 中的 Albumentations 1.4.18 与实际已验证的 2.0.8 不同，新环境按代码当前所用的 2.0.8 声明。没有在本次整理中重训或声称新装环境能逐位复现 GPU 运算。

外部图像数据统一登记在 `provenance/datasets.json`，默认 `/Dataset/deepfake_detection/DF40_all`。只回放已有结果不读取图像；重新训练/推理需要这些数据及 CLIP 预训练权重（首次可能下载到用户缓存）。换数据位置时另存配置，保留冻结配置用于审计。迁移数据会改变原缓存记录的文件状态，重新提取缓存后应作为新实验登记。

Git 不保存大型模型、特征缓存及逐帧预测。当前本地副本包含这些文件；仅克隆源码时须另行传输 `experiments/*/results/` 和所需 `figures/`。从 `provenance/import_manifest.json` 可查每个原始文件的大小、SHA256 和来源。
