# Conference21: 会议训练流程＋DF40增强＋最新21方法协议

协议标识 `conference21-ema-df40-20261102`。用户于2026-11-02指定本次组合；新结果必须通过重新训练得到，目前只准备代码和数据清单。

## 模块

- `trainer.py`：真实每卡batch128，15轮上限，epoch LR、AdamW/AMP/CE/clip1、参数EMA.999、源DEV pooled mean-prob AUC、patience5、可恢复最佳EMA。
- `models.py`：OpenAI CLIP B/16(512)，官方DINOv2 B/14(768)，torchvision EfficientNet B0(1280)，全参数＋线性二分类头。训练不做输出L2。
- `augmentation.py`：DF40官方变换；Albumentations1.3.1固定，按seed/epoch/样本/抽取位置固定增强种子，不污染模型随机状态。
- `data_d.py`：沿用最新FS9/FR12完整关联图、RGB去重、固定8帧、源对划分和概率采样。
- `prepare.py`：核对已冻结清单所有哈希并复制到新输出目录；独立登记15个模型。不会复制旧权重或旧预测。
- `cache_d.py`：只接受已完成训练的源DEV最佳EMA checkpoint，验证身份与最佳epoch后导出FP32特征和原二分类头。
- `scoring_d.py / analysis_d.py / bootstrap_fast.py / summary_d.py`：沿用最新八组对照、FP64读出、AUC独立重算、校准和5种子汇总。
- `third_party/provenance.json`：固定的DF40和官方DINOv2来源提交；DINO原始源码无修改，CPU位置插值桥接在models.py。

## 训练与保存语义

学习率固定CLIP=1e-5、DINOv2=5e-5、EfficientNet=1e-4。会议warmup=1表示第1轮即base LR，第2轮余弦起点仍为base LR；以后按会议公式衰减。每轮内部LR不变。没有30轮训练和5%最小LR策略。

AMP成功参数更新才更新EMA；跳过的更新单独记录。EMA平滑所有可训练参数，BN等buffer使用当轮live值，与会议实现一致。每轮源DEV验证使用EMA和正确自然视频ID：逐帧fake概率→每视频平均→pooled AUC。它是checkpoint选择指标；macro AUC同时记录，测试两者均报告。

`state.pt` 保存raw模型/优化器/scaler/EMA/随机状态/早停计数/最佳EMA；`best_model.pth` 保存源DEV最优的EMA全state（含当轮buffer）。中断恢复可从state重建最佳文件。所有测试从该最佳文件开始，同一模型、相同帧、相同原分类头比较Raw和L2。不会生成可混淆的 `final.pt`。

真实源验证集不是源CAL。源CAL140真实用于阈值；全CDF177真实＋12876伪造用于全体排序；目标真实校准子实验80 CAL、97 TEST真实＋4111 TEST伪造，200组共享嵌套子集10/20/40/80。沿用现有内容/命名代理隔离边界，不宣称完整生成祖先独立保证。

## 操作

在仓库根运行 `python run.py --help`。新机器先在独立Python3.11环境安装 `requirements.execution.lock.txt`，提供 `execution.example.json` 中的初始化权重与既有数据清单来源。该配置包含本机绝对路径，迁移机器必须调整路径并重新prepare。环境文件记录本机实际版本。

```bash
PYTHONPATH=studies/conference21 python -m pytest studies/conference21/tests -q
```

CPU合成检查与真实预训练模型单样本前后向验证，不等同于完整训练或每卡batch128显存验证。GPU 0–3每卡一个训练进程可以并行不同种子；代码不触碰4–7或其他服务。
