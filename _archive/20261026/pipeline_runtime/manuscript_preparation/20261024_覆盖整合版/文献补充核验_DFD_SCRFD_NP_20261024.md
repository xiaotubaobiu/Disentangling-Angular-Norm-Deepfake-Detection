# DFD、SCRFD与次序统计阈值：原始来源补充核验 / Primary-source addendum

核验日期：2026-10-24。此文件补齐既有方法准备中的三项来源缺口，配套 BibTeX 仅含已核实字段。旧 references.json、references.bib、主中英文稿和既有方法版本均未改写；正式论文整合时再合并键及引用。这里不报告任何检测性能。

## DFD 来源 / Dataset provenance

Google Research 的原始发布说明由 Nick Dufour 与 Andrew Gully 于 2019-09-24 署名，说明 Google/Jigsaw 将视觉伪造数据纳入 FaceForensics。适合支撑数据来源介绍；本研究的候选、成功预处理和最终共同队列数量仍引用本地冻结清单，不能由发布博客替代。[原始发布说明](https://research.google/blog/contributing-data-to-deepfake-detection-research/)

The release identifies the Google/Jigsaw contribution to FaceForensics. Cite it for provenance; use accepted local manifests for study-specific cohort denominators. The release does not establish identity independence or the ancestry of this local cohort. The host repository provides a second primary record of the dataset relationship. [Dataset host](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)

建议键 / Suggested key: `dufour2019dfd`。发布博客不是期刊论文，不补造会议、页码或 DOI。精确的 363/3068 本地候选数不以博客作为核验来源。

## SCRFD 来源 / Detector attribution

arXiv 的作者、题名、编号和 2021 年提交时间已核实；正文将 SCRFD 方法链接至作者 InsightFace 实现。补充条目明确引用该预印本。作者仓库另说明 ICLR 2022 接收；本轮 OpenReview 页面与 PDF 被浏览器验证拦截，未冒称读取其出版元数据，也未把 2021 年预印本年份写成 ICLR 年份。[预印本元数据](https://arxiv.org/abs/2105.04714)；[方法正文](https://arxiv.org/html/2105.04714v1)；[作者仓库](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/README.md)

The preprint is sufficient for method attribution. It does not certify the exact ONNX artifact used here: reproducibility continues to depend on the local detector hash and frozen preprocessing settings. The conference-version bibliographic export can replace this entry after direct verification, without changing the experiment.

建议键 / Suggested key: `guo2021scrfd`。当前条目仅主张 arXiv 版本；不添加未经核实的会议 DOI。

## 次序统计阈值 / Order-statistic thresholding

Tong、Feng、Li 的期刊论文及作者站点 PDF 已核实作者、2018 年、Science Advances 4(2)、eaao1659 和 DOI。出版社 DOI 页本轮返回 403，改读作者提供的已发表 PDF，并以 arXiv 的期刊关联元数据交叉核对；不将 2016 年首次预印本年份当作期刊年份。[作者保留的已发表 PDF](https://jsb-lab.org/wp-content/uploads/2018/02/ScienceAdvances_eaao1659.full_.pdf)；[期刊关联元数据](https://arxiv.org/abs/1608.03109)

Proposition 1 and Equations (1)–(2), printed pages 3–4, give the binomial-tail order-statistic construction for independent held-out class-0 scores. Its extreme-order feasibility condition is n ≥ log(δ)/log(1−α); printed page 8 explicitly lists n_min=59 for α=δ=0.05. This identifies the source of the existing conditional calculation, rather than proving that this study meets the assumptions.

本研究不能仅凭校准池大小 ≥59 宣称总体 FPR 得到保证。CDF/DFD 的未知祖先依赖、采样来源及训练/校准独立性限制仍需披露；该文的独立留出抽样假设不能由 RGB 去重与文件名代理闭包证明。也不能把经验 FPR 控制直接写成总体保证。该条件针对该次序统计构造，不是所有阈值规则的无条件最小样本量。

建议键 / Suggested key: `tong2018np`。维持冻结的 ≥ 比较、nextafter 边界与 NA 规则，不因引用核验改变阈值算法、数据池或实验协议。

## 尚未解决 / Remaining bibliographic issue

此前会议稿是否已正式发表以及其 DOI，仍缺可核实的具体出版记录；没有从稿件题名或计划推断发表状态。完整期刊论文整合前应继续检查本项目已有出版材料。三项补充来源的核验不等于完整参考文献、全文文献评述或最终论文已完成。
