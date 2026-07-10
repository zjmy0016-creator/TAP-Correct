# 工作日志与项目文档同步实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** 依据现有实验产物补全中文工作日志，同步所有活跃文档和当前论文 Word 稿，并将可安全识别的旧稿及重复文件集中到待确认删除目录。

**Architecture:** 以 `outputs/` 中的可核验结果为数据事实层，以 `工作日志.md` 为历史审计层，以 README/复现材料/证据地图为当前说明层，以 `_manuscript.md` 和对应 DOCX 为唯一论文交付层。文件归档与内容更新分开执行，所有移动均写入清单且不做删除。

**Tech Stack:** Markdown、CSV、PowerShell、Python 3、OpenXML/DOCX 渲染工具、Git、pytest

---

## 文件结构与职责

- `docs/superpowers/specs/2026-07-10-worklog-document-sync-design.md`：本次整理的已确认设计。
- `docs/superpowers/plans/2026-07-10-worklog-document-sync.md`：本实施计划。
- `工作日志.md`：完整实验演进记录，保留旧结果但明确废弃关系。
- `README.md`、`REPRODUCIBILITY.md`、`FINAL_EXPERIMENT_PACKAGE.md`、`V1_PROTOCOL.md`：当前项目入口、复现步骤、正式实验包和冻结协议。
- `执行计划.md`、`证据地图.md`、`审稿人问题清单.md`、`论文重写大纲_PR版.md`：当前写作与证据状态。
- `_manuscript.md`：唯一当前论文 Markdown 源稿。
- `论文_TAP-Correct_PR中文正文.docx`：唯一当前 Word 交付稿。
- `_待确认删除_2026-07-10/README_待删除清单.md`：所有移动文件的审计记录。

### Task 1: 建立实验事实核对表

**Files:**
- Read: `outputs/**/*.csv`
- Read: `outputs/**/*.md`
- Create: `_qa/2026-07-10_document_sync/experiment_fact_audit.md`

- [ ] **Step 1: 读取当前关键实验汇总**

核对以下事实源：

```text
outputs/unified_baseline_reeval/unified_baseline_reeval_K16.csv
outputs/endpoint_source_ablation/endpoint_source_ablation_vitb32_K16.csv
outputs/endpoint_source_ablation/endpoint_source_ablation_vitb16_K16.csv
outputs/endpoint_source_ablation/endpoint_source_ablation_vitl14_K16.csv
outputs/tomato_endpoint_source/tomato_source_frontier_laboro_tomato_vitb16_K16.csv
outputs/uncertainty_evidence/uncertainty_evidence_summary_K16.csv
outputs/linear_probe_reference/linear_probe_reference_K16.csv
outputs/calibration_budget_sweep/calibration_budget_sweep_K16.csv
outputs/k_ablation/k_ablation_summary.csv
outputs/hyperparam_sensitivity/hyperparam_sensitivity_K16.csv
outputs/strawberryds_eval/strawberryds_eval_maincal_summary.csv
outputs/strawberryds_eval/strawberryds_eval_recalib_summary.csv
```

预期：每项都能读取，且表头、行数和关键工作点与脚本/Markdown 汇总一致。

- [ ] **Step 2: 形成事实核对表**

在审计文件中记录“结论—数值—来源文件—来源行/筛选条件—应更新文档”，重点覆盖：E(tip) 主线、统一基线重评、跨骨干/K/预算、番茄三源、三源不确定性、LP-48/LP-648、Strawberry-DS 三场景。

- [ ] **Step 3: 检查事实核对表无空值**

Run:

```powershell
Select-String -Path '_qa/2026-07-10_document_sync/experiment_fact_audit.md' -Pattern 'TBD|TODO|待核|未知|缺失'
```

Expected: 无输出。

### Task 2: 统一并补全工作日志

**Files:**
- Modify: `工作日志.md`
- Reference: `_qa/2026-07-10_document_sync/experiment_fact_audit.md`

- [ ] **Step 1: 更新日志顶部阅读说明**

加入当前主线：E(tip)、q80、K=16、ViT-B/32 的 7.6/92.4/82.8/18.5，并说明 V0、旧番茄自适应标定和旧 visual-only 结果只用于历史追溯。

- [ ] **Step 2: 重写 2026-07-08 T1–T5**

把英文标题和正文统一改成中文，结构固定为：实验目的、协议与设置、脚本与命令、结果、解释与边界、产物、结论/后续影响。保留所有可核验数字，不扩大主张。

- [ ] **Step 3: 合并 2026-07-09 重复记录**

删除第 2375 行附近的五条一句话重复摘要，把唯一信息合并到后续详细节；保留端点源消融、T2/T3 重建、LP 与预算扫描、文件事故、最终定位各一份详细记录。

- [ ] **Step 4: 补记 Strawberry-DS 与 2026-07-10 工作**

新增 Strawberry-DS 数据准备、特征缓存、三种部署情景及关键结果；新增正文重构、DOCX 生成和本轮文档同步记录。

- [ ] **Step 5: 扫描日志旧口径与结构**

Run:

```powershell
Select-String -Path '工作日志.md' -Pattern '^## 2026-07-08 T[1-5]:|Purpose:|Motivation:|Honest interpretation:|Role in paper:'
Select-String -Path '工作日志.md' -Pattern 'V1 降低 false-pick 风险 33.4%|11.5% vs 17.3%'
```

Expected: 第一条无输出；第二条只出现在明确标注“旧口径/已废弃”的历史段落。

### Task 3: 同步活跃项目文档

**Files:**
- Modify: `README.md`
- Modify: `REPRODUCIBILITY.md`
- Modify: `FINAL_EXPERIMENT_PACKAGE.md`
- Modify: `V1_PROTOCOL.md`
- Modify: `执行计划.md`
- Modify: `证据地图.md`
- Modify: `审稿人问题清单.md`
- Modify: `论文重写大纲_PR版.md`

- [ ] **Step 1: 更新项目入口与复现材料**

把推荐实例、统一基线重评、新增脚本/产物、Strawberry-DS 和当前限制写入 README、复现说明、正式实验包与协议；删除把旧 visual endpoint 当最终主线的表述。

- [ ] **Step 2: 更新计划和证据状态**

在执行计划、证据地图和审稿清单中：替换旧番茄结果；把丢失的 T2/T3/T4 脚本路径改为 `scripts/uncertainty_evidence.py`；记录 A1/A2/B1/B2 和独立域实验的完成状态；保留未完成事项。

- [ ] **Step 3: 更新论文重写大纲**

核对主表、图表清单和三场景独立域数据，使其与事实核对表一致；标记已经进入 `_manuscript.md` 的内容，不再保留错误的“待做”状态。

- [ ] **Step 4: 扫描所有活跃 Markdown 的残留冲突**

Run:

```powershell
$files = 'README.md','REPRODUCIBILITY.md','FINAL_EXPERIMENT_PACKAGE.md','V1_PROTOCOL.md','执行计划.md','证据地图.md','审稿人问题清单.md','论文重写大纲_PR版.md'
Select-String -Path $files -Pattern 'scripts/t2_borderline_auroc.py|scripts/t3_aurc_main_evidence.py|scripts/t4_tomato_uncertainty.py|11.5% vs 17.3%|V1 降.*33.4%'
```

Expected: 无未标注的当前主张；如为历史说明，同行必须包含“旧/废弃/作废/取代”之一。

### Task 4: 核对当前论文 Markdown

**Files:**
- Modify: `_manuscript.md`
- Reference: `_qa/2026-07-10_document_sync/experiment_fact_audit.md`

- [ ] **Step 1: 对照事实表检查摘要、方法、结果和局限**

确认 E(tip) 7.6/92.4/82.8/18.5、统一基线三段 regime、跨骨干/K/预算、番茄压力测试、Strawberry-DS 三场景、LP 参照和序贯测试限制全部有来源且前后一致。

- [ ] **Step 2: 修正事实冲突和过时路径**

只改动与最新实验事实不一致的句子、表格和图注；不做无关扩写，不新增无法由产物支持的主张。

- [ ] **Step 3: 检查正文结构和关键数字**

Run:

```powershell
Select-String -Path '_manuscript.md' -Pattern '7\.6%|92\.4%|82\.8%|18\.5%|Strawberry-DS|linear probe|序贯'
Select-String -Path '_manuscript.md' -Pattern '11\.5% vs 17\.3%|33\.4%'
```

Expected: 第一条覆盖摘要、结果或局限中的所需证据；第二条无输出。

### Task 5: 生成并视觉校验当前 Word 正文

**Files:**
- Modify: `论文_TAP-Correct_PR中文正文.docx`
- Create: `_qa/2026-07-10_document_sync/docx_render/`

- [ ] **Step 1: 从当前 Markdown 同步 DOCX 内容**

使用文档技能规定的 DOCX 管线，保留现有页面设置和正文样式；当前文件若无法安全就地修改，先输出临时候选，验证通过后再替换目标文件。

- [ ] **Step 2: 渲染全部页面**

使用 `render_docx.py` 生成逐页 PNG；预期无渲染错误，页数大于 0。

- [ ] **Step 3: 页面级视觉检查**

检查标题层级、中文字体、表格溢出、图注、分页、孤行、字符乱码和页尾截断；发现问题即修改并重新渲染。

- [ ] **Step 4: 文本一致性检查**

从 DOCX 提取纯文本，检查主线数字、Strawberry-DS、linear probe 和局限段存在；同时确认旧数字没有作为当前结论残留。

### Task 6: 集中重复和旧文件

**Files:**
- Create: `_待确认删除_2026-07-10/README_待删除清单.md`
- Move: `_v.docx`
- Move: approved historical drafts/backups from project root
- Move: byte-identical duplicate outputs only

- [ ] **Step 1: 建立四类待删除目录和清单**

建立 `01_精确重复`、`02_旧稿与备份`、`03_临时与锁文件`、`04_重复实验产物`；清单记录 SHA-256、原路径、新路径、保留副本和判定理由。

- [ ] **Step 2: 移动精确重复文件**

至少处理 `_v.docx`（保留 `论文_TAP-Correct_PR中文正文.docx`）、Strawberry-DS 的同哈希别名，以及 K16 消融与 official eval 的同哈希副本。人工标注图像的结构性重复不移动，以免破坏审计链。

- [ ] **Step 3: 移动确认的旧稿和临时文件**

移动名称明确为初稿、v2、备份、旧大纲的文件；若 `TAP-Correct_中文版_带行号_初版备份.docx` 仍被占用，仅在清单记录“未移动：被占用”，不强制操作。Word 锁文件只在对应文档已关闭时移动。

- [ ] **Step 4: 验证没有直接删除**

对照移动前后清单，确认每个候选在待删除区可找到，权威副本仍在原位置。

### Task 7: 全项目验证与交付

**Files:**
- Modify: `_qa/2026-07-10_document_sync/final_verification.md`

- [ ] **Step 1: 运行测试**

Run:

```powershell
python -m pytest -q
```

Expected: 所有测试通过；若存在与本次文档工作无关的既有失败，记录完整命令和失败原因，不掩盖。

- [ ] **Step 2: 运行 Markdown 和路径检查**

检查 UTF-8、标题层级、表格列数、链接/相对路径存在性、旧脚本名和旧数字残留。

- [ ] **Step 3: 审计 Git 状态和差异**

Run:

```powershell
git status --short
git diff --check
git diff --stat
```

Expected: 无空白错误；差异只覆盖计划内文档、QA 报告和待删除移动，不覆盖用户未授权的代码/实验改动。

- [ ] **Step 4: 完成交付说明**

在最终验证文件列出：更新文件、补记实验、仍保留的历史口径、待删除候选数量、因占用未移动的文件、测试和 DOCX 渲染结果。

## Verification

- 关键实验数据逐项映射到来源 CSV/汇总文件。
- `工作日志.md` 的 2026-07-08 T1–T5 不再含英文流水账，7 月 9 日无重复摘要，Strawberry-DS 与 7 月 10 日工作已补记。
- 活跃文档不再把旧番茄 11.5%/17.3% 或 visual-only 端点当成最终主线。
- `_manuscript.md` 与当前 DOCX 的核心文本和数字一致，DOCX 全页渲染通过。
- 所有移动可由待删除清单逆向恢复，没有直接删除。
- 项目测试与 Git diff 审计完成。

**Next skill:** `$superpower-executing-plans`
