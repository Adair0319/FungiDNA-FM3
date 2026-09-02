# FungiDNA-FM 守卫式调用工作流 — 设计文档

日期：2026-08-26
状态：已定稿，待实施
依据：论文《FungiDNA-FM》 Fig 7「A guarded dual-invocation access to FungiDNA-FM」

## 1. 目标

为 FungiDNA-FM 提供一个**守卫式调用工作流**，暴露两种用户入口（嵌入式 API、
自然语言终端），二者收敛到同一个结构化请求 + 同一套确定性验证 + 分发管线。

核心约束（不可违背）：

1. LLM 语义提取是**可选的兜底**，且只做候选提取，**绝不直接执行任务**。
2. 每个请求必须过**同一个**确定性验证门。
3. 无效请求**绝不触发模型执行**。
4. 两种入口**共用**下游实现，不重复业务逻辑。
5. 原始自然语言文本**不直接**传入任务执行层。

## 2. 支持的任务白名单

canonical 任务名（唯一真相源，集中在 `fungidna/invoke/tasks.py`）：

| canonical | 别名（须规范化为 canonical） |
|---|---|
| `representation` | representations, embedding, embeddings |
| `taxonomy` | taxonomic classification, classification |
| `coding_potential` | coding potential, coding-potential |
| `splice_site` | splice site, splice-site, splice |
| `bgc_boundary` | bgc boundary, BGC boundary, bgc |

白名单集中在 `tasks.py`，其余模块只 import，禁止复制。未知任务规范化失败即
返回 `TASK_UNSUPPORTED`，**绝不静默接受**。

## 3. 结构化请求

`fungidna/invoke/request.py` 定义：

```python
class Task(str, enum.Enum):
    REPRESENTATION = "representation"
    TAXONOMY = "taxonomy"
    CODING_POTENTIAL = "coding_potential"
    SPLICE_SITE = "splice_site"
    BGC_BOUNDARY = "bgc_boundary"

@dataclasses.dataclass(frozen=True)
class StructuredRequest:
    task: Task            # 枚举，类型层面杜绝非法任务
    file_path: pathlib.Path
```

`task` 用枚举而非裸字符串，使"不存在的任务"在类型层面难以构造。请求一旦
构造成功，其 `task` 必为白名单成员。

**规范化发生在验证门内部**：`parse_prompt` 与 `run_task` 先产出**候选**
（task 为字符串或 None、file_path 为字符串或 None），`validation.validate()`
调用 `tasks.normalize()` 把候选 task 字符串转成 `Task` 枚举（失败即
`TASK_UNSUPPORTED`），全部检查通过后才返回类型化的 `StructuredRequest`。
候选态与已校验态是两种不同的结构，避免"谁来规范化"产生歧义。

## 4. 模块划分

```
fungidna/invoke/
├── __init__.py        # 导出 run_task、parse_prompt、Report
├── tasks.py           # 白名单 + 别名表 + normalize()
├── request.py         # Task 枚举 + StructuredRequest
├── validation.py      # 确定性验证门（唯一入口）
├── parser.py          # 终端：正则提取 + 可选 LLM 兜底
├── dispatcher.py      # 分发器
├── report.py          # Report 输出结构
├── handlers/
│   ├── base.py        # handler 基类 + 可插拔 WeightLoader
│   ├── representation.py
│   ├── taxonomy.py
│   ├── coding_potential.py
│   ├── splice_site.py
│   └── bgc_boundary.py
└── (fungidna/__main__.py)   # python -m fungidna "<prompt>"
```

## 5. 数据流

```
run_task(file_path, task)             # 嵌入式 API
parse_prompt("请预测剪接位点 ...fa")   # 终端
        │
        ├──(终端: 正则 → 可选 LLM 兜底)──┐
        └──(API: 显式参数)──────────────┤
                                       ▼
              StructuredRequest(task, file_path)
                                       ▼
              validation.validate()      ← 唯一验证入口
                 FAIL → ValidationError(code, message)，绝不执行
                 PASS → dispatcher.dispatch(request) → handler.run() → Report
```

`run_task` 内部 = `StructuredRequest → validate → dispatch`；`parse_prompt`
只是把文本变成 `run_task` 的参数，随后完全共用同一条下游路径。

## 6. 模式 A：嵌入式 API

```python
from fungidna.invoke import run_task
report = run_task(file_path="/path/test.fasta", task="representation")
```

语义：调用方显式给 task 与 file_path，无自然语言解析；走同一验证门；验证后
用同一 dispatcher。签名可适配项目约定，但语义不变。

## 7. 模式 B：终端

```bash
python -m fungidna "Please predict the splice sites using /home/ssd/test.fasta"
python -m fungidna "Please generate sequence representations for /home/ssd/test.fasta"
```

混合解析（`parser.py`）：

1. **正则优先**：抽取 `file_path`（绝对路径，或带 `.fa/.fasta/.fq/.fastq`
   扩展名的相对路径）与 `task`（别名关键词）。
2. **LLM 兜底（可选）**：仅当正则抽不全时，才调用 LLM 输出候选
   `{task, file_path}`；输出只作候选，仍走 `tasks.normalize()` + 同一验证门。
3. 未配置 LLM 后端且正则抽不全 → 返回 `EXTRACTION_INCOMPLETE`，提示补全，
   不编造。

LLM 后端做成可插拔接口（`parser.py` 里一个 `extract_with_llm(text)` 钩子），
默认不绑定任何具体供应商；由环境变量或配置文件注入。未配置时走纯正则路径。

## 8. 确定性验证门（`validation.py`）

规则（按顺序，任一失败即返回带 code 的 `ValidationError`）：

1. `task` 能规范化为 canonical 之一 → 否则 `TASK_UNSUPPORTED`
2. 恰好一个 task、恰好一个 file_path → 否则 `AMBIGUOUS_TASK` / `MISSING_FILE_PATH`
3. file_path 存在、普通文件、可读 → `FILE_NOT_FOUND` / `FILE_UNREADABLE`
4. 扩展名 ∈ {.fa,.fasta,.fq,.fastq} 且不压缩 → 否则 `UNSUPPORTED_FORMAT`
5. 首记录结构合法：FASTA 首行 `>` 开头 / FASTQ 首行 `@` 开头 → 否则 `INVALID_SEQUENCE_FORMAT`

全部通过后返回规范化后的 `StructuredRequest`（file_path 解析为绝对路径）。

## 9. 任务处理器与权重加载器

`handlers/base.py` 的 `WeightLoader`：从环境变量或 `~/.config/fungidna/weights.yaml`
读取「任务 → checkpoint 路径」映射，`load(task)` 返回路径。当前无权重时抛
`WeightsNotFoundError`，消息指引用户设置对应环境变量或配置文件。权重就位后
不改任何 handler 代码即可用。

| 任务 | 加载模型 | 输入处理 | 输出 |
|---|---|---|---|
| `representation` | `StripedMambaBackbone` | `encode_bpe` → 均值池化 768 维 | 每条序列 768 维嵌入 |
| `taxonomy` | `StripedMambaBackbone` + 小 MLP 头 | 10 kb 滑窗 → 均值池化 → MLP 预测 → 聚合 5 rank | 各 rank 预测标签 + 置信度 |
| `coding_potential` | `FungiDNAForSequenceClassification(num_classes=2)` | 每条序列二分类 | label + 概率 |
| `splice_site` | `FungiDNAForSequenceClassification(num_classes=3)` | 扫候选二核苷酸位点 → 各取 ±200 bp（401 bp 窗口）→ 3 类 | donor/acceptor/non-site + 概率 |
| `bgc_boundary` | `FungiDNAForBGCBoundary` | 10 kb 窗口（围绕锚定核心基因）→ 回归 | 左右边界距离（bp） |

> 模型类已按实际训练脚本核实：分类学是 `StripedMambaBackbone` + MLP
> （`train_phase2_joint_mlp.py`），剪接 v2 是 `FungiDNAForSequenceClassification
> (num_classes=3)`（`run_splice_v2_experiments.py`）；`FungiDNAForSpliceSite`
> 是 v1 双二分类头，不用于本工作流。

两个已确认的语义决策：

- `splice_site`：handler 内部先扫描序列中符合剪接位点二核苷酸（GT/GC/AG 等）
  的位置作为候选，再逐个打分，使单个 FASTA 即可运行，无需额外输入。
- `bgc_boundary`：接受可选 `anchor` 参数（核心基因坐标/关键词）；未提供时退化
  为「整条序列的左右边界距离」，并在 Report 中明确标注为退化行为。

## 10. 补回 BGC 模型类

论文更新后 BGC 边界回归是正式内容（Fig 6），但发布时被摘除。需最小补回：

- `fungidna/model/heads.py`：补回 `MeanPoolHead`（均值池化 → MLP → 回归输出）。
- `fungidna/model/fungi_dna.py`：补回 `FungiDNAForBGCBoundary`（backbone +
  `MeanPoolHead(768, 2)`，2 个输出 = 左右边界距离）。

不补 `FungiDNAForBGCEnzyme`、`MultiLabelHead`（论文外任务）。

## 11. Report 结构

`Report(task, file_path, status, results)`，`results` 为各任务自定义 dict。
终端模式序列化为 JSON 打到 stdout；API 模式返回 `Report` 对象。

## 12. 验收标准

- [ ] 白名单仅在 `tasks.py` 一处，无重复
- [ ] 两种入口共用 `validate()` + `dispatcher`
- [ ] 无效请求（未知任务/缺文件/坏格式/压缩文件）返回结构化错误且不执行
- [ ] `python -m fungidna "..."` 与 `run_task(...)` 产出相同结构
- [ ] LLM 只做候选提取，无法绕过验证门
- [ ] 五个 handler 各自加载对应 checkpoint；无权重时抛明确 `WeightsNotFoundError`
- [ ] `FungiDNAForBGCBoundary` / `MeanPoolHead` 补回且 import 通过
- [ ] 单测覆盖：别名规范化、验证门各失败分支、正则抽取、Report 序列化
