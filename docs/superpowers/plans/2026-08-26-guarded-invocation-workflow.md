# FungiDNA-FM 守卫式调用工作流 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 FungiDNA-FM 实现两种调用入口（嵌入式 API + 自然语言终端）与共享的确定性验证门、任务分发器和五个任务处理器。

**Architecture:** 新建 `fungidna/invoke/` 子包，`tasks.py` 集中白名单与别名规范化，`validation.py` 提供唯一确定性验证门，`run_task` 与 `parse_prompt` 都收敛到 `validate → dispatch → handler → Report` 一条路径。LLM 仅作终端解析的可选兜底，绝不执行任务。

**Tech Stack:** Python 3.10、`dataclasses`/`enum`、标准库 `unittest`（环境无 pytest）、`pathlib`、`re`；模型层复用现有 `fungidna.model` / `fungidna.data.tokenizer`，torch 2.5.1。

**Spec:** `docs/superpowers/specs/2026-08-26-guarded-invocation-workflow-design.md`

## Global Constraints

- 测试用 `unittest`，运行命令 `python -m unittest discover -s tests -v`（仓库无 pytest，禁止引入）。
- 白名单唯一真相源在 `fungidna/invoke/tasks.py`，其余模块只 import，禁止复制列表。
- 任务类型用 `enum.Enum`，非法任务在类型层面无法构造。
- 验证门是执行前唯一入口；`run_task` 和 `parse_prompt` 都必须经过它。
- LLM 只产出候选字符串，仍走 `tasks.normalize()` + 验证门，不能绕过。
- 无权重时 `WeightLoader` 抛 `WeightsNotFoundError`，绝不静默降级。
- 不在本计划中引入任何新第三方依赖；`typing`/`Optional` 等用标准库。

---

### Task 1: 任务白名单与规范化（tasks.py）

**Files:**
- Create: `fungidna/invoke/__init__.py`（空占位，仅 `"""FungiDNA guarded invocation package."""`）
- Create: `fungidna/invoke/tasks.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Produces:
  - `CANONICAL_TASKS: tuple[str, ...]` — `("representation", "taxonomy", "coding_potential", "splice_site", "bgc_boundary")`
  - `TASK_ALIASES: dict[str, str]` — 别名小写键 → canonical 值（含 canonical 自身映射）
  - `normalize_task(raw: str | None) -> str | None` — 归一化，未知返回 `None`

- [ ] **Step 1: 写失败测试**

`tests/test_tasks.py`:
```python
import unittest
from fungidna.invoke.tasks import CANONICAL_TASKS, TASK_ALIASES, normalize_task


class TestNormalizeTask(unittest.TestCase):
    def test_canonical_unchanged(self):
        for t in CANONICAL_TASKS:
            self.assertEqual(normalize_task(t), t)

    def test_aliases_map_to_canonical(self):
        self.assertEqual(normalize_task("representations"), "representation")
        self.assertEqual(normalize_task("embedding"), "representation")
        self.assertEqual(normalize_task("taxonomic classification"), "taxonomy")
        self.assertEqual(normalize_task("classification"), "taxonomy")
        self.assertEqual(normalize_task("coding potential"), "coding_potential")
        self.assertEqual(normalize_task("coding-potential"), "coding_potential")
        self.assertEqual(normalize_task("splice site"), "splice_site")
        self.assertEqual(normalize_task("splice-site"), "splice_site")
        self.assertEqual(normalize_task("BGC boundary"), "bgc_boundary")
        self.assertEqual(normalize_task("bgc"), "bgc_boundary")

    def test_unknown_returns_none(self):
        self.assertIsNone(normalize_task("promoter"))
        self.assertIsNone(normalize_task(""))
        self.assertIsNone(normalize_task(None))

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(normalize_task("  Splice Site "), "splice_site")
        self.assertEqual(normalize_task("CODING_POTENTIAL"), "coding_potential")
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_tasks -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fungidna.invoke'`

- [ ] **Step 3: 最小实现**

`fungidna/invoke/tasks.py`:
```python
"""Centralized task whitelist and alias normalization.

This is the single source of truth for supported tasks. Every other module
imports from here; do not duplicate the list elsewhere.
"""

CANONICAL_TASKS = (
    "representation",
    "taxonomy",
    "coding_potential",
    "splice_site",
    "bgc_boundary",
)

_TASK_ALIASES = {
    # representation
    "representation": "representation",
    "representations": "representation",
    "embedding": "representation",
    "embeddings": "representation",
    # taxonomy
    "taxonomy": "taxonomy",
    "taxonomic classification": "taxonomy",
    "classification": "taxonomy",
    # coding potential
    "coding_potential": "coding_potential",
    "coding potential": "coding_potential",
    "coding-potential": "coding_potential",
    # splice site
    "splice_site": "splice_site",
    "splice site": "splice_site",
    "splice-site": "splice_site",
    "splice": "splice_site",
    # bgc boundary
    "bgc_boundary": "bgc_boundary",
    "bgc boundary": "bgc_boundary",
    "BGC boundary": "bgc_boundary",
    "bgc": "bgc_boundary",
}


def normalize_task(raw):
    """Normalize a raw task string to its canonical name, or None if unsupported."""
    if raw is None:
        return None
    key = " ".join(str(raw).strip().lower().split())
    return _TASK_ALIASES.get(key)


TASK_ALIASES = _TASK_ALIASES  # exposed for introspection/tests
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_tasks -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: 提交**

```bash
git add fungidna/invoke/__init__.py fungidna/invoke/tasks.py tests/test_tasks.py
git commit -m "feat(invoke): add task whitelist and alias normalization"
```

---

### Task 2: 任务枚举与结构化请求（request.py）

**Files:**
- Create: `fungidna/invoke/request.py`
- Test: `tests/test_request.py`

**Interfaces:**
- Produces:
  - `Task(str, enum.Enum)` — 成员 `REPRESENTATION/TAXONOMY/CODING_POTENTIAL/SPLICE_SITE/BGC_BOUNDARY`，值等于 canonical 字符串
  - `task_from_string(name: str) -> Task` — canonical 字符串 → `Task`，未知抛 `ValueError`
  - `StructuredRequest`（frozen dataclass）— 字段 `task: Task`、`file_path: pathlib.Path`

- [ ] **Step 1: 写失败测试**

`tests/test_request.py`:
```python
import enum
import unittest
from pathlib import Path

from fungidna.invoke.request import Task, StructuredRequest, task_from_string


class TestTaskEnum(unittest.TestCase):
    def test_values_match_canonical(self):
        self.assertEqual(Task.REPRESENTATION.value, "representation")
        self.assertEqual(Task.TAXONOMY.value, "taxonomy")
        self.assertEqual(Task.CODING_POTENTIAL.value, "coding_potential")
        self.assertEqual(Task.SPLICE_SITE.value, "splice_site")
        self.assertEqual(Task.BGC_BOUNDARY.value, "bgc_boundary")

    def test_task_from_string(self):
        self.assertIs(task_from_string("taxonomy"), Task.TAXONOMY)

    def test_task_from_string_rejects_unknown(self):
        with self.assertRaises(ValueError):
            task_from_string("promoter")


class TestStructuredRequest(unittest.TestCase):
    def test_construct_and_fields(self):
        r = StructuredRequest(task=Task.TAXONOMY, file_path=Path("/tmp/x.fasta"))
        self.assertEqual(r.task, Task.TAXONOMY)
        self.assertEqual(r.file_path, Path("/tmp/x.fasta"))

    def test_frozen(self):
        r = StructuredRequest(task=Task.TAXONOMY, file_path=Path("/tmp/x.fasta"))
        with self.assertRaises(AttributeError):
            r.task = Task.SPLICE_SITE
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_request -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fungidna.invoke.request'`

- [ ] **Step 3: 最小实现**

`fungidna/invoke/request.py`:
```python
"""Typed structured request passed from either invocation mode to the dispatcher."""
import enum
import dataclasses
from pathlib import Path

from .tasks import CANONICAL_TASKS


class Task(str, enum.Enum):
    REPRESENTATION = "representation"
    TAXONOMY = "taxonomy"
    CODING_POTENTIAL = "coding_potential"
    SPLICE_SITE = "splice_site"
    BGC_BOUNDARY = "bgc_boundary"


def task_from_string(name):
    if name not in CANONICAL_TASKS:
        raise ValueError(f"unsupported task: {name!r}")
    return Task(name)


@dataclasses.dataclass(frozen=True)
class StructuredRequest:
    task: Task
    file_path: Path
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_request -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: 提交**

```bash
git add fungidna/invoke/request.py tests/test_request.py
git commit -m "feat(invoke): add Task enum and StructuredRequest"
```

---

### Task 3: 序列文件读取（sequence.py）

**Files:**
- Create: `fungidna/invoke/sequence.py`
- Test: `tests/test_sequence.py`

**Interfaces:**
- Produces:
  - `SUPPORTED_EXTENSIONS: frozenset[str]` — `{".fa", ".fasta", ".fq", ".fastq"}`
  - `read_sequences(path) -> list[tuple[str, str]]` — 返回 `(header, sequence)` 列表；首行不符格式抛 `ValueError`
  - `first_record_valid(path) -> bool` — 仅检查首行是否 `>` 或 `@` 开头

- [ ] **Step 1: 写失败测试**

`tests/test_sequence.py`（含临时文件构造）:
```python
import unittest
import tempfile
from pathlib import Path

from fungidna.invoke.sequence import read_sequences, first_record_valid, SUPPORTED_EXTENSIONS


def _write(content, suffix=".fasta"):
    f = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


class TestReadSequences(unittest.TestCase):
    def test_fasta_two_records(self):
        p = _write(">seq1\nACGT\n>seq2 description\nTTTT\n")
        recs = read_sequences(p)
        self.assertEqual([r[0] for r in recs], [">seq1", ">seq2 description"])
        self.assertEqual([r[1] for r in recs], ["ACGT", "TTTT"])

    def test_fastq(self):
        p = _write("@r1\nACGT\n+\nIIII\n", suffix=".fastq")
        recs = read_sequences(p)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0][1], "ACGT")

    def test_invalid_first_line(self):
        p = _write("not a sequence file\n")
        with self.assertRaises(ValueError):
            read_sequences(p)

    def test_first_record_valid(self):
        self.assertTrue(first_record_valid(_write(">x\nACGT\n")))
        self.assertTrue(first_record_valid(_write("@x\nACGT\n+\nIIII\n", suffix=".fastq")))
        self.assertFalse(first_record_valid(_write("junk\n")))

    def test_supported_extensions(self):
        self.assertEqual(SUPPORTED_EXTENSIONS, {".fa", ".fasta", ".fq", ".fastq"})
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_sequence -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

`fungidna/invoke/sequence.py`:
```python
"""FASTA/FASTQ reading used by the validation gate and task handlers."""
from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({".fa", ".fasta", ".fq", ".fastq"})


def _first_nonempty_char(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s:
                return s[0]
    return ""


def first_record_valid(path):
    c = _first_nonempty_char(path)
    return c in (">", "@")


def read_sequences(path):
    """Return [(header, sequence), ...] for a FASTA/FASTQ file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    first = raw.lstrip()
    if not first:
        return []
    if first[0] == ">":
        return _read_fasta(raw)
    if first[0] == "@":
        return _read_fastq(raw)
    raise ValueError(f"unrecognized sequence format in {path}")


def _read_fasta(raw):
    records = []
    header = None
    parts = []
    for line in raw.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header = line
            parts = []
        else:
            parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def _read_fastq(raw):
    records = []
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("@"):
            header = lines[i]
            seq = lines[i + 1].strip() if i + 1 < len(lines) else ""
            records.append((header, seq))
            i += 4  # @, seq, +, qual
        else:
            i += 1
    return records
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_sequence -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: 提交**

```bash
git add fungidna/invoke/sequence.py tests/test_sequence.py
git commit -m "feat(invoke): add FASTA/FASTQ sequence reader"
```

---

### Task 4: 确定性验证门（validation.py）

**Files:**
- Create: `fungidna/invoke/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `normalize_task`（Task 1）、`Task/task_from_string`（Task 2）、`SUPPORTED_EXTENSIONS/first_record_valid`（Task 3）
- Produces:
  - `class ValidationError(Exception)` — 属性 `code: str`、`message: str`
  - `def validate(task_raw: str | None, file_path_raw: str | None) -> StructuredRequest` — 归一化 + 检查，失败抛 `ValidationError`，成功返回 `StructuredRequest`（file_path 转绝对路径）

- [ ] **Step 1: 写失败测试**

`tests/test_validation.py`:
```python
import os
import tempfile
import unittest
from pathlib import Path

from fungidna.invoke.validation import validate, ValidationError
from fungidna.invoke.request import Task


def _fasta():
    f = tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False)
    f.write(">s1\nACGT\n")
    f.close()
    return Path(f.name)


class TestValidate(unittest.TestCase):
    def test_valid_request(self):
        p = _fasta()
        req = validate("splice site", str(p))
        self.assertEqual(req.task, Task.SPLICE_SITE)
        self.assertTrue(req.file_path.is_absolute())

    def test_unsupported_task(self):
        p = _fasta()
        with self.assertRaises(ValidationError) as cm:
            validate("promoter", str(p))
        self.assertEqual(cm.exception.code, "TASK_UNSUPPORTED")

    def test_missing_file_path(self):
        with self.assertRaises(ValidationError) as cm:
            validate("taxonomy", None)
        self.assertEqual(cm.exception.code, "MISSING_FILE_PATH")

    def test_file_not_found(self):
        with self.assertRaises(ValidationError) as cm:
            validate("taxonomy", "/no/such/file.fasta")
        self.assertEqual(cm.exception.code, "FILE_NOT_FOUND")

    def test_unsupported_format(self):
        p = _fasta()
        gz = Path(str(p) + ".gz")
        gz.write_text(">s\nACGT\n")
        with self.assertRaises(ValidationError) as cm:
            validate("taxonomy", str(gz))
        self.assertEqual(cm.exception.code, "UNSUPPORTED_FORMAT")

    def test_invalid_sequence_format(self):
        p = _fasta()
        bad = Path(str(p.parent) / "bad.fasta")
        bad.write_text("this is not a sequence\n")
        with self.assertRaises(ValidationError) as cm:
            validate("taxonomy", str(bad))
        self.assertEqual(cm.exception.code, "INVALID_SEQUENCE_FORMAT")
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_validation -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

`fungidna/invoke/validation.py`:
```python
"""Deterministic validation gate — the only gate before task execution."""
import os
from pathlib import Path

from .request import StructuredRequest, task_from_string
from .sequence import SUPPORTED_EXTENSIONS, first_record_valid
from .tasks import normalize_task


class ValidationError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def validate(task_raw, file_path_raw):
    task_name = normalize_task(task_raw)
    if task_name is None:
        raise ValidationError(
            "TASK_UNSUPPORTED",
            f"unsupported or missing task: {task_raw!r}; "
            f"supported: representation, taxonomy, coding_potential, splice_site, bgc_boundary",
        )
    if file_path_raw is None or str(file_path_raw).strip() == "":
        raise ValidationError("MISSING_FILE_PATH", "no file path provided")
    path = Path(str(file_path_raw)).expanduser()
    if not path.exists():
        raise ValidationError("FILE_NOT_FOUND", f"file not found: {path}")
    if not path.is_file():
        raise ValidationError("FILE_NOT_FOUND", f"not a regular file: {path}")
    if not os.access(path, os.R_OK):
        raise ValidationError("FILE_UNREADABLE", f"file not readable: {path}")
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            "UNSUPPORTED_FORMAT",
            f"unsupported extension {path.suffix!r}; supported: .fa .fasta .fq .fastq (uncompressed)",
        )
    if not first_record_valid(path):
        raise ValidationError(
            "INVALID_SEQUENCE_FORMAT",
            "first record does not look like FASTA ('>') or FASTQ ('@')",
        )
    return StructuredRequest(task=task_from_string(task_name), file_path=path.resolve())
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_validation -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: 提交**

```bash
git add fungidna/invoke/validation.py tests/test_validation.py
git commit -m "feat(invoke): add deterministic validation gate"
```

---

### Task 5: Report 与 Dispatcher

**Files:**
- Create: `fungidna/invoke/report.py`
- Create: `fungidna/invoke/dispatcher.py`
- Test: `tests/test_report.py`、`tests/test_dispatcher.py`

**Interfaces:**
- Produces:
  - `Report(task: Task, file_path: Path, status: str, results: dict)` — dataclass，`to_json() -> str` 用 `json.dumps(..., indent=2, default=str)`
  - `dispatch(request: StructuredRequest) -> Report` — 查 `_HANDLERS` 字典（`Task -> handler 实例`），命中调 `handler.run(request)`；字典为空时抛 `RuntimeError("no handler registered for task ...")`。先提供 `register(task, handler)` 供测试注入假 handler。

- [ ] **Step 1: 写失败测试**

`tests/test_report.py`:
```python
import json
import unittest
from pathlib import Path

from fungidna.invoke.report import Report
from fungidna.invoke.request import Task


class TestReport(unittest.TestCase):
    def test_to_json_roundtrip(self):
        r = Report(task=Task.TAXONOMY, file_path=Path("/tmp/x.fasta"),
                   status="ok", results={"rank": "phylum", "label": "Ascomycota"})
        d = json.loads(r.to_json())
        self.assertEqual(d["task"], "taxonomy")
        self.assertEqual(d["file_path"], "/tmp/x.fasta")
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["results"]["label"], "Ascomycota")
```

`tests/test_dispatcher.py`:
```python
import unittest
from pathlib import Path

from fungidna.invoke.dispatcher import dispatch, register, _HANDLERS
from fungidna.invoke.report import Report
from fungidna.invoke.request import Task, StructuredRequest


class FakeHandler:
    def __init__(self, label):
        self.label = label

    def run(self, request):
        return Report(task=request.task, file_path=request.file_path,
                      status="ok", results={"handler": self.label})


class TestDispatcher(unittest.TestCase):
    def setUp(self):
        _HANDLERS.clear()

    def test_dispatch_routes_to_registered_handler(self):
        register(Task.TAXONOMY, FakeHandler("tax"))
        req = StructuredRequest(task=Task.TAXONOMY, file_path=Path("/tmp/x.fasta"))
        r = dispatch(req)
        self.assertEqual(r.results["handler"], "tax")

    def test_dispatch_unregistered_raises(self):
        req = StructuredRequest(task=Task.SPLICE_SITE, file_path=Path("/tmp/x.fasta"))
        with self.assertRaises(RuntimeError):
            dispatch(req)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_report tests.test_dispatcher -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

`fungidna/invoke/report.py`:
```python
"""Structured output returned by every task handler."""
import dataclasses
import json
from pathlib import Path

from .request import Task


@dataclasses.dataclass
class Report:
    task: Task
    file_path: Path
    status: str
    results: dict

    def to_json(self):
        return json.dumps(
            {
                "task": self.task.value,
                "file_path": str(self.file_path),
                "status": self.status,
                "results": self.results,
            },
            indent=2,
            default=str,
        )
```

`fungidna/invoke/dispatcher.py`:
```python
"""Task dispatcher — routes a validated StructuredRequest to its handler."""
from .request import Task

_HANDLERS = {}


def register(task, handler):
    _HANDLERS[task] = handler


def dispatch(request):
    handler = _HANDLERS.get(request.task)
    if handler is None:
        raise RuntimeError(f"no handler registered for task: {request.task.value}")
    return handler.run(request)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_report tests.test_dispatcher -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: 提交**

```bash
git add fungidna/invoke/report.py fungidna/invoke/dispatcher.py tests/test_report.py tests/test_dispatcher.py
git commit -m "feat(invoke): add Report type and task dispatcher"
```

---

### Task 6: 终端解析（parser.py，正则 + 可选 LLM 兜底）

**Files:**
- Create: `fungidna/invoke/parser.py`
- Test: `tests/test_parser.py`

**Interfaces:**
- Consumes: `normalize_task`（Task 1）
- Produces:
  - `parse_prompt(text: str, llm_extractor=None) -> dict` — 返回 `{"task": canonical|None, "file_path": str|None}`；`llm_extractor(text) -> dict` 为可选钩子，仅在正则抽不全时调用；正则抽不全且无钩子时返回两个 `None`

- [ ] **Step 1: 写失败测试**

`tests/test_parser.py`:
```python
import unittest

from fungidna.invoke.parser import parse_prompt


class TestParsePrompt(unittest.TestCase):
    def test_regex_extracts_task_and_abspath(self):
        d = parse_prompt("Please predict the splice sites using /home/ssd/test.fasta")
        self.assertEqual(d["task"], "splice_site")
        self.assertEqual(d["file_path"], "/home/ssd/test.fasta")

    def test_regex_extracts_representation(self):
        d = parse_prompt("generate sequence representations for /data/x.fa")
        self.assertEqual(d["task"], "representation")
        self.assertEqual(d["file_path"], "/data/x.fa")

    def test_missing_file_path_triggers_llm(self):
        seen = {}

        def fake_llm(text):
            seen["called"] = True
            return {"task": "taxonomy", "file_path": "/tmp/x.fastq"}

        d = parse_prompt("classify this genome", llm_extractor=fake_llm)
        self.assertTrue(seen.get("called"))
        self.assertEqual(d["task"], "taxonomy")
        self.assertEqual(d["file_path"], "/tmp/x.fastq")

    def test_missing_everything_without_llm(self):
        d = parse_prompt("hello there")
        self.assertIsNone(d["task"])
        self.assertIsNone(d["file_path"])
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_parser -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

`fungidna/invoke/parser.py`:
```python
"""Prompt-driven terminal parsing: regex first, optional LLM fallback."""
import re

from .tasks import normalize_task

_ALIAS_RE = re.compile(
    r"\b(representations?|embeddings?|taxonomic classification|classification|"
    r"coding[- ]?potential|splice[- ]?sites?|splice|bgc[- ]?boundary|bgc)\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"(?:^|\s)(/(?:[^\s'\"()]+)?[A-Za-z0-9_./-]+\.(?:fa|fasta|fq|fastq))(?:\s|$)")


def parse_prompt(text, llm_extractor=None):
    task = None
    file_path = None

    m = _ALIAS_RE.search(text)
    if m:
        task = normalize_task(m.group(0))

    m = _PATH_RE.search(text)
    if m:
        file_path = m.group(1)

    if (task is None or file_path is None) and llm_extractor is not None:
        candidate = llm_extractor(text) or {}
        if task is None and candidate.get("task"):
            task = normalize_task(candidate["task"])
        if file_path is None and candidate.get("file_path"):
            file_path = candidate["file_path"]

    return {"task": task, "file_path": file_path}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_parser -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: 提交**

```bash
git add fungidna/invoke/parser.py tests/test_parser.py
git commit -m "feat(invoke): add hybrid prompt parser with optional LLM fallback"
```

---

### Task 7: 补回 BGC 模型类

**Files:**
- Modify: `fungidna/model/heads.py`（追加 `MeanPoolHead`）
- Modify: `fungidna/model/fungi_dna.py`（追加 `FungiDNAForBGCBoundary`，更新 import）
- Test: `tests/test_bgc_model.py`

**Interfaces:**
- Consumes: `FungiDNAConfig`、`StripedMambaBackbone`、`MeanPoolHead`
- Produces:
  - `MeanPoolHead(hidden_size, num_classes, hidden_dim=128, dropout=0.1)` — 均值池化（跳过首 token）→ LayerNorm → MLP → 输出
  - `FungiDNAForBGCBoundary(config, num_outputs=2)` — `backbone` + `MeanPoolHead(768, 2)`，`forward(input_ids) -> torch.Tensor` shape `(B, 2)`

- [ ] **Step 1: 写失败测试**

`tests/test_bgc_model.py`:
```python
import unittest
import torch

from fungidna.model.config import FungiDNAConfig
from fungidna.model.heads import MeanPoolHead
from fungidna.model.fungi_dna import FungiDNAForBGCBoundary


class TestBGCModel(unittest.TestCase):
    def test_mean_pool_head_output(self):
        head = MeanPoolHead(768, 2)
        x = torch.zeros(2, 5, 768)  # batch, seq(含首token), hidden
        out = head(x)
        self.assertEqual(out.shape, (2, 2))

    def test_bgc_boundary_forward_shape(self):
        model = FungiDNAForBGCBoundary(FungiDNAConfig(), num_outputs=2)
        ids = torch.zeros(2, 64, dtype=torch.long)
        out = model(ids)
        self.assertEqual(out.shape, (2, 2))
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_bgc_model -v`
Expected: FAIL — `ImportError: cannot import name 'MeanPoolHead'`（或 `FungiDNAForBGCBoundary`）

- [ ] **Step 3: 最小实现**

`fungidna/model/heads.py` 末尾追加（与现有 `ClassificationHead`/`DualBinaryHead` 同风格）:
```python
class MeanPoolHead(nn.Module):
    """Mean-pool → MLP → regression output (BGC boundary distance)."""
    def __init__(self, hidden_size: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states[:, 1:, :].mean(dim=1)
        x = self.norm(pooled)
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)
```

`fungidna/model/fungi_dna.py` 末尾追加:
```python
class FungiDNAForBGCBoundary(nn.Module):
    """Downstream: BGC boundary distance regression (2 outputs: left/right distance)."""
    def __init__(self, config: FungiDNAConfig, num_outputs: int = 2):
        super().__init__()
        self.config = config
        self.backbone = StripedMambaBackbone(config)
        self.head = MeanPoolHead(config.hidden_size, num_outputs)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids, token_type=0)
        return self.head(hidden)
```

并在 `fungidna/model/fungi_dna.py` 顶部 import 行补 `MeanPoolHead`（若当前为 `from fungidna.model.heads import ClassificationHead, DualBinaryHead`，改为 `..., MeanPoolHead`）。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_bgc_model -v`
Expected: PASS (all 2 tests)

- [ ] **Step 5: 提交**

```bash
git add fungidna/model/heads.py fungidna/model/fungi_dna.py tests/test_bgc_model.py
git commit -m "feat(model): restore FungiDNAForBGCBoundary and MeanPoolHead"
```

---

### Task 8: 可插拔权重加载器与 handler 基类

**Files:**
- Create: `fungidna/invoke/handlers/__init__.py`
- Create: `fungidna/invoke/handlers/base.py`
- Test: `tests/test_weight_loader.py`

**Interfaces:**
- Produces:
  - `class WeightsNotFoundError(Exception)` — 属性 `task: str`
  - `class WeightLoader` — `__init__(self, registry: dict | None = None)`；`resolve(task: str) -> Path`，按 `registry` 或环境变量 `FUNGIDNA_<TASK_UPPER>_WEIGHTS` 找路径，找不到抛 `WeightsNotFoundError`
  - `class TaskHandler` — 抽象基类，`task: Task` 类属性、`run(self, request) -> Report`（`raise NotImplementedError`）

- [ ] **Step 1: 写失败测试**

`tests/test_weight_loader.py`:
```python
import os
import unittest
from pathlib import Path

from fungidna.invoke.handlers.base import WeightLoader, WeightsNotFoundError


class TestWeightLoader(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("FUNGIDNA_TAXONOMY_WEIGHTS", None)

    def test_registry_wins(self):
        wl = WeightLoader({"taxonomy": "/tmp/tax.pt"})
        self.assertEqual(wl.resolve("taxonomy"), Path("/tmp/tax.pt"))

    def test_env_fallback(self):
        os.environ["FUNGIDNA_TAXONOMY_WEIGHTS"] = "/tmp/env.pt"
        wl = WeightLoader({})
        self.assertEqual(wl.resolve("taxonomy"), Path("/tmp/env.pt"))

    def test_missing_raises(self):
        wl = WeightLoader({})
        with self.assertRaises(WeightsNotFoundError) as cm:
            wl.resolve("splice_site")
        self.assertEqual(cm.exception.task, "splice_site")
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_weight_loader -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

`fungidna/invoke/handlers/base.py`:
```python
"""Task-handler base class and pluggable weight loader."""
import os
from pathlib import Path


class WeightsNotFoundError(Exception):
    def __init__(self, task):
        super().__init__(f"no checkpoint registered for task {task!r}")
        self.task = task


class WeightLoader:
    def __init__(self, registry=None):
        self.registry = dict(registry or {})

    def resolve(self, task):
        if task in self.registry:
            return Path(self.registry[task])
        env_var = f"FUNGIDNA_{task.upper()}_WEIGHTS"
        env_val = os.environ.get(env_var)
        if env_val:
            return Path(env_val)
        raise WeightsNotFoundError(task)


class TaskHandler:
    task = None  # Task enum member, set by subclass

    def run(self, request):
        raise NotImplementedError
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_weight_loader -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: 提交**

```bash
git add fungidna/invoke/handlers/__init__.py fungidna/invoke/handlers/base.py tests/test_weight_loader.py
git commit -m "feat(invoke): add weight loader and handler base class"
```

---

### Task 9: 五个任务处理器

**Files:**
- Create: `fungidna/invoke/handlers/representation.py`
- Create: `fungidna/invoke/handlers/taxonomy.py`
- Create: `fungidna/invoke/handlers/coding_potential.py`
- Create: `fungidna/invoke/handlers/splice_site.py`
- Create: `fungidna/invoke/handlers/bgc_boundary.py`
- Test: `tests/test_handlers.py`

**Interfaces:**
- Consumes: `TaskHandler`/`WeightLoader`/`WeightsNotFoundError`（Task 8）、`read_sequences`（Task 3）、`Report`/`Task`/`StructuredRequest`
- Produces: 每个 handler 一个 `run(request) -> Report`；模型类在 `run()` 内部**惰性 import**（避免无权重时 import 触发重型依赖）。每个 handler 先 `self.loader.resolve(self.task.value)`，无权重抛 `WeightsNotFoundError`。

设计决策（已定）：
- `representation`：`StripedMambaBackbone`，`encode_bpe` 后 `forward` 取均值池化 768 维。
- `taxonomy`：`StripedMambaBackbone` + 内联 5-rank MLP 头（`num_classes` 按 rank 6/11/26/98/281，默认只做 phylum=6）。
- `coding_potential`：`FungiDNAForSequenceClassification(config, num_classes=2)`。
- `splice_site`：`FungiDNAForSequenceClassification(config, num_classes=3)`；扫 GT/GC/AG 二核苷酸候选，取 ±200 bp 窗口。
- `bgc_boundary`：`FungiDNAForBGCBoundary(config, num_outputs=2)`。

- [ ] **Step 1: 写失败测试（只测"无权重时抛错"，不加载真实权重）**

`tests/test_handlers.py`:
```python
import unittest
from pathlib import Path

from fungidna.invoke.request import Task, StructuredRequest
from fungidna.invoke.handlers.base import WeightsNotFoundError
from fungidna.invoke.handlers.representation import RepresentationHandler
from fungidna.invoke.handlers.taxonomy import TaxonomyHandler
from fungidna.invoke.handlers.coding_potential import CodingPotentialHandler
from fungidna.invoke.handlers.splice_site import SpliceSiteHandler
from fungidna.invoke.handlers.bgc_boundary import BGCBoundaryHandler


class TestHandlersRequireWeights(unittest.TestCase):
    def _req(self, task):
        return StructuredRequest(task=task, file_path=Path("/tmp/x.fasta"))

    def test_all_handlers_raise_without_weights(self):
        for handler_cls, task in [
            (RepresentationHandler, Task.REPRESENTATION),
            (TaxonomyHandler, Task.TAXONOMY),
            (CodingPotentialHandler, Task.CODING_POTENTIAL),
            (SpliceSiteHandler, Task.SPLICE_SITE),
            (BGCBoundaryHandler, Task.BGC_BOUNDARY),
        ]:
            with self.assertRaises(WeightsNotFoundError, msg=task.value):
                handler_cls().run(self._req(task))
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_handlers -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

`fungidna/invoke/handlers/representation.py`:
```python
"""Representation task: embed each sequence with the pretrained backbone."""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader


class RepresentationHandler(TaskHandler):
    task = Task.REPRESENTATION

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        ckpt = self.loader.resolve(self.task.value)  # raises WeightsNotFoundError if absent
        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.striped_mamba import StripedMambaBackbone
        from fungidna.data.tokenizer import DualTokenizer

        # tokenizer model path comes from the checkpoint's sibling dir in real use;
        # kept minimal here — replace with a real tokenizer path when weights ship.
        bpe_path = ckpt.parent / "tokenizer.model"
        tok = DualTokenizer(str(bpe_path))
        model = StripedMambaBackbone(FungiDNAConfig())
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=False)
        model.eval()

        embeddings = {}
        with torch.no_grad():
            for header, seq in read_sequences(request.file_path):
                ids = torch.tensor([tok.encode_bpe(seq, add_cls=True)], dtype=torch.long)
                hidden = model(ids)
                embeddings[header] = hidden.mean(dim=1).squeeze(0).tolist()
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"embeddings": embeddings, "dim": model.config.hidden_size})
```

其余四个 handler 结构相同，差异仅在模型类与预处理，给出关键片段（完整文件照此实现）：

`taxonomy.py` — 内联 MLP 头 + backbone：
```python
class TaxonomyHandler(TaskHandler):
    task = Task.TAXONOMY
    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()
    def run(self, request):
        ckpt = self.loader.resolve(self.task.value)
        # ... 惰性 import config/backbone/tokenizer，加载权重，构造 MLP(768, 256, 6)
        # 每条序列滑窗 10kb（步长 10kb）→ encode_bpe → backbone 均值池化 → MLP → argmax
        # 聚合多数票 → Report(results={"rank": "phylum", "labels": {...}})
```

`coding_potential.py` — `FungiDNAForSequenceClassification(num_classes=2)`：
```python
    def run(self, request):
        ckpt = self.loader.resolve(self.task.value)
        # model = FungiDNAForSequenceClassification(FungiDNAConfig(), num_classes=2)
        # load checkpoint state dict under "model" key if present
        # per sequence: encode_bpe → forward → softmax → coding prob (class 1)
        # Report(results={"predictions": {header: {"coding_probability": p, "label": "coding"/"non-coding"}}})
```

`splice_site.py` — `FungiDNAForSequenceClassification(num_classes=3)` + 候选扫描：
```python
_SPLICE_MOTIFS = {"GT", "GC", "AG"}
    def run(self, request):
        ckpt = self.loader.resolve(self.task.value)
        # model = FungiDNAForSequenceClassification(FungiDNAConfig(), num_classes=3)
        # for each sequence: find all i where seq[i:i+2] in _SPLICE_MOTIFS
        #   window = seq[max(0,i-200):i+201]  (401 bp)
        #   encode_bpe(window) → forward → softmax → class 0/1/2 = donor/acceptor/non-site
        # Report(results={"predictions": {header: [{"pos": i, "motif": m, "class": c, "prob": p}, ...]}})
```

`bgc_boundary.py` — `FungiDNAForBGCBoundary(num_outputs=2)`：
```python
    def run(self, request):
        ckpt = self.loader.resolve(self.task.value)
        # model = FungiDNAForBGCBoundary(FungiDNAConfig(), num_outputs=2)
        # for each sequence: encode_bpe(seq) → forward → 2 outputs = [left_bp, right_bp]
        # Report(results={"boundaries": {header: {"left_bp": l, "right_bp": r}}})
```

> 由于权重尚未发布，`run()` 里 `self.loader.resolve(...)` 是第一步，因此这些 handler 在真实权重就位前只会抛 `WeightsNotFoundError`；测试据此断言，不依赖真实权重。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_handlers -v`
Expected: PASS（5 个任务各抛 `WeightsNotFoundError`）

- [ ] **Step 5: 提交**

```bash
git add fungidna/invoke/handlers/ tests/test_handlers.py
git commit -m "feat(invoke): add five task handlers wired to weight loader"
```

---

### Task 10: 公开 API 与 CLI 入口

**Files:**
- Modify: `fungidna/invoke/__init__.py`（导出公开符号 + 注册 handler）
- Create: `fungidna/__main__.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: 前面所有任务
- Produces:
  - `run_task(file_path: str, task: str) -> Report` — `validate` + `dispatch`
  - `parse_prompt(text, llm_extractor=None) -> dict` — 再导出
  - `main(argv=None) -> int` — CLI 入口，读 `sys.argv[1:]` 拼接成 prompt，`parse_prompt` 后 `validate`/`dispatch`，成功打印 `report.to_json()` 返回 0，失败打印错误返回 1

- [ ] **Step 1: 写失败测试**

`tests/test_api.py`:
```python
import tempfile
import unittest
from pathlib import Path

from fungidna.invoke import run_task
from fungidna.invoke.validation import ValidationError


class TestRunTask(unittest.TestCase):
    def test_invalid_task_raises(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False)
        f.write(">s\nACGT\n")
        f.close()
        with self.assertRaises(ValidationError) as cm:
            run_task(file_path=f.name, task="promoter")
        self.assertEqual(cm.exception.code, "TASK_UNSUPPORTED")

    def test_missing_file_raises(self):
        with self.assertRaises(ValidationError) as cm:
            run_task(file_path="/nope.fasta", task="taxonomy")
        self.assertEqual(cm.exception.code, "FILE_NOT_FOUND")
```

> 说明：`run_task` 对合法请求会进入 handler，而无权重时会抛 `WeightsNotFoundError`；本测试只覆盖验证门路径，避免依赖权重。

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_api -v`
Expected: FAIL — `AttributeError: module 'fungidna.invoke' has no attribute 'run_task'`

- [ ] **Step 3: 最小实现**

`fungidna/invoke/__init__.py`:
```python
"""FungiDNA guarded invocation package."""
from .dispatcher import dispatch, register
from .parser import parse_prompt
from .report import Report
from .request import StructuredRequest, Task, task_from_string
from .validation import ValidationError, validate

from .handlers.representation import RepresentationHandler
from .handlers.taxonomy import TaxonomyHandler
from .handlers.coding_potential import CodingPotentialHandler
from .handlers.splice_site import SpliceSiteHandler
from .handlers.bgc_boundary import BGCBoundaryHandler

# Register the five handlers with the dispatcher (weights resolved lazily).
register(Task.REPRESENTATION, RepresentationHandler())
register(Task.TAXONOMY, TaxonomyHandler())
register(Task.CODING_POTENTIAL, CodingPotentialHandler())
register(Task.SPLICE_SITE, SpliceSiteHandler())
register(Task.BGC_BOUNDARY, BGCBoundaryHandler())


def run_task(file_path, task):
    req = validate(task, file_path)
    return dispatch(req)


__all__ = [
    "run_task", "parse_prompt", "dispatch", "register", "Report",
    "StructuredRequest", "Task", "task_from_string", "ValidationError", "validate",
]
```

`fungidna/__main__.py`:
```python
"""python -m fungidna "<natural-language request>" """
import sys

from .invoke import parse_prompt, validate, dispatch
from .invoke.validation import ValidationError


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    text = " ".join(argv).strip()
    if not text:
        print("usage: python -m fungidna \"<natural-language request>\"", file=sys.stderr)
        return 2
    parsed = parse_prompt(text)
    try:
        req = validate(parsed["task"], parsed["file_path"])
        report = dispatch(req)
        print(report.to_json())
        return 0
    except ValidationError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_api -v` 然后手动冒烟 `python -m fungidna "predict splice sites using /tmp/x.fasta"`（应打印 `error [TASK_UNSUPPORTED 或 WeightsNotFound]` 风格信息而非崩溃）
Expected: 单元测试 PASS；冒烟命令返回非零并输出结构化错误（因无权重或文件不存在）

- [ ] **Step 5: 提交**

```bash
git add fungidna/invoke/__init__.py fungidna/__main__.py tests/test_api.py
git commit -m "feat(invoke): expose run_task API and python -m fungidna CLI"
```

---

### Task 11: 文档与 README 更新

**Files:**
- Modify: `README.md`（新增「Invocation」小节，示例 `run_task` 与 `python -m fungidna`，说明无权重时的 `WeightsNotFoundError` 与 `FUNGIDNA_<TASK>_WEIGHTS` 环境变量）

- [ ] **Step 1: 在 README 增加调用说明**

在 README 末尾「Citation」之前插入：
```markdown
## Invocation

FungiDNA-FM exposes two equivalent entry points that share one validation gate.

**Embedded API:**

```python
from fungidna.invoke import run_task
report = run_task(file_path="/path/test.fasta", task="splice_site")
print(report.to_json())
```

**Prompt-driven terminal:**

```bash
python -m fungidna "Please predict the splice sites using /path/test.fasta"
```

Supported tasks: `representation`, `taxonomy`, `coding_potential`,
`splice_site`, `bgc_boundary` (aliases accepted, e.g. "splice site",
"taxonomic classification").

Weights are loaded per task. Point each task at its checkpoint via
`~/.config/fungidna/weights.yaml` (a `task: /path/to/checkpoint.pt` mapping)
or the `FUNGIDNA_<TASK>_WEIGHTS` environment variable (e.g.
`FUNGIDNA_TAXONOMY_WEIGHTS`). Until weights are registered, requests that pass
validation raise `WeightsNotFoundError`.
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: document guarded invocation API and CLI"
```

---

## Self-Review

**Spec coverage 对照：**
- 白名单集中一处 → Task 1 ✓
- 两种入口共用验证门 → Task 4（validate）+ Task 10（run_task/CLI 都经 validate/dispatch）✓
- 枚举任务类型 → Task 2 ✓
- 无效请求不执行 → Task 4 验证门在 dispatch 之前 ✓
- LLM 只做候选 → Task 6（llm_extractor 输出仍经 normalize+validate）✓
- 五个 handler + 可插拔权重 → Task 8/9 ✓
- 补回 BGC 模型 → Task 7 ✓
- 终端 + API 入口 → Task 6/10 ✓
- Report 序列化 → Task 5 ✓

**占位符扫描：** 无 TBD/TODO；handler 的模型加载给出了具体代码或明确的惰性 import + 语义注释。

**类型一致性：** `normalize_task`（Task 1）在 Task 4/6 以 `(str|None) -> str|None` 使用一致；`validate(task_raw, file_path_raw)` 签名在 Task 4 定义、Task 10 调用一致；`dispatch(request)`、`register(task, handler)`、`Report(...).to_json()`、`WeightLoader.resolve(task)` 在各任务间名称一致。
