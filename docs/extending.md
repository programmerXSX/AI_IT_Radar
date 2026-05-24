# 扩展指南 — AI IT Radar

本文回答"我要给雷达加新东西，怎么动手"。

四个最常见的扩展场景：

1. [加一个新信息源](#一加一个新信息源)
2. [加一个评测维度](#二加一个评测维度)
3. [加一个新 Agent](#三加一个新-agent)
4. [加一种报告格式](#四加一种报告格式)

每个场景都给出"改哪几个文件 + 完整代码骨架 + 测试方法"。

---

## 一、加一个新信息源

### 场景示例：知乎技术专栏 RSS、PapersWithCode、Reddit r/MachineLearning

### 1.1 改哪几处

```
src/ai_it_radar/sources/<your_source>.py    # 新建：实现 SourceAdapter
src/ai_it_radar/sources/__init__.py         # 注册到 all_sources()
src/ai_it_radar/schemas.py                  # 加 SourceKind 枚举值
config/sources.yaml                          # 加配置块
```

### 1.2 实现 SourceAdapter

参照 [src/ai_it_radar/sources/base.py](../src/ai_it_radar/sources/base.py)：

```python
# src/ai_it_radar/sources/papers_with_code.py
from __future__ import annotations
import logging
from typing import Iterable

import httpx

from ..schemas import Candidate, CandidateKind, SourceKind
from .base import SourceAdapter

log = logging.getLogger(__name__)


class PapersWithCodeSource(SourceAdapter):
    name = "papers_with_code"

    def fetch(self) -> Iterable[Candidate]:
        cfg = self.config
        if not cfg.get("enabled", False):
            return []

        max_per_run = int(cfg.get("max_per_run", 20))
        url = "https://paperswithcode.com/api/v1/papers/?ordering=-published"

        out: list[Candidate] = []
        try:
            r = httpx.get(url, timeout=30.0)
            r.raise_for_status()
            for item in r.json().get("results", [])[:max_per_run]:
                uid = f"pwc:{item['id']}"
                out.append(Candidate(
                    uid=uid,
                    source=SourceKind.OTHER,                  # 或新增枚举
                    kind=CandidateKind.PAPER,
                    title=item["title"],
                    url=item.get("url_pdf") or item.get("url_abs", ""),
                    summary=item.get("abstract", "")[:1500],
                    content=item.get("abstract", ""),
                    authors=item.get("authors", []),
                    metadata={
                        "stars": item.get("stars", 0),
                        "tasks": item.get("tasks", []),
                        "code_url": item.get("url_code"),
                    },
                ))
        except Exception as e:
            log.warning("PapersWithCode fetch failed: %s", e)
        return out
```

### 1.3 注册到 `all_sources()`

```python
# src/ai_it_radar/sources/__init__.py
from .papers_with_code import PapersWithCodeSource

def all_sources() -> dict[str, type[SourceAdapter]]:
    return {
        "arxiv": ArxivSource,
        "github_trending": GitHubTrendingSource,
        "huggingface": HuggingFaceSource,
        "papers_with_code": PapersWithCodeSource,    # 加这行
    }
```

### 1.4 （可选）加 SourceKind 枚举

如果想让 Triage / 报告里区分这个源，在 [schemas.py](../src/ai_it_radar/schemas.py) 加：

```python
class SourceKind(str, Enum):
    ARXIV = "arxiv"
    GITHUB = "github"
    HUGGINGFACE = "huggingface"
    PAPERS_WITH_CODE = "papers_with_code"   # 新加
    OTHER = "other"
```

### 1.5 配置 `config/sources.yaml`

```yaml
papers_with_code:
  enabled: true
  max_per_run: 20
  ordering: "-published"
```

### 1.6 测试

```powershell
# 先单测一下源能拿到数据
uv run python -c "from ai_it_radar.sources import all_sources; from ai_it_radar.settings import load_sources_config; cfg = load_sources_config(); s = all_sources()['papers_with_code'](cfg.get('papers_with_code', {})); items = list(s.fetch()); print(f'got {len(items)} items'); print(items[0].title if items else 'EMPTY')"

# 再跑完整流水线
uv run radar run --source papers_with_code
```

### 1.7 注意事项

- **`uid` 必须稳定且唯一**：相同候选两次抓取应得到完全一致的 uid（用源原生 ID）
- **`fetch()` 必须容错**：网络挂、API 改字段不能让整个 cycle 崩溃 —— 用 `try/except` + log warning
- **`content` 要喂得够**：Evaluator/Analyst 主要靠 content 工作。如果只能拿到 200 字 summary，请额外抓详情页（参考 [github_trending.py:_enrich_readme](../src/ai_it_radar/sources/github_trending.py)）
- **Rate limit 友好**：自带 delay / 指数退避；arxiv 库默认 `delay_seconds=3`

---

## 二、加一个评测维度

### 场景示例：商业风险（Commercial Risk）、社区活跃度（Community Vitality）

### 2.1 改哪几处

```
config/eval_specs/<your_dim>.yaml    # 新建：声明式 rubric + prompt
config/eval_specs/<old_dim>.yaml     # 调权重，让总和保持合理
```

注意：**完全不需要改任何 Python 代码**。`load_eval_specs()` 自动扫 `config/eval_specs/*.yaml`。这是 declarative harness 的红利。

### 2.2 编写 EvalSpec YAML

```yaml
# config/eval_specs/community_vitality.yaml
id: community_vitality
display_name: "Community Vitality"
weight: 0.10                    # 全部维度 weight 总和不必=1，但相对大小决定影响力
rubric:
  0: "Dead — no commits/issues in last 6 months."
  1: "Niche — single maintainer, few external contributors."
  2: "Emerging — modest activity, < 50 stars/week."
  3: "Healthy — multiple contributors, regular releases."
  4: "Thriving — strong issue triage, active discussions."
  5: "Movement-tier — referenced by other prominent projects/papers."

rag_neighbors_k: 0              # 这维度不需要 RAG 锚点

prompt_template: |
  You are evaluating COMMUNITY VITALITY for an AI/ML candidate.

  ## Candidate
  Title: {{ title }}
  Type: {{ kind }}
  Summary: {{ summary }}

  Content excerpt:
  {{ content }}

  Metadata signals:
  Stars: {{ metadata.get('stargazers_count', 'N/A') }}
  Forks: {{ metadata.get('forks', 'N/A') }}
  Last commit: {{ metadata.get('pushed_at', 'N/A') }}
  Issues open: {{ metadata.get('open_issues', 'N/A') }}

  ## Task
  Score COMMUNITY VITALITY 0-5 using the rubric. Justify by quoting concrete
  README signals or metadata numbers. If metadata is unavailable, infer from content.

  Rubric:
  {% for k, v in rubric.items() %}- {{ k }}: {{ v }}
  {% endfor %}

  Output STRICT JSON:
  {
    "score": <int>, "confidence": <float>,
    "rationale": "<...>", "quote": "<...>"
  }
```

### 2.3 调整其它维度的权重（可选）

打开各 spec 文件，让 `weight` 总体表达你的优先级。当前默认：

| 维度 | weight | 占比 |
|---|---|---|
| novelty | 0.20 | 20% |
| maturity | 0.20 | 20% |
| fit | 0.25 | 25% |
| reproduction_cost | 0.15 | 15% |
| risk | 0.20 | 20% |

加了 community_vitality (0.10) 后，可把 maturity 降到 0.15、novelty 降到 0.15 维持总和接近 1。

### 2.4 跑回归确认没破坏现有评分稳定性

```powershell
uv run radar regression
```

如果 drift 很大，要么是新维度的 weight 太大，要么 prompt 影响了其它维度的"上下文"。一般 weight ≤ 0.1 影响很小。

### 2.5 用 prompt 中可访问的变量

Evaluator 在 [evaluator.py](../src/ai_it_radar/agents/evaluator.py) 里给 Jinja2 喂的变量：

| 变量 | 类型 | 说明 |
|---|---|---|
| `title` | str | candidate.title |
| `kind` | str | "paper" / "repo" / "model" / ... |
| `summary` | str | candidate.summary |
| `content` | str | candidate.content（已截断） |
| `metadata` | dict | candidate.metadata —— 例 GitHub 源的 stargazers_count |
| `lab_profile_summary` | str | LabProfile 摘要 |
| `anchors` | list[dict] | profile anchors |
| `auto_learned` | list[dict] | 反馈学习出的主题 |
| `neighbors` | list[dict] | RAG 召回的同类历史项（含 score） |
| `rubric` | dict[int,str] | 自动注入的 rubric |

### 2.6 注意事项

- **rubric 0-5 区间是约定**：绝对不要改成 0-10 或 1-5，[DimensionScore](../src/ai_it_radar/schemas.py) 限制 `score: int = Field(..., ge=0, le=5)`
- **prompt 要求引用**：`"quote"` 字段是反幻觉的一道关
- **`rag_neighbors_k` 设 0**：表示这维度不需要历史锚点（如 risk / community_vitality 这类自包含维度）
- **添加后必须跑黄金集回归**

---

## 三、加一个新 Agent

### 场景示例：在 Triage 和 Analyst 之间加一个 `Enricher`，专门补全候选的 README/abstract；或在 Evaluator 后加一个 `MarketWatcher`，从外部新闻源补商业上下文。

### 3.1 改哪几处

```
src/ai_it_radar/agents/<your_agent>.py    # 新建 Agent 节点
src/ai_it_radar/agents/__init__.py        # 导出
src/ai_it_radar/graph.py                   # 加 add_node + add_edge
src/ai_it_radar/schemas.py                 # （可选）GraphState 加新字段
```

### 3.2 实现 Agent 节点

每个 Agent 是一个**纯函数 `(state) -> dict[partial_state]`**：

```python
# src/ai_it_radar/agents/enricher.py
from __future__ import annotations
import logging
from typing import Any

from ..schemas import GraphState, Candidate

log = logging.getLogger(__name__)


def enricher_node(state: GraphState, config: dict[str, Any] | None = None) -> dict:
    """Fetch full README/PDF for thin candidates kept by Triage."""
    keep_uids = {t.candidate_uid for t in state.triage if t.decision in ("keep", "explore")}
    enriched: list[Candidate] = []
    for c in state.candidates:
        if c.uid not in keep_uids:
            enriched.append(c)
            continue
        if len(c.content) >= 500:
            enriched.append(c)
            continue
        # 这里做你的补全逻辑（拉详情页、PDF 提取等）
        new_content = _fetch_full(c)
        c2 = c.model_copy(update={"content": new_content})
        enriched.append(c2)

    return {"candidates": enriched}


def _fetch_full(c: Candidate) -> str:
    # ... 实际抓取逻辑
    return c.content
```

**关键约定**：
- 节点签名必须是 `(state: GraphState, config) -> dict`
- 返回 dict 的键必须是 `GraphState` 里的字段名
- LangGraph 用浅合并（默认覆盖）—— 想用 list 累加得用 `Annotated[list, operator.add]`

### 3.3 注册到 graph

打开 [src/ai_it_radar/graph.py](../src/ai_it_radar/graph.py)：

```python
from .agents.enricher import enricher_node

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)
    g.add_node("scout", scout_node)
    g.add_node("triage", triage_node)
    g.add_node("enricher", enricher_node)        # 新加
    g.add_node("analyst", analyst_node)
    g.add_node("evaluator", evaluator_node)
    g.add_node("reporter", reporter_node)

    g.set_entry_point("scout")
    g.add_edge("scout", "triage")
    g.add_edge("triage", "enricher")             # 改
    g.add_edge("enricher", "analyst")            # 改
    g.add_edge("analyst", "evaluator")
    g.add_edge("evaluator", "reporter")
    g.add_edge("reporter", END)

    return g
```

### 3.4 测试

```powershell
uv run radar run --source arxiv -v   # 看日志里有没有 "enricher" 节点的输出
```

### 3.5 高级用法：条件分支

LangGraph 支持 `add_conditional_edges`，举例"低 quality 候选跳过 Analyst 直接 Reporter"：

```python
def route_after_triage(state: GraphState) -> str:
    if not any(t.decision == "keep" for t in state.triage):
        return "reporter"   # 跳过 analyst/evaluator
    return "analyst"

g.add_conditional_edges("triage", route_after_triage, {
    "analyst": "analyst",
    "reporter": "reporter",
})
```

---

## 四、加一种报告格式

### 场景示例：飞书 / 钉钉 卡片、PDF、纯 JSON、邮件 HTML

### 4.1 改哪几处

```
src/ai_it_radar/reporter/templates/<your_template>.j2   # 新建
src/ai_it_radar/agents/reporter.py                       # 加渲染调用
config/sources.yaml 或 .env                              # （可选）配输出目标
```

### 4.2 加 Jinja2 模板

参考现有 [report.html.j2](../src/ai_it_radar/reporter/templates/report.html.j2) / [report.md.j2](../src/ai_it_radar/reporter/templates/report.md.j2)，新建：

```jinja2
{# src/ai_it_radar/reporter/templates/report.feishu.j2 #}
{
  "msg_type": "interactive",
  "card": {
    "header": {"title": {"tag": "plain_text", "content": "AI 雷达周报 {{ report.period_start.strftime('%Y-%m-%d') }} ~ {{ report.period_end.strftime('%Y-%m-%d') }}"}},
    "elements": [
      {% for item in report.strong_recommend %}
      {
        "tag": "div",
        "text": {"tag": "lark_md", "content": "**[{{ item.candidate.title }}]({{ item.candidate.url }})**\n{{ item.score.aggregate | round(2) }} / 5\n{{ item.analysis.method_summary or '' }}"}
      },
      {%- if not loop.last %}{"tag": "hr"},{% endif %}
      {% endfor %}
    ]
  }
}
```

### 4.3 在 Reporter 里调用

打开 [src/ai_it_radar/agents/reporter.py](../src/ai_it_radar/agents/reporter.py)，找到 `_render_report()` 函数，加：

```python
def _render_report(report: Report) -> None:
    # 现有 HTML / MD 渲染 ...
    
    feishu_tpl = env.get_template("report.feishu.j2")
    feishu_payload = feishu_tpl.render(report=report)
    feishu_path = REPORTS_DIR / "latest.feishu.json"
    feishu_path.write_text(feishu_payload, encoding="utf-8")
    log.info("Wrote Feishu card payload to %s", feishu_path)
```

### 4.4 实际推送（飞书示例）

```python
import httpx
import os

webhook = os.getenv("FEISHU_WEBHOOK_URL")
if webhook:
    httpx.post(webhook, content=feishu_payload, headers={"Content-Type": "application/json"})
```

把 `FEISHU_WEBHOOK_URL` 加到 `.env`，需要时启用。

### 4.5 模板里能拿到什么

`report` 是一个 [Report](../src/ai_it_radar/schemas.py) 实例，包含三档：

```python
report.strong_recommend: list[ReportItem]
report.watch:            list[ReportItem]
report.monitor:          list[ReportItem]
```

每个 `ReportItem` 有：
- `candidate.title / url / summary / content / metadata`
- `analysis.method_summary / key_capabilities / dependency_stack / known_limitations / related_uids`
- `score.aggregate / band / needs_human / dimensions[].score / dimensions[].rationale / dimensions[].quote / critic_verdicts[]`

---

## 五、扩展时的通用注意事项

### 5.1 改 schema 一定要小心

`GraphState` / `Candidate` / `Score` 这些是落到数据库和 ChromaDB 的。加字段一般 OK（向后兼容），改字段名/删字段会让历史数据失效，**改前先备份 `data/`**：

```powershell
$stamp = Get-Date -Format "yyyyMMdd"
Compress-Archive -Path data\* -DestinationPath "backups\pre-schema-change-$stamp.zip"
```

### 5.2 改 prompt 必跑回归

任何对 `config/eval_specs/*.yaml` 的修改、任何对 [llm.py](../src/ai_it_radar/llm.py) 的修改，都跑：

```powershell
uv run radar regression
```

drift > tolerance 会非零退出。

### 5.3 加 LLM 调用时的标准模板

参考 [evaluator.py](../src/ai_it_radar/agents/evaluator.py) 和 [analyst.py](../src/ai_it_radar/agents/analyst.py)：

```python
from ..llm import llm_json

result = llm_json(
    user_prompt=prompt,
    model=settings.llm.model,
    temperature=0.2,
    max_tokens=800,
)
```

`llm_json()` 已自动：
- 拼接 Chinese system prompt + 语言提醒
- 解析 JSON（含三级兜底处理坏字符）
- 异常时 raise `LLMError`，不会让节点静默崩

### 5.4 加 embedding 调用

参考 [rag/indexer.py](../src/ai_it_radar/rag/indexer.py)：

```python
from ..rag import get_embedder, embedding_text

embedder = get_embedder()
text = embedding_text(candidate)   # 标准化后的可 embedding 文本
vec = embedder.embed_query(text)   # list[float]
```

### 5.5 加进 KB（向量库 + 结构化库同步）

```python
from ..memory import KnowledgeBase

kb = KnowledgeBase()
kb.upsert_candidate(candidate, embedding=vec)
```

单一入口，避免向量与 SQL 不同步。

### 5.6 不要在 Agent 里直接读 .env

所有配置走 `Settings` 单例：

```python
from ..settings import get_settings
settings = get_settings()
threshold = settings.triage_dedup_threshold
```

而不是 `os.getenv("RADAR_TRIAGE_DEDUP_THRESHOLD")`。

---

## 六、调试工具箱

### 6.1 单独跑某个 Agent 节点

```python
# 在 REPL 或一次性 python 文件
from ai_it_radar.agents.scout import scout_node
from ai_it_radar.schemas import GraphState

state = GraphState(cycle_id="manual-test")
result = scout_node(state, {"configurable": {"sources": ["arxiv"]}})
print(result["candidates"][0].title)
```

### 6.2 看 LangGraph state 演化

跑 `radar -v run` 看每个节点的输入/输出 size。或用 `LANGCHAIN_TRACING_V2=true` + LangSmith key（如有）。

### 6.3 看 Chroma 里有什么

```python
from ai_it_radar.memory import KnowledgeBase
kb = KnowledgeBase()
print(kb.collection.count())
print(kb.collection.peek(5))   # 头 5 个文档
```

### 6.4 看 prompt 实际长什么样

每条 LLM 调用都写到 `eval_traces` 表：

```powershell
sqlite3 data\radar.sqlite "SELECT prompt FROM eval_traces ORDER BY id DESC LIMIT 1" > last_prompt.txt
```

打开看完整发出去的 prompt。

---

## 总结：扩展难度速查

| 扩展类型 | 触及文件数 | 难度 | 是否需改代码 |
|---|---|---|---|
| 加评测维度 | 1（YAML） | ★ | 否 |
| 加报告格式 | 2（模板 + reporter.py） | ★★ | 是 |
| 加新源 | 4 | ★★★ | 是 |
| 加新 Agent | 3-5 | ★★★★ | 是 |
| 改记忆层 | 5+ | ★★★★★ | 是（且需迁移数据） |

**第一次扩展建议**：从"加一个评测维度"起步，零代码改动，只改 YAML。
