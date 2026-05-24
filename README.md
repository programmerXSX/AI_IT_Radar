# AI IT Radar

为内部 AI Lab 打造的常态化"技术情报雷达"。持续扫描 AI 社区动态（arXiv / GitHub Trending / HuggingFace Trending），结合 Lab 兴趣画像分流候选项，用 Critic 校验的 LLM 评测打分，每周生成"强烈推荐 / 关注 / 观望"分级周报，并通过反馈闭环不断演化画像 —— 全部跑在 LangGraph 多 Agent 流水线 + 三层记忆系统之上。

## 核心特性

- **多 Agent (LangGraph)** — Scout / Triage / Analyst / Evaluator / Reporter 五节点流水线，自带 checkpoint
- **RAG 三处召回** — 向量去重、类比检索、评分锚点
- **L1 声明式 Harness** — YAML EvalSpec + Critic 二次校验 + 配对比较 + 黄金集回归
- **三层记忆** — 短期 LangGraph checkpoint、长期 SQLite + ChromaDB、价值层 Lab Profile
- **反馈闭环** — Adopt / Watch / Ignore 三档反馈周期更新画像，Ignore 进入反向 centroid 过滤
- **零基础设施** — 文件型 SQLite + ChromaDB，单机部署即可

## 快速上手

需要 [uv](https://docs.astral.sh/uv/) 管理环境（一次性安装）：

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

然后：

```bash
uv sync                                 # 装依赖
cp .env.example .env                    # 配 API key（DeepSeek / DashScope）
uv run radar init                       # 初始化 + 灌黄金种子
uv run radar run --source arxiv         # 跑一轮
uv run radar feedback-server            # 起反馈 Web (http://127.0.0.1:8765/latest)
```

完整部署/调优/排错见 **[docs/operations.md](docs/operations.md)**。

## 文档

完整文档在 **[`docs/`](docs/)**：

| 文档 | 内容 |
|---|---|
| **[docs/README.md](docs/README.md)** | 文档索引（按角色 / 目的导航） |
| **[docs/architecture.md](docs/architecture.md)** | 系统架构 + 9 张 Mermaid 图（C4 / Agent 拓扑 / 记忆 / RAG / 反馈 / Harness / 数据模型 / 配置 / 模块依赖） |
| **[docs/design.md](docs/design.md)** | 12 个关键设计决策（ADR 风格 — context / decision / consequences / alternatives）+ 已知局限 |
| **[docs/operations.md](docs/operations.md)** | 9 节运维手册：部署 / 配置 / CLI / 调优 / 排错 / 成本 / 备份 / cheatsheet |
| **[docs/extending.md](docs/extending.md)** | 4 类扩展场景：加源 / 加评测维度 / 加 Agent / 加报告格式 |

第一次阅读建议按"30 分钟接手"路径：[docs/README.md → 推荐阅读顺序](docs/README.md#推荐阅读顺序)。

## 架构一览

```
                       ┌─────────────────────────────────────┐
                       │          决策者 (Lab 5-8 人)         │
                       └─────────────────────────────────────┘
                                  ▲                ▼
                                  │    Adopt / Watch / Ignore
                                  │                │
arXiv ──┐                   ┌─────┴────────┐       │
GitHub ─┼─►  Scout ─► Triage ─► Analyst ─► Evaluator ─► Reporter
HF ─────┘   (采集)  (去重画像)  (结构化抽取)  (5 维评分+Critic)  (分级周报)
                       │           │           │            │
                       └───────────┴──────┬────┘            │
                                          ▼                 ▼
                            ┌──────────────────────┐  reports/latest.html
                            │   ChromaDB + SQLite   │  reports/latest.md
                            │       LabProfile       │
                            └──────────────────────┘
```

详细图见 [docs/architecture.md](docs/architecture.md)。

## 配置文件

- [config/sources.yaml](config/sources.yaml) — 三大源开关、关键词、配额
- [config/lab_profile.yaml](config/lab_profile.yaml) — Lab 兴趣 anchors（人工锚点）+ auto_learned（反馈学习）
- [config/eval_specs/](config/eval_specs/) — 5 个评测维度的声明式 rubric + prompt 模板

## 项目布局

```
src/ai_it_radar/
  agents/       Scout / Triage / Analyst / Evaluator / Reporter 节点
  sources/      arxiv / github_trending / huggingface 适配器
  memory/       short_term (checkpoint) / kb (sqlite+chroma) / profile
  rag/          embedder / indexer / retriever / reranker
  harness/      eval_spec / critic / pairwise / regression
  reporter/     Jinja2 模板 + 渲染 + FastAPI 反馈服务
  feedback/     store / profile_updater / ignore_filter
  graph.py      LangGraph 拓扑装配
  cli.py        Typer CLI
  scheduler.py  APScheduler 调度
  schemas.py    Pydantic 数据模型
  settings.py   Pydantic-Settings 配置
  llm.py        LLM 提供商抽象
config/         YAML 配置（sources / eval_specs / lab_profile）
docs/           本项目完整文档（架构 / 设计 / 运维 / 扩展）
tests/
  golden/       24 条黄金集种子 (回归用)
  manual/       check_keys.py 等手工验证脚本
data/           运行时产物：radar.sqlite + chroma/ + checkpoints.sqlite
reports/        生成的 HTML/Markdown 周报
```

## 常用命令

```bash
uv run radar run --source arxiv         # 跑单源
uv run radar run --force                # 强制重评（绕去重）
uv run radar report --rebuild           # 仅从 KB 重渲染报告
uv run radar update-profile --lookback 30  # 反馈 → 画像
uv run radar regression                 # 黄金集回归测试
uv run radar ignore-stats               # 看 IgnoreFilter 状态
uv run radar feedback-server            # 起反馈 Web
uv run radar schedule                   # 起 cron 调度（前台）
```

完整 CLI 参考：[docs/operations.md §四](docs/operations.md#四所有-cli-命令)。

## License

Internal project — TBD.
