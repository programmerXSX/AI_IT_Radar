# 运维手册 — AI IT Radar

本文回答"怎么把它跑起来 / 怎么调优 / 出问题怎么办"。架构 _what_ 见 [architecture.md](architecture.md)，设计 _why_ 见 [design.md](design.md)。

## 目录

- [一、从零启动（首次部署）](#一从零启动首次部署)
- [二、配置详解](#二配置详解)
- [三、日常运行](#三日常运行)
- [四、所有 CLI 命令](#四所有-cli-命令)
- [五、调优指南](#五调优指南)
- [六、故障排查](#六故障排查)
- [七、成本控制](#七成本控制)
- [八、备份与迁移](#八备份与迁移)

---

## 一、从零启动（首次部署）

### 1.1 安装 uv（包管理器）

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.2 拉取项目并装依赖

```powershell
cd E:\LLM\AI_IT_Radar
uv sync
```

uv 会：
1. 自动下载 cpython-3.12 到 `~/AppData/Roaming/uv/python/`
2. 创建 `.venv/` 虚拟环境
3. 解析 [pyproject.toml](../pyproject.toml) → 生成 `uv.lock`
4. 安装 165 个包（约 5-10 分钟，受网速影响）

### 1.3 配置 `.env`

```powershell
Copy-Item .env.example .env
# 用编辑器打开 .env，填入三家 API key
```

最关键的三组 key：

```env
# 主评分 LLM (推荐 DeepSeek)
RADAR_LLM__API_KEY=sk-xxxxxxxxxx
RADAR_LLM__BASE_URL=https://api.deepseek.com/v1
RADAR_LLM__MODEL=deepseek-chat

# Critic LLM (推荐 Qwen，故意不同家族)
RADAR_CRITIC__API_KEY=sk-xxxxxxxxxx
RADAR_CRITIC__BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RADAR_CRITIC__MODEL=qwen-plus

# Embedding (推荐 DashScope)
DASHSCOPE_API_KEY=sk-xxxxxxxxxx
RADAR_EMBEDDING__PROVIDER=dashscope
RADAR_EMBEDDING__MODEL=text-embedding-v3
```

获取 key：
- DeepSeek：https://platform.deepseek.com/api_keys
- DashScope（阿里通义）：https://dashscope.console.aliyun.com/apiKey

可选：
```env
RADAR_GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxx   # 提升 GitHub Search API rate limit
TAVILY_API_KEY=tvly-...                    # 预留给未来网络搜索源
```

### 1.4 烟雾测试（不烧 API token）

```powershell
uv run pytest tests/test_smoke.py -v
```

**预期**：7 个用例全 PASS。验证导入、schema、SQLite 建表、EvalSpec 解析、黄金集加载、画像加载、LangGraph 编译。

### 1.5 三家 LLM 联调（消耗约 ¥0.01）

```powershell
uv run python tests/manual/check_keys.py
```

**预期**：三段绿色 `OK`：
```
─── Embedding (DashScope) ──────
OK  query dim = 1024
OK  batch returned 3 vectors of dim 1024
─── Primary LLM (DeepSeek) ─────
OK  parsed = {'ping': 'pong', 'answer': 42}
─── Critic LLM (Qwen via DashScope) ─
OK  parsed = {'agree': True, 'reason': 'ok'}
All three providers OK.
```

任何一项报 `AuthenticationError`：检查 `.env` 对应 `_API_KEY` 是否填对、cwd 是不是项目根目录。

### 1.6 灌入黄金种子（消耗约 ¥0.01 embedding）

```powershell
uv run radar init
```

**预期**：
```
Initialized: data_dir=data
Seeded 24 golden items into KB
```

此时 `data/` 下应该有 `radar.sqlite` + `chroma/` 两份文件。

### 1.7 第一次真扫一轮（消耗约 ¥0.5）

先把范围缩小避免烧太多：

编辑 [config/sources.yaml](../config/sources.yaml)：

```yaml
arxiv:
  enabled: true
  max_per_run: 5      # 改 30 -> 5
  
github_trending:
  enabled: false      # 暂关

huggingface:
  enabled: false      # 暂关
```

然后跑：

```powershell
uv run radar run --source arxiv
```

预期日志：
```
scout: 5 unique candidates collected
triage: 4 kept (1 explore), 0 duplicate, 1 oos
analyst: produced 4 analyses
evaluator: scored 4 candidates
reporter (period=7d): 1 strong / 2 watch / 1 monitor
Cycle done — strong:1 watch:2 monitor:1
```

### 1.8 起反馈服务

新开 PowerShell：
```powershell
uv run radar feedback-server
```
浏览器打开 http://127.0.0.1:8765/latest 看周报，点 Adopt / Watch / Ignore。

---

## 二、配置详解

### 2.1 配置文件清单

| 文件 | 作用 | 谁会改 |
|---|---|---|
| [`.env`](../.env.example) | API key + 阈值参数 | 部署人员 |
| [`config/sources.yaml`](../config/sources.yaml) | 三大源开关、关键词、配额 | 运维 |
| [`config/eval_specs/*.yaml`](../config/eval_specs/) | 5 个评测维度的 rubric + prompt | prompt 调优时改 |
| [`config/lab_profile.yaml`](../config/lab_profile.yaml) | Lab 兴趣 anchors + auto_learned | 人工写 anchors，`update-profile` 写 auto_learned |

### 2.2 `.env` 全参数表

```env
# ---- LLM 主评分 ----
RADAR_LLM__PROVIDER=openai            # openai-compatible 总入口
RADAR_LLM__MODEL=deepseek-chat
RADAR_LLM__BASE_URL=https://api.deepseek.com/v1
RADAR_LLM__API_KEY=
RADAR_LLM__TEMPERATURE=0.2            # 评分温度低 = 稳定
RADAR_LLM__TIMEOUT_S=60

# ---- Critic ----
RADAR_CRITIC__PROVIDER=openai
RADAR_CRITIC__MODEL=qwen-plus
RADAR_CRITIC__BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RADAR_CRITIC__API_KEY=

# ---- Embedding ----
RADAR_EMBEDDING__PROVIDER=dashscope   # local | openai | dashscope
RADAR_EMBEDDING__MODEL=text-embedding-v3
RADAR_EMBEDDING__API_KEY=             # 留空则读 DASHSCOPE_API_KEY

# ---- Pipeline 阈值 ----
RADAR_TRIAGE_DEDUP_THRESHOLD=0.86          # 越大越严（更难判定为 duplicate）
RADAR_TRIAGE_PROFILE_THRESHOLD=0.32         # 越大越严（更多东西被判为 OOS）
RADAR_TRIAGE_IGNORE_THRESHOLD=0.62          # 越小反向过滤越激进
RADAR_EXPLORATION_BUDGET=0.15               # 0.0-1.0
RADAR_CRITIC_DISAGREEMENT_THRESHOLD=2       # |主-Critic| >= 此值标 needs_human
RADAR_BAND_STRONG_RECOMMEND=3.5             # band 切线
RADAR_BAND_WATCH=2.5

# ---- 杂项 ----
RADAR_DATA_DIR=./data
RADAR_GITHUB_TOKEN=
```

### 2.3 sources.yaml 详解

```yaml
arxiv:
  enabled: true
  categories: [cs.AI, cs.CL, cs.LG, cs.CV]   # 改成你 Lab 关心的方向
  keywords:
    - "large language model"
    - "agent"
  max_per_run: 30      # 单源单次扫多少
  lookback_days: 7     # 只看最近 N 天发的论文

github_trending:
  enabled: true
  languages: [python, ""]   # 空字符串 = 全语言
  since: weekly             # daily | weekly | monthly
  max_per_run: 20
  fallback_search_api: true # HTML 爬挂掉时用 Search API
  search_api_query: "stars:>50 created:>{lookback_iso} topic:llm OR topic:ai"

huggingface:
  enabled: true
  models:
    enabled: true
    sort: trendingScore     # 按热度排
    filter_tags: [text-generation, text2text-generation]
    limit: 25
  datasets:
    enabled: false          # 数据集默认关，开启后限流也大

exploration_budget: 0.15    # 全局，被 Triage 用
```

### 2.4 lab_profile.yaml 详解

```yaml
name: "Internal AI Lab"
description: |
  描述 Lab 的总体方向，会喂给 Evaluator 的 fit 维度 prompt
  
anchors:                    # 人工锚点 - 永不被自动覆盖
  - topic: "LLM serving / inference optimization"
    weight: 1.0
    keywords: [vllm, sglang, kv cache, ...]
  - topic: "Agent frameworks & orchestration"
    weight: 1.0
    keywords: [agent, langgraph, ...]
  # ... 6 个起步锚点

exclude_keywords:           # 命中即在 Triage 阶段直接 OOS
  - blockchain
  - cryptocurrency
  - nft

auto_learned: []            # 不要手编辑！由 radar update-profile 写入
```

**提示**：anchors 越精确，雷达越聚焦。建议初期手写 5-8 个，每月 review 一次。

---

## 三、日常运行

### 3.1 周报模式（推荐）

```powershell
uv run radar schedule
```

前台启动 APScheduler，默认：
- 每周一 09:00 UTC（北京时间下午 5 点）跑 `radar run` 全源
- 每周一 09:30 UTC 跑 `radar update-profile`

可改 cron：
```powershell
uv run radar schedule --scan-cron "0 1 * * MON" --profile-cron "30 1 * * MON"
```

要做成系统服务：用 NSSM (Windows) / systemd (Linux) 把这个进程托管。

### 3.2 手动按需跑

```powershell
# 全源全跑
uv run radar run

# 单源
uv run radar run --source arxiv
uv run radar run --source github_trending
uv run radar run --source huggingface

# 强制重评（绕过 Triage 去重 + 反向过滤）
uv run radar run --source arxiv --force
```

### 3.3 看历史 / 重渲染

```powershell
# 控制台看过去 7 天评分表（不抓源不调 LLM）
uv run radar report --period 7d

# 仅用 KB 现有数据重渲染 HTML/MD
uv run radar report --rebuild

# 或换周期
uv run radar report --period 30d --rebuild
```

### 3.4 反馈服务

```powershell
uv run radar feedback-server               # 默认 :8765
uv run radar feedback-server --port 9000   # 改端口
```

服务起在前台。决策者通过 http://<host>:8765/latest 浏览（建议给 Lab 内部一个固定的内网 URL）。

### 3.5 反馈学习

```powershell
# 一次性手工触发画像更新（一般不需要，schedule 已自动）
uv run radar update-profile --lookback 30
```

### 3.6 看 IgnoreFilter 状态

```powershell
uv run radar ignore-stats
```

输出形如：
```
ignored items in pool: 5
centroid active:       True
ignored uids:
  - arxiv:2605.22823
  - github:foo/bar
Threshold: 0.62.
```

---

## 四、所有 CLI 命令

| 命令 | 用途 | 烧 LLM？ | 备注 |
|---|---|---|---|
| `radar version` | 看版本 | ✗ | |
| `radar init` | 初始化 + 灌种子 | ✗ (仅 24 次 embedding) | 首次必跑 |
| `radar run [--source ...] [--force]` | 跑一个 cycle | ✓ | 主入口 |
| `radar report [--period 7d] [--rebuild]` | 控制台 + 重渲染 | ✗ | 只读 KB |
| `radar regression [--seeds path]` | 跑黄金集回归 | ✓ | prompt 调优后必跑 |
| `radar feedback <uid> <tag>` | 命令行加反馈 | ✗ | 调试 |
| `radar feedback-server` | 起反馈 Web | ✗ | 长驻服务 |
| `radar update-profile --lookback 30` | 反馈 → 画像 | ✓ (1 次 LLM) | schedule 已自动 |
| `radar ignore-stats` | 看 IgnoreFilter | ✗ | |
| `radar schedule` | 起 cron 调度 | - | 定时调上面 |

`-v` / `--verbose` 是**全局开关**，必须放子命令前：
```powershell
uv run radar -v run --source arxiv     # ✓
uv run radar run --source arxiv -v     # ✗ Typer 会报 No such option
```

---

## 五、调优指南

### 5.1 觉得 monitor 档项目太多？

可能是 `triage_profile_threshold` 太低导致很多边缘项目被 KEEP。

```env
RADAR_TRIAGE_PROFILE_THRESHOLD=0.40   # 默认 0.32，调高更严
```

或扩展画像 anchors（让你 Lab 真关心的话题"中心引力"更强）。

### 5.2 觉得 strong_recommend 永远是 0？

可能：
- 5 个评测维度的权重不够分散（看 [config/eval_specs/](../config/eval_specs/) 里的 `weight` 和 `rag_neighbors_k`）
- band 切线太高：`RADAR_BAND_STRONG_RECOMMEND=3.5` 调低到 3.2
- 候选源都不够新颖（arXiv 关键词太宽？把 keywords 收紧到具体话题）

### 5.3 Critic 太爱反对

分歧阈值默认 2，调高到 3 可让"小分歧"不再触发 needs_human：

```env
RADAR_CRITIC_DISAGREEMENT_THRESHOLD=3
```

### 5.4 IgnoreFilter 误伤太多

```env
RADAR_TRIAGE_IGNORE_THRESHOLD=0.70   # 调高 = 更宽容
```

或反过来，没起作用：

```env
RADAR_TRIAGE_IGNORE_THRESHOLD=0.55   # 调低 = 更激进
```

调试：跑 `radar ignore-stats`，再 `radar run --source arxiv`，观察日志中 `out_of_scope` 的 rationale 里是否带 "matches ignored pattern (cos=0.6X >= 0.62)"。

### 5.5 prompt 调优后必须跑回归

```powershell
uv run radar regression
```

输出 `"ok": true, "drift_count": 0` 才算通过。否则按 `failed_items` 的 delta 与 expected_aggregate 微调 prompt 或 tolerance。

### 5.6 评测过程审查

```powershell
sqlite3 data\radar.sqlite "SELECT candidate_uid, dimension_id, raw_response FROM eval_traces ORDER BY id DESC LIMIT 5"
```

每一条都有：
- `prompt`：发给 LLM 的全部内容
- `raw_response`：LLM 原始回答（含理由）
- `critic_response`：Critic 判定

定位"评分为什么打这个数"用这个表。

---

## 六、故障排查

### 问题：`radar` 命令找不到

```
The term 'radar' is not recognized as the name of a cmdlet
```

uv 不会把项目命令注册到全局 PATH。统一用 `uv run radar ...`。

或者激活 venv 后：
```powershell
.\.venv\Scripts\Activate.ps1
radar version
```

### 问题：JSONDecodeError "Invalid \escape"

LLM 偶发输出含野生反斜杠的 JSON。已在 [llm.py](../src/ai_it_radar/llm.py) 加三重兜底：
1. 直接 `json.loads`
2. regex 抓最大 `{...}` 块
3. 替换非合法 `\escape` 后再试

如果还是挂，看错误信息里 "Output (truncated 300 chars)" 部分可定位 LLM 实际返回了什么。

### 问题：第二次跑报告全 0

这是**正常行为**：第二轮所有 candidate 被 Triage uid-dedup 拦下。报告应该从 KB 拉过去 7 天，已修复（[`reporter.py`](../src/ai_it_radar/agents/reporter.py)）。

如果还看到全 0，跑：
```powershell
uv run radar report --rebuild
```

### 问题：反馈按钮点了无响应 / "feedback-server 未启动"

- 必须从 http://127.0.0.1:8765/latest 打开报告（不是 `file://` 双击）
- 检查 feedback-server 进程在跑：`Get-Process | findstr uv`
- 改端口冲突：`uv run radar feedback-server --port 9000`，浏览器对应改

### 问题：Chroma 报 dim mismatch

```
ValueError: Embedding dimension X does not match collection dimension Y
```

中途换了 embedding 模型导致维度变化。最简单：清空 Chroma：

```powershell
Remove-Item -Recurse -Force data\chroma
uv run radar init
```

会重新灌入种子 + 用新模型 embedding。

### 问题：DashScope 报 InvalidApiKey

- `.env` 里 `DASHSCOPE_API_KEY` 必须是阿里云控制台 https://dashscope.console.aliyun.com/apiKey 创建的，不是 deepseek 的
- key 前缀是 `sk-` 但 DeepSeek 也是 `sk-`，确认对应账户

### 问题：scout 拉 arxiv 报 timeout

国外网络问题，把 `lookback_days` 调短减少返回量：
```yaml
arxiv:
  lookback_days: 3
  max_per_run: 10
```
或挂代理（设置 HTTP_PROXY 环境变量）。

### 问题：BGE-M3 下载卡死（local embedder）

切到镜像：
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
uv run radar init
```

或用 DashScope embedding（推荐），完全不下载本地模型。

---

## 七、成本控制

### 单次 cycle 成本估算

| 阶段 | 操作次数 | 单次成本 | 小计 |
|---|---|---|---|
| Scout | API 拉取 | 免费 | ¥0 |
| Triage embedding | N 次（候选数） | ~¥0.001 | ¥0.05 (50 候选) |
| Triage 反向过滤 | N 次 cosine | 免费 | ¥0 |
| Triage LLM dedup | M 次（高重叠候选） | ~¥0.005 | ¥0.05 |
| Analyst | K 次 LLM（保留候选数） | ~¥0.01 | ¥0.30 (30 保留) |
| Evaluator 主评分 | K × 5 次 LLM | ~¥0.01 | ¥1.50 |
| Evaluator Critic | K × 5 次 LLM | ~¥0.005 | ¥0.75 |
| **合计** | | | **~¥2.5** |

每周一次的话每月约 **¥10**。

### 省钱技巧

1. **缩小源配额**：`max_per_run` 改小（最有效）
2. **关 GitHub Trending 的 README enrichment**（[github_trending.py:_enrich_readme](../src/ai_it_radar/sources/github_trending.py)）—— 当前每条都拉 README，省钱版可只拉 description
3. **Critic 用更便宜的模型**：`RADAR_CRITIC__MODEL=qwen-turbo`
4. **降低 `rag_neighbors_k`**：每个 EvalSpec 改 5→3，prompt 短，token 少

### 大事故防御

- DeepSeek / DashScope 都支持余额预警和限额
- 任何 LLM 调用都有 `try/except`，单次失败不会传染整个 cycle
- 跑前先 `radar regression` 验证 prompt 没变质

---

## 八、备份与迁移

### 8.1 关键文件

```
data/radar.sqlite        # 所有候选 / 评分 / 反馈
data/chroma/             # 向量索引（可重建，但费 embedding 调用）
data/checkpoints.sqlite  # LangGraph checkpoint（可丢，仅影响 resume）
config/lab_profile.yaml  # Lab 画像（重要！包含历史学习成果）
.env                     # API key（不要提交！）
```

### 8.2 备份脚本

```powershell
$stamp = Get-Date -Format "yyyyMMdd"
Compress-Archive -Path data\radar.sqlite, data\chroma, config\lab_profile.yaml `
                 -DestinationPath "backups\radar-$stamp.zip"
```

建议每周一次。`reports/` 不需要备份（每次 run 重新生成）。

### 8.3 换机器迁移

```powershell
# 老机器
Compress-Archive data, config\lab_profile.yaml -DestinationPath radar-state.zip

# 新机器
git clone <repo>
cd ai-it-radar
uv sync
Expand-Archive radar-state.zip -DestinationPath .
Copy-Item .env.example .env  # 重新填 key
uv run radar report --rebuild  # 验证数据完整
```

### 8.4 升级依赖

```powershell
uv sync --upgrade           # 升级所有锁定版本到最新允许范围
uv lock --upgrade-package langgraph   # 仅升级单个

# 升级后必跑回归
uv run pytest tests/test_smoke.py -v
uv run radar regression
```

如果 LangGraph / pydantic 大版本更新破坏了 schema，从备份恢复 `data/` 即可，源码层我们已用 strict typing。

---

## 九、Cheatsheet（贴墙用）

```powershell
# 安装
uv sync

# 一次性 bootstrap
uv run radar init

# 跑一轮
uv run radar run --source arxiv

# 报告 + 反馈
uv run radar feedback-server               # 长驻
# 浏览器: http://127.0.0.1:8765/latest

# 反馈 → 画像
uv run radar update-profile --lookback 30

# 重评所有现有候选（绕去重）
uv run radar run --source arxiv --force

# 调试
uv run radar ignore-stats
sqlite3 data\radar.sqlite "SELECT * FROM feedback ORDER BY id DESC LIMIT 10"

# CI / prompt 调优后
uv run radar regression
```
