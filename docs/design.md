# 设计决策文档 — AI IT Radar

本文回答"为什么这么搭"。架构 _what_ 见 [architecture.md](architecture.md)，运维 _how_ 见 [operations.md](operations.md)。

每个决策按 **ADR (Architecture Decision Record)** 风格记录：
- **Context** — 当时面对什么问题
- **Decision** — 选择了什么
- **Consequences** — 带来的好与坏
- **Alternatives** — 拒绝的选项 + 拒绝理由

---

## 1. 编排框架：LangGraph vs CrewAI / AutoGen

**Context**：5 个 Agent 串行的流水线，每个节点的状态需要持久化（Cycle 中途崩溃时不能丢失已抓的 75 条 candidate）。

**Decision**：LangGraph + LangChain 生态。

**Consequences**：
- ✅ `SqliteSaver` checkpoint 是内置能力，5 行代码就能"断点续跑"
- ✅ `StateGraph` 用 Pydantic 模型做 state，类型安全 + IDE 友好
- ✅ 后续可加条件边、subgraph，演化空间充足
- ✅ 社区资料最丰富
- ❌ Pydantic state 的 reducer 偶尔反直觉（默认 overwrite，要 list 累加得 `Annotated[list, operator.add]`）—— 我们用了显式覆写，回避此坑
- ❌ 比"自研 asyncio 状态机"多一层抽象

**Alternatives 拒绝**：
- **CrewAI**：角色化好理解，但流程灵活度低，且"任务依赖"模型与我们"流水线 + 共享记忆"模式不贴合
- **AutoGen / AG2**：擅长 Agent 间辩论，对纯流水线是 overkill；版本分裂（0.2 vs 0.4 vs AG2 fork）增加未来迁移风险
- **自研 asyncio**：完全可控但 checkpoint / 可观测要全部自己造

---

## 2. Harness 深度：L1 静态 vs L2 沙箱 vs L3 真跑 benchmark

**Context**：原始需求"评测其工程价值能力边界"——可以解读为从"读 README 打分"到"真跑 MMLU/HumanEval"的连续光谱。

**Decision**：MVP 锁定 **L1 轻量化** —— 只读 README / abstract / repo 元数据 + LLM 给分，不真跑模型。

**Consequences**：
- ✅ 零基础设施门槛：无需 GPU、无沙箱、无任务调度
- ✅ 跑一周报告成本约 ¥10-30（每个候选 ≈ 10 次 LLM 调用）
- ✅ 单机起步，便于内部 Lab 5-8 人快速上手
- ❌ 给不出"真实跑通率""推理速度"等运行时指标
- ❌ 严重依赖 LLM 对 README 的理解（README 写得不好的项目会被低估）

**Alternatives 拒绝**：
- **L2 代码沙箱**：在 Docker 里 clone + quickstart + smoke test —— 投入产出比要等 MVP 验证 Lab 真在用之后再升
- **L3 真跑 benchmark**：lm-evaluation-harness / OpenCompass —— GPU + 任务队列 + 异常处理是另一个项目的工作量
- **L1+L2 分级触发（Triage 标"高优"才升级）**：是正确的最终形态，但 MVP 不做（升级路径已留）

---

## 3. LLM 提供商组合：DeepSeek 主 + Qwen Critic + DashScope embedding

**Context**：需要主评分模型、Critic 校验模型、embedding 模型三类。预算约束："日常情报雷达每月不应超过百元量级"。

**Decision**：
- 主评分：**DeepSeek (`deepseek-chat`)** — 成本约为 GPT-4o-mini 1/3
- Critic：**Qwen Plus**，走 DashScope OpenAI-compatible mode — **故意与主评分不同模型家族**
- Embedding：**DashScope `text-embedding-v3`**（1024 维，多语言）

**Consequences**：
- ✅ 不同家族的 Critic 能更敏感发现共谋幻觉（"Critic 不敢反驳同家族主评分"是真问题）
- ✅ 三家全 OpenAI-compatible 接口，统一抽象（[`llm.py`](../src/ai_it_radar/llm.py) 只有一个 `ChatOpenAI` 包装）
- ✅ DashScope embedding 对中文理解强，适合未来扩展中文社区源
- ❌ 三个 API key 管理负担
- ❌ 任一家服务故障会影响整个 cycle —— 已有 `try/except` 但未做 fallback 模型

**Alternatives 拒绝**：
- **全用 OpenAI**：Critic 与主评分同家族，会高度同意，Critic 价值打折
- **本地 sentence-transformers (BGE-M3)**：仍保留为 embedding fallback（`RADAR_EMBEDDING__PROVIDER=local`）；不做默认是因为 2.3GB 首次下载劝退
- **完全本地 Ollama**：JSON 输出稳定性差，前期会大量解析失败

---

## 4. 三层记忆系统（vs. 单一向量库 vs. 单一 SQL）

**Context**：系统要同时支持：(a) 单 cycle 崩溃续跑、(b) 跨 cycle 去重 / 类比 / 评分历史、(c) 决策者偏好的演化。

**Decision**：三层显式分离：
- **短期** — LangGraph SqliteSaver checkpoint (cycle 内)
- **长期** — SQLite (结构化) + ChromaDB (向量)
- **价值/偏好** — `lab_profile.yaml` (anchors + auto_learned) + 衍生 centroid

**Consequences**：
- ✅ 每层独立的生命周期与读写模式，性能 / 成本 / 可维护性都最优
- ✅ Profile 用 YAML 是关键 —— 人工锚点可 git diff，自动学习不会偷偷漂移
- ❌ 三层之间需要小心同步（一个 candidate 同时存在 SQLite 和 Chroma）—— 我们用 `kb.upsert_candidate(c, embedding=vec)` 单一入口避免漂移

**Alternatives 拒绝**：
- **只用 ChromaDB**：向量能去重但不擅长结构化 join、统计、反馈追溯
- **只用 SQLite + 自建相似度索引**：BLAS 暴力搜索在 N=数千就够了，但 ChromaDB 的 HNSW 是免费午餐
- **Postgres + pgvector**：要装服务，违反"零运维"基线

---

## 5. RAG 在三处召回（而非单一去重）

**Context**：纯做"看到新候选 → 与历史比 → 重的丢"的去重很容易；但只用这一处，等于浪费了向量库。

**Decision**：在 Triage / Analyst / Evaluator **三处**用 RAG，每处目的不同：
1. Triage：去重（vector + LLM 二次判定）
2. Analyst：召回 3 条相似项作为"对比写作"的输入
3. Evaluator：召回 K 条同类历史评分作为锚点，让 LLM 知道"以往同类我们打几分"

**Consequences**：
- ✅ 评分跨 cycle 一致性显著提升 —— 第 100 条 Llama 变种不会比第 1 条得高出 2 分
- ✅ Analyst 写出来的对比有抓手，不再是空泛的方法描述
- ❌ Cold start 期间（KB 中无同类项）这两条降级为常规打分 —— 黄金种子集 24 条就是为缓解这个问题

**Alternatives 拒绝**：
- **只在 Triage 用**：浪费向量库
- **每个 prompt 都跑 RAG**：边际收益递减，且增加 token 成本

---

## 6. 评测幻觉的三重防线

**Context**：LLM 评分极易"理由编造"——给 5 分但理由跟内容无关。

**Decision**：
1. **强制引用**：每个维度 prompt 要求 `quote: <source content 原文>`，写入 `eval_traces` 可核查
2. **Critic 二次校验**：独立 LLM（不同家族）审核主评分；分歧 ≥2 标 `needs_human=true`
3. **配对比较**（[harness/pairwise.py](../src/ai_it_radar/harness/pairwise.py)）：A vs B 谁更强，比绝对打分更稳定（暂供未来扩展用）

**Consequences**：
- ✅ 引用可核查 —— 任何时候 grep `eval_traces.raw_response` 看 quote 在不在 candidate.content 里
- ✅ Critic 把"3 分但 Critic 建议 0 分"flag 出来，避免直接采信
- ❌ Critic 会让 LLM 调用翻倍（成本 ×2）
- ❌ Critic 偶尔会过度反对，目前用 `disagreement_threshold=2` 隔离极端

**Alternatives 拒绝**：
- **不要 Critic，只信主评分**：成本省一半，但失去"共谋幻觉"防线
- **三个 LLM 投票**：成本 ×3，验证不如 Critic 直接审核有效

---

## 7. 黄金集回归（vs. 不做、做更大）

**Context**：prompt / 模型版本变更会让评分静默漂移 —— 上周给 4.2 的 vllm 这周变 3.1，决策者无从感知。

**Decision**：[tests/golden/seeds.yaml](../tests/golden/seeds.yaml) 24 条人工标注种子覆盖三档极值，`radar regression` 跑全集对比 `expected_aggregate ± tolerance`。

**Consequences**：
- ✅ 每次改 prompt 跑一次，drift > 阈值非零退出 —— 是 CI 友好的硬规则
- ✅ 24 条覆盖了 strong（vllm/Llama3）/ watch（AutoGen/TinyLlama）/ monitor（abandoned-toy）三档极值
- ❌ 24 条不够测细粒度（评 4.0 vs 4.5 区分能力）
- ❌ 当前是合成种子，理想应替换为 Lab 历史采纳/拒绝项

**Alternatives 拒绝**：
- **不做**：prompt 调优会变成手感工程
- **5000 条大集**：维护成本高，每次跑爆 token

---

## 8. 反馈三档（Adopt / Watch / Ignore）vs. 五星打分

**Context**：决策者每周扫几十条候选，需要点击成本极低的反馈机制。

**Decision**：三个按钮：Adopt / Watch / Ignore。

**Consequences**：
- ✅ 决策成本最小化 —— 一秒决定
- ✅ 三档恰好对应"激活正向画像"/"延迟价值信号"/"激活负向 IgnoreFilter"
- ❌ 表达粒度粗 —— "Adopt 但仅作背景了解"和"Adopt 想用于核心项目"区分不出
- 🔮 后续可加 `note` 字段（HTML 已预留）做 few-shot

**Alternatives 拒绝**：
- **0-5 星打分**：决策者懒得想精确分数，会全打 3 星
- **Like / Dislike 两档**：失去"延迟价值"信号（Watch → Adopt 翻转）

---

## 9. IgnoreFilter：反向 centroid（vs. 黑名单）

**Context**：用户点 Ignore 应该让"长得像的"下次也别出现，但又不能 hard-block 一切相似词的项目。

**Decision**：把所有 Ignore 过的 candidate embedding 求平均得反向 centroid；新候选与它的 cosine 相似度 ≥ 阈值（默认 0.62）就丢弃。

**Consequences**：
- ✅ O(1) 一次 cosine 比较，性能等同 profile match
- ✅ 与正向 profile 对称设计，逻辑一致
- ✅ Adopt/Watch 翻转能覆盖之前的 Ignore（latest positive signal wins）
- ❌ 阈值需调（首次部署可能太严或太松）—— 留了 `RADAR_TRIAGE_IGNORE_THRESHOLD` 可调
- ❌ Cold start 期 0 个 Ignore 时不工作 —— 自然如此

**Alternatives 拒绝**：
- **逐项 NN 检查**：N×M 比较成本翻倍，无显著收益
- **Hard 黑名单（uid 或 keyword）**：太死板，不能泛化到"语义相近的新项"
- **训练一个分类器**：数据量小，过拟合风险大；运维负担大

---

## 10. 探索预算（每期 15% 低画像匹配但高质量项强制保留）

**Context**：完全跟着画像走会让系统越来越保守，错过"surprising-but-valuable"。

**Decision**：每个 cycle 把通过去重的"低 profile match 项"按比例随机选出一部分 admit，标 `EXPLORE`。

**Consequences**：
- ✅ 缓解"画像过拟合"，给意外发现留通道
- ✅ 比例可调（`exploration_budget` in sources.yaml）
- ❌ 偶尔会推一些真不相关的项给决策者 —— 但这就是探索的代价
- ❌ 当前用随机抽样；理想应改"按 quality signal 排序后取 top N%"

**Alternatives 拒绝**：
- **不做**：系统越用越窄
- **完全随机扫描**：无法收敛到 Lab 兴趣

---

## 11. Cycle 内去重（uid + 向量）vs. Reporter 拉周期数据

**Context**：Triage 强力去重（已扫过的 uid 直接 DUPLICATE）会让本 cycle 的 `state.scores` 为空，但报告应该是"周报"——过去 7 天的全集。

**Decision**：分离去重与报告：
- **Triage 永远去重**（save token + 一致性）
- **Reporter 不读 `state.scores`，从 KB 读"过去 N 天评分"**（按 uid 去重保留最新）

**Consequences**：
- ✅ 一周内重跑 N 次，每次报告都完整
- ✅ Re-evaluation（`--force`）后报告自然反映新分数
- ❌ 周期跨度需要在 reporter 里硬编码一个常量 `REPORT_PERIOD_DAYS=7`（未做成可配置）

**Alternatives 拒绝**：
- **报告只看本 cycle**：周二跑发现报告空的（典型 bug）
- **取消 Triage 去重**：每周浪费 50% LLM 成本评测同一批

---

## 12. 配置：YAML + Pydantic Settings + .env（vs. 单一 JSON / TOML）

**Context**：配置分多层 —— 跑参（阈值）、源（启用 / 关键词）、评测维度（rubric + prompt）、画像（anchors + auto_learned）。

**Decision**：
- **运行时参数 / API key** → `.env` (Pydantic Settings, prefix `RADAR_`)
- **声明式数据**（源、rubric、画像）→ YAML

**Consequences**：
- ✅ 运维敏感（key）与项目数据（rubric）分离
- ✅ YAML 对人友好，rubric 多行 prompt 模板可读性远超 JSON
- ✅ Pydantic 验证错误信息非常友好
- ❌ 两套语法（env + yaml）轻微心智负担

**Alternatives 拒绝**：
- **全 env 变量**：rubric 多行 prompt 模板塞 env 是噩梦
- **TOML**：Python 生态对 YAML 支持更广（PyYAML 内置），且 YAML 更适合多行字符串

---

## 已知局限与未来工作

| 局限 | 影响 | 改进方向 |
|---|---|---|
| L1 静态评测 | 给不出"真跑得通"信号 | 升级到 L2（Docker 沙箱跑 quickstart） |
| 无中文社区源 | 错过国内研究/工具动态 | 加 `sources/zhihu.py` / `wechat_official.py`（合规允许下） |
| 单机存储 | 多人同时写会冲突 | 升级到 PostgreSQL + pgvector |
| 报告周期硬编码 7 天 | 无法做日报/月报 | `Reporter` 接受 `--period` 参数 |
| Critic 偶发过度反对 | Critic 时不时大幅否定主评分 | 引入"温度低 + temperature lock"或更精细的 disagreement 评分 |
| 反馈表达粒度粗（三档） | 无法区分"采纳但仅参考"vs"采纳并立项" | 加可选 `note` 字段 + LLM 按文本分类 |
| Cold start 期 RAG 锚点弱 | 头几周评分稳定性较差 | 黄金种子已缓解；可再扩到 50 条 |
| 探索预算用纯随机 | 偶尔推真不相关项 | 用 quality signal（stars / citations）做加权采样 |

---

## 决策时间线

| 时间 | 决策 | 触发事件 |
|---|---|---|
| Sprint 0 (规划) | LangGraph + L1 + DeepSeek/Qwen | 初始方案讨论 |
| Sprint 1 | 三层记忆 + Pydantic schemas | 项目骨架 |
| Sprint 1 | 反馈三档 + 反向 centroid 设计预留 | UI 设计 |
| Sprint 1 | 24 条黄金种子 | 防 prompt drift |
| Sprint 2（修复） | Reporter 改为读 KB 周期数据 | 重跑后报告空白 bug |
| Sprint 2 | `--force` 标志 | 评测改 prompt 后需重评 |
| Sprint 2 | IgnoreFilter 反向 centroid | 用户点 Ignore 但无效 |
| Sprint 2 | DashScope embedding 切换 | 用户偏好 + 中文场景 |
