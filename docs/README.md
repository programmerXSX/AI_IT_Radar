# AI IT Radar — 文档索引

为 AI Lab 设计的常态化技术情报雷达系统，多 Agent 架构 + RAG + L1 评测 Harness + 反馈记忆。

## 文档地图

按你的角色 / 目的选择入口：

| 我想... | 看哪一篇 |
|---|---|
| 30 秒了解项目做什么 | [项目根 README](../README.md) |
| 理解系统怎么搭起来的（含图） | [architecture.md](architecture.md) |
| 理解为什么这么搭、有什么 trade-off | [design.md](design.md) |
| 把它跑起来 / 调优 / 排错 | [operations.md](operations.md) |
| 加新源 / 加评测维度 / 加 Agent | [extending.md](extending.md) |

## 一句话说明四份文档的不同

- **`architecture.md`** = _What & How it fits together_ —— 12 节，含 9 张 Mermaid 图
- **`design.md`** = _Why we chose this way_ —— ADR 风格，12 个关键决策
- **`operations.md`** = _Run / tune / fix it_ —— 9 节运维手册，含 cheatsheet
- **`extending.md`** = _Grow it_ —— 4 类扩展场景，每类完整代码骨架

## 快速链接

### 架构图（9 张）

- [系统总览（C4 Context）](architecture.md#2-系统总览c4--context-level)
- [多 Agent 流水线（LangGraph 拓扑）](architecture.md#3-多-agent-流水线langgraph-拓扑)
- [三层记忆系统](architecture.md#4-三层记忆系统)
- [RAG 三处召回（时序图）](architecture.md#5-rag-三处召回)
- [评测 Harness 流程](architecture.md#6-评测-harnessl1-轻量但可回归)
- [反馈闭环](architecture.md#7-反馈闭环)
- [数据模型（class diagram）](architecture.md#8-数据模型)
- [配置层级](architecture.md#9-配置层级)
- [包级依赖](architecture.md#12-模块依赖包级)

### 12 个设计决策

1. [编排框架：LangGraph vs CrewAI / AutoGen](design.md#1-编排框架langgraph-vs-crewai--autogen)
2. [Harness 深度：L1 vs L2 vs L3](design.md#2-harness-深度l1-静态-vs-l2-沙箱-vs-l3-真跑-benchmark)
3. [LLM 提供商组合：DeepSeek + Qwen + DashScope](design.md#3-llm-提供商组合deepseek-主--qwen-critic--dashscope-embedding)
4. [三层记忆系统设计](design.md#4-三层记忆系统vs-单一向量库-vs-单一-sql)
5. [RAG 在三处召回](design.md#5-rag-在三处召回而非单一去重)
6. [评测幻觉的三重防线](design.md#6-评测幻觉的三重防线)
7. [黄金集回归](design.md#7-黄金集回归vs-不做做更大)
8. [反馈三档（Adopt / Watch / Ignore）](design.md#8-反馈三档adopt--watch--ignore-vs-五星打分)
9. [IgnoreFilter：反向 centroid](design.md#9-ignorefilter反向-centroidvs-黑名单)
10. [探索预算](design.md#10-探索预算每期-15-低画像匹配但高质量项强制保留)
11. [Cycle 内去重 vs Reporter 拉周期数据](design.md#11-cycle-内去重uid--向量vs-reporter-拉周期数据)
12. [配置：YAML + Pydantic Settings + .env](design.md#12-配置yaml--pydantic-settings--envvs-单一-json--toml)

### 运维高频章节

- [首次部署 7 步](operations.md#一从零启动首次部署)
- [所有 CLI 命令](operations.md#四所有-cli-命令)
- [调优指南](operations.md#五调优指南)
- [故障排查](operations.md#六故障排查)
- [成本控制](operations.md#七成本控制)
- [Cheatsheet（贴墙用）](operations.md#九cheatsheet贴墙用)

### 扩展高频场景

- [加一个新信息源](extending.md#一加一个新信息源)（4 个文件，难度 ★★★）
- [加一个评测维度](extending.md#二加一个评测维度)（1 个 YAML，难度 ★，无代码改动）
- [加一个新 Agent](extending.md#三加一个新-agent)（3-5 个文件，难度 ★★★★）
- [加一种报告格式](extending.md#四加一种报告格式)（2 个文件，难度 ★★）

## 推荐阅读顺序

### 第一次接手项目（约 30 分钟）

1. 项目根 [README.md](../README.md) — 5 分钟
2. [architecture.md §1-§3](architecture.md#1-系统定位) — 看 C4 + Agent 拓扑 — 10 分钟
3. [operations.md §一](operations.md#一从零启动首次部署) — 装起来跑一轮 — 15 分钟

### 深度评审（约 2 小时）

1. [architecture.md](architecture.md) 通读 — 30 分钟
2. [design.md](design.md) 通读 — 60 分钟（重点看你不同意的决策）
3. [operations.md §五调优](operations.md#五调优指南) — 30 分钟

### 给同事讲项目（10 分钟版）

只用这 4 张图：
1. [系统总览](architecture.md#2-系统总览c4--context-level)
2. [多 Agent 流水线](architecture.md#3-多-agent-流水线langgraph-拓扑)
3. [三层记忆](architecture.md#4-三层记忆系统)
4. [反馈闭环](architecture.md#7-反馈闭环)

## 文档维护约定

- 任何代码改动如果影响架构或决策，同步改 `architecture.md` / `design.md`
- 任何 CLI / 配置项变更，同步改 `operations.md`
- 设计决策反思（"如果重做会怎么改"）追加到 [design.md 已知局限](design.md#已知局限与未来工作) 表

## 联系

文档反馈 / 不懂的部分，直接在内部 issue tracker 提一条。
