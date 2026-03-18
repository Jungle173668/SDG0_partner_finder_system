# SDGZero Intelligence System — 项目框架 v5.0

> **v5 更新内容：** 融入 mcp-db-server 作为原始数据查询的独立 MCP 入口；HyDE 移至 SearchAgent（召回阶段）；ScoringAgent 精简为 Cross-encoder 精排 + 理由生成，不再重复检索。

---

## 1. 系统架构总览

```
用户入口层
前端界面（Next.js）
筛选表单 + 搜索结果 + 报告展示
        ↓
服务层
FastAPI  ←→  Redis（缓存 + Celery broker）
        ↓
AI / ML 层
Multi-Agent Pipeline（LangGraph）
  SearchAgent → ResearchAgent → ScoringAgent → ReportAgent
Embedding Model（sentence-transformers）
RAG Pipeline
ML Classifier（SetFit SDG 分类器）
MCP Server（自建）+ mcp-db-server（原始数据查询）
        ↓
存储层
PostgreSQL + pgvector    Redis    AWS S3
        ↓
MLOps 层
MLflow + Airflow + Evidently + GitHub Actions CI/CD
        ↓
部署层
Railway（API + 数据库）    Vercel（前端）    Ollama/Groq（LLM）    AWS S3（Artifacts）
```

---

## 2. 技术栈清单

### 2.1 数据采集层

| 组件 | 技术 | 用途 |
|------|------|------|
| HTTP 客户端 | `httpx` + `asyncio` | 异步调用 SDGZero REST API |
| HTML 解析 | `BeautifulSoup4` | 提取结构化字段（备用） |
| 数据模型 | `Pydantic v2` | 数据验证与序列化 |
| 去重检测 | `hashlib` MD5 | 检测页面内容变化，避免重复处理 |
| 爬取限速 | `asyncio.sleep` | 礼貌性延迟 |
| 外部调研 | `Tavily Python SDK` | 抓取目标公司官网内容（extract 模式） |

**采集字段（来自 `/wp-json/geodir/v2/businesses`）：**

| 分类 | 字段 | 说明 |
|------|------|------|
| 身份 | `id`, `slug`, `name`(←title), `url`(←link) | 主键与页面地址 |
| 时间 | `scraped_at`(←modified) | 最后修改时间，用于增量更新 |
| 位置 | `street`, `city`, `region`, `country`, `zip`, `latitude`, `longitude` | 地理信息，支持 SQL 精确过滤 |
| 联系 | `phone`, `website` | 基础联系方式 |
| 社交 | `linkedin`, `facebook`, `twitter`, `instagram`, `tiktok`, `video` | 外联 Agent 使用 |
| 企业信息 | `business_type`(B2B/B2C/Both), `job_sector`(Private/Public/Third), `company_size`, `package_id`(会员等级), `claimed`(是否认证), `founder_name` | 过滤与分析用 |
| 分类 | `categories`(←post_category) | 如 "Energy & Renewables" |
| 文本内容 | `content`(完整描述~2000字), `summary`, `achievements_summary`, `sdg_involvement` | **核心 embedding 语料，100% 填充** |
| SDG | `sdg_tags`(←post_tags, 名称列表), `sdg_slugs`(slug列表), `membership_tier`(←sdg_badges) | RAG 检索与分类器训练 |
| 评分 | `rating`, `rating_count` | SQL 排序用 |

**embedding 文本组合顺序（`to_embedding_text()`）：**

```
Company → Categories → Sector → City → Country →
Description → Summary → Achievements → SDG involvement → SDGs
```

---

### 2.2 存储层

| 组件 | 技术 | 用途 |
|------|------|------|
| 关系型数据库 | `PostgreSQL 15+` | 结构化企业数据 |
| 向量扩展 | `pgvector` | embedding 向量存储与检索 |
| ORM | `SQLAlchemy 2.0` | Python 操作数据库 |
| 数据库迁移 | `Alembic` | Schema 版本管理 |
| 连接驱动 | `psycopg2` / `asyncpg` | 数据库连接 |
| 缓存 | `Redis` | 热点查询结果缓存 |
| 对象存储 | `AWS S3` | 原始 HTML 归档、MLflow artifacts、模型文件 |

**为什么选 pgvector 而不是专用向量数据库：**
- 与结构化数据在同一事务内，保证一致性
- 支持 SQL + 向量混合查询（如"伦敦的 + 语义相似的"）
- 490 家企业规模完全够用，无需额外运维成本

---

### 2.3 AI / ML 层

#### Embedding 与检索模型

| 组件 | 技术 | 用途 |
|------|------|------|
| Bi-encoder | `sentence-transformers` (`all-MiniLM-L6-v2`) | 向量化 + 快速召回（384-dim） |
| Cross-encoder | `sentence-transformers` cross-encoder 系列 | 精排：对 Top-20 候选重新打分 |

> Bi-encoder 和 Cross-encoder 的区别：Bi-encoder 分别编码两段文字再比较向量，速度快适合全库召回；Cross-encoder 把两段文字拼在一起输入模型，精度更高但只适合对少量候选做精排。参考：Reimers & Gurevych 2019（SBERT）。

#### LLM & Agent

| 组件 | 技术 | 用途 |
|------|------|------|
| LLM | `Ollama`（本地）/ `Groq`（API，有免费额度） | HyDE 描述生成、排序理由生成、报告生成，零/低成本 |
| Agent 框架 | `LangGraph` | Multi-Agent 状态机编排 |
| RAG Pipeline | `LangGraph` + `pgvector` | 检索增强生成 |
| Agent State | `LangGraph State` + `PostgreSQL` | 会话内 state 传递 + 搜索结果持久化 |
| MCP Server（自建） | `mcp` Python SDK | 暴露 Agent 工具（search / detail / outreach） |
| MCP Server（数据库） | `mcp-db-server` | 自然语言直查 PostgreSQL，独立演示入口，本地 NL-to-SQL 模型零成本 |
| 外部调研 | `Tavily Python SDK` | 抓取目标公司官网（extract 模式，处理反爬和 JS 渲染） |

#### 搜索工具定义（SearchAgent 使用）

| 工具名 | 触发条件 | 底层实现 |
|--------|----------|----------|
| `semantic_search` | 用户描述业务类型、寻找语义相关公司 | pgvector 余弦相似度 |
| `sql_filter` | 用户指定精确条件（城市/SDG编号/认证状态） | PostgreSQL WHERE / ORDER BY |
| `hybrid_search` | 同时有语义需求 + 精确条件 | 向量搜索 + SQL WHERE 组合 |

> LLM 根据工具描述自动判断调用哪个，无需硬编码规则。

#### Multi-Agent Pipeline（四层流水线）

```
用户填写入口表单
        ↓
[SearchAgent]     HyDE 生成理想描述 + Schema 注入 → hybrid_search 召回 Top-20
        ↓
[ResearchAgent]   Tavily 对每家公司做外部调研，补充官网信息
        ↓
[ScoringAgent]    Cross-encoder 精排 Top-20 → Top-5 + LLM 生成推荐理由
        ↓
[ReportAgent]     将 Top-5 + 理由格式化为结构化报告 + 外联消息草稿
        ↓
报告页面 + 可分享 URL
```

#### ML 模型

| 组件 | 技术 | 用途 |
|------|------|------|
| SDG 分类器 | `SetFit` (zero-shot) | 批量预填充 predicted_sdg_tags（覆盖率从4%→90%） |
| 基础模型 | `all-MiniLM-L6-v2` | 与 bi-encoder 一致，复用权重 |
| 模型评估 | `classification_report` + `MLflow` | F1、Precision、Recall 追踪 |
| 企业聚类 | `KMeans` / `HDBSCAN` | 发现企业画像群组 |
| Prompt 管理 | 版本化 Prompt Registry | A/B 测试不同 prompt 策略 |

---

### 2.4 MLOps 层

| 组件 | 技术 | 用途 |
|------|------|------|
| 实验追踪 | `MLflow` | 记录模型参数、指标、版本 |
| 模型注册 | `MLflow Model Registry` | Champion/Challenger 版本管理 |
| 流水线调度 | `Apache Airflow` | 定时爬取、增量更新、模型重训练 |
| 数据漂移监控 | `Evidently AI` | 检测 embedding 分布变化 |
| CI/CD | `GitHub Actions` | 自动测试 + 模型训练 + 部署 |
| Artifacts 存储 | `AWS S3` | MLflow 模型文件远程存储 |

#### CI/CD 流程

```
Push to main
    ↓
[GitHub Actions]
├── pytest 单元测试
├── 数据质量检查
├── 模型重新训练（如数据有更新）
├── MLflow 记录新实验
├── 评估指标是否达标（F1 > 0.85）
└── 达标 → 自动部署到生产
         不达标 → 告警，保留旧版本
```

---

### 2.5 服务层

| 组件 | 技术 | 用途 |
|------|------|------|
| API 框架 | `FastAPI` | REST API + 自动 Swagger 文档 |
| 异步任务 | `Celery` + `Redis` | Agent Pipeline 异步执行 |
| 数据验证 | `Pydantic v2` | 请求/响应校验 |
| 认证 | `JWT` | 内部接口鉴权 |

#### 核心 API 接口

```
POST /search                    # 触发 Multi-Agent Pipeline，返回 session_id
GET  /search/{session_id}       # 读取搜索结果（支持 URL 分享）
GET  /businesses/{slug}         # 企业详情
POST /internal/refresh/{slug}   # 手动触发数据更新
GET  /health                    # 服务健康检查
```

---

### 2.6 前端层

| 组件 | 技术 | 用途 |
|------|------|------|
| 框架 | `Next.js` | 前端页面 |
| 样式 | `Tailwind CSS` | 快速布局 |
| 部署 | `Vercel`（免费） | 前端托管，连接 Railway 后端 |

**页面结构：**

```
/                首页：入口表单（筛选条件 + 公司描述）
/results/{id}    结果页：推荐报告 + 外联消息（可分享 URL）
/company/{slug}  公司详情页：完整信息 + AI 分析
```

---

### 2.7 基础设施

| 组件 | 技术 | 用途 |
|------|------|------|
| 容器化 | `Docker` + `docker-compose` | 本地一键启动所有服务 |
| 云部署 | `Railway`（免费） | API + 数据库生产部署 |
| 前端部署 | `Vercel`（免费） | Next.js 前端托管 |
| AWS 服务 | `S3` + `Bedrock` | 对象存储 + LLM 调用 |
| 日志 | `structlog` | 结构化日志 |
| 测试 | `pytest` + `pytest-asyncio` | 单元 & 集成测试 |
| 版本控制 | `Git` + `GitHub` | 代码管理 |

---

## 3. 数据库 Schema

```sql
-- 企业主表
businesses (
    -- 身份
    id                   INTEGER PRIMARY KEY,
    slug                 TEXT UNIQUE,
    name                 TEXT,              -- ← API title
    url                  TEXT,              -- ← API link
    scraped_at           TIMESTAMP,         -- ← API modified

    -- 位置
    street               TEXT,
    city                 TEXT,
    region               TEXT,
    country              TEXT,
    zip                  TEXT,
    latitude             FLOAT,
    longitude            FLOAT,

    -- 联系方式
    phone                TEXT,
    website              TEXT,

    -- 社交媒体
    linkedin             TEXT,
    facebook             TEXT,
    twitter              TEXT,
    instagram            TEXT,
    tiktok               TEXT,
    video                TEXT,
    logo                 TEXT,

    -- 企业信息
    business_type        TEXT,              -- B2B / B2C / Both
    job_sector           TEXT,              -- Private / Public / Third Sector
    company_size         TEXT,
    package_id           INTEGER,           -- 会员等级
    claimed              TEXT,              -- Yes/No 是否认证
    founder_name         TEXT,

    -- 文本内容（embedding 核心语料）
    content              TEXT,              -- 完整描述 ~2000字，100% 填充
    summary              TEXT,
    achievements_summary TEXT,
    sdg_involvement      TEXT,

    -- SDG 预测（SetFit 批量填充，入库后运行）
    predicted_sdg_tags   TEXT,              -- 覆盖率从 4% → ~90%

    -- 评分
    rating               FLOAT,
    rating_count         INTEGER,

    -- 运维
    html_hash            TEXT,
    is_active            BOOLEAN DEFAULT TRUE,
    updated_at           TIMESTAMP
)

-- SDG 标签（来自 post_tags）
business_sdg_tags (
    business_id  INTEGER REFERENCES businesses(id),
    sdg_name     TEXT,                      -- e.g. "Climate Action"
    sdg_slug     TEXT,                      -- e.g. "climate-action"
    PRIMARY KEY (business_id, sdg_slug)
)

-- 分类（来自 post_category）
business_categories (
    business_id  INTEGER REFERENCES businesses(id),
    category     TEXT,                      -- e.g. "Energy & Renewables"
    PRIMARY KEY (business_id, category)
)

-- 会员等级（来自 sdg_badges）
business_membership_tiers (
    business_id  INTEGER REFERENCES businesses(id),
    tier         TEXT,                      -- e.g. "Ambassador"
    PRIMARY KEY (business_id, tier)
)

-- 向量表（pgvector 核心）
business_embeddings (
    id           SERIAL PRIMARY KEY,
    business_id  INTEGER REFERENCES businesses(id),
    full_text_vec VECTOR(384),              -- to_embedding_text() 完整输出
    model_name   TEXT,
    created_at   TIMESTAMP
)

-- 搜索会话（支持 URL 分享，无需用户注册）
search_sessions (
    id                   TEXT PRIMARY KEY,  -- 随机生成，如 "x7k2m9"，用于 URL
    filters              JSONB,             -- 用户填写的筛选条件
    user_company_desc    TEXT,              -- 用户公司描述
    other_requirements   TEXT,             -- 其他要求
    candidate_companies  JSONB,            -- SearchAgent 结果（Top-20）
    research_results     JSONB,            -- ResearchAgent 结果
    scored_companies     JSONB,            -- ScoringAgent 结果（Top-5 + 理由）
    report               TEXT,             -- ReportAgent 生成的报告
    created_at           TIMESTAMP,
    expires_at           TIMESTAMP         -- 30天后过期清理
)

-- Agent 会话记忆
agent_sessions (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT UNIQUE,
    history     JSONB,
    created_at  TIMESTAMP,
    updated_at  TIMESTAMP
)

-- BD 跟进记录（PipelineTrackerAgent）
pipeline_contacts (
    id           SERIAL PRIMARY KEY,
    session_id   TEXT,
    business_id  INTEGER REFERENCES businesses(id),
    status       TEXT,     -- contacted/replied/in_conversation/partnered
    contacted_at TIMESTAMP,
    last_updated TIMESTAMP,
    notes        TEXT
)

-- 爬取日志
scrape_log (
    id         SERIAL PRIMARY KEY,
    url        TEXT,
    status     TEXT,
    html_hash  TEXT,
    changed    BOOLEAN,
    scraped_at TIMESTAMP
)

-- 模型预测日志（SetFit，用于再训练）
prediction_log (
    id              SERIAL PRIMARY KEY,
    model_version   TEXT,
    input_text      TEXT,
    predicted_sdgs  JSONB,
    confidence      FLOAT,
    actual_sdgs     JSONB,  -- 人工标注后回填
    created_at      TIMESTAMP
)
```

---

## 4. 用户入口界面设计

```
┌─────────────────────────────────────────────────┐
│  SDGZero Partner Finder                         │
│      ── 筛选条件（部分不确定的可留空）─────────────────────────── │                                                 │
│                                                 │
│  行业分类：  [Energy & Renewables ▼] [必须●/优先○]  │  ← categories 字段
│  SDG 目标：  [SDG 7 ▼] [SDG 13 ▼]  [必须○/优先●]  │  ← sdg_tags 字段
│  位置：      [London              ▼] [必须●/优先○] │  ← city 字段
│  企业类型：  [B2B ▼]               [必须○/优先●]  │  ← business_type 字段
│  仅认证企业：[✓]                                  │  ← claimed 字段
│                                                 │
│  我的公司做什么：                                  │
│  [我们提供企业碳排放审计与减排咨询服务...      ] │  ← 自由输入，最重要

│  其他要求：                                     │
│  [希望对方有主动合作意愿，规模在50人以下...    ] │  ← 自由输入
│                                                 │
│  [开始分析]                                     │
└─────────────────────────────────────────────────┘
```

**字段设计说明：**
- "我的公司做什么"是整个推荐的核心锚点，贯穿所有 Agent 的分析，尤其是 ScoringAgent 的 HyDE 阶段
- 筛选条件直接映射到 SearchAgent 的 `sql_filter` 工具
- "其他要求"作为自由文本传入 `semantic_search`
- 两者同时存在时触发 `hybrid_search`

**Hard / Soft 筛选模式（Phase 3 实现）：**

每个筛选条件可由用户选择"必须"或"优先"：
- **必须（Hard）**：进入 ChromaDB WHERE 子句，不满足直接排除；不足时触发三层降级兜底
- **优先（Soft）**：不进 WHERE 子句，传给 ScoringAgent 作为加分项；满足则排名靠前，不满足也不排除

两者互不干扰：三层降级只管理 Hard 过滤器这一侧；Soft 过滤器完全绕过 WHERE，只在 ScoringAgent 打分阶段生效。

筛选的字段需要匹配严格数据库中有的结果。比如，地区在数据库里有London，等，则筛选框中不能出现比他大的England，Wales等。

**搜索完成后生成可分享 URL：**

```
https://yourapp.com/results/x7k2m9

用户可以：
├── 书签收藏（30天有效）
├── 分享给同事
└── 下次直接打开看历史结果
（无需注册，无需登录）
```

---

## 5. Multi-Agent 完整流程

### 5.1 LangGraph State 设计

所有 Agent 通过共享 `AgentState` 对象传递数据，每个 Agent 读取上游字段，写入自己的输出字段：

```
AgentState 字段：

用户输入（所有 Agent 可读）：
  user_company_desc    → 用户公司描述，SearchAgent 的 HyDE 输入
  filters              → 筛选条件，SearchAgent 的 SQL 过滤依据
  other_requirements   → 自由文本，搜索时追加到语义查询
  session_id           → 用于持久化和 URL 分享

SearchAgent 写入：
  hypothetical_partner_desc → HyDE 生成的理想合作伙伴描述
  candidate_companies       → Top-20 候选公司

ResearchAgent 写入：
  research_results     → 每家公司的调研摘要（数据库 + 官网）

ScoringAgent 写入：
  scored_companies     → Top-5 公司 + Cross-encoder 分数 + 排序理由
                         + soft_filter_hit（每家公司满足了哪些 soft 条件）
                         + match_quality（"strong" | "partial" | "fallback"）

ReportAgent 写入：
  report               → 最终 Markdown 报告
```

**LangGraph 流水线结构：**

```
SearchAgent → ResearchAgent → ScoringAgent → ReportAgent
```

四个节点顺序执行，每个节点完成后将结果写入 State，下一个节点直接读取。

---

### 5.2 SearchAgent

```
职责：生成理想合作伙伴描述（HyDE），从数据库召回高质量 Top-10

Step 1：HyDE + Schema 注入
  输入：user_company_desc + other_requirements
  LLM 同时拿到：
    用户描述
    Schema 摘要（city / categories / sdg_tags 的真实值，从 Redis 缓存读取）
  输出：hypothetical_partner_desc
       一段风格和信息密度接近真实公司描述的"理想合作伙伴画像"
       LLM 知道数据库里有哪些真实字段值，不会生成幻觉 SQL

  Schema 注入内容（每天刷新一次，缓存 Redis）：
    city 现有值：London, Edinburgh, Bristol...
    categories 现有值：Energy & Renewables, Tech, Finance...
    sdg_tags 现有值：SDG7, SDG13, SDG12...
    → LLM 生成 SQL 时用真实存在的值，不会查空

Step 2：hybrid_search
  向量部分：embed(hypothetical_partner_desc) 替代用户原始输入
           与数据库公司描述语义空间更接近，召回质量更高
  SQL 部分：来自用户筛选条件（filters）
  两者组合为单条查询（pgvector 阶段）

过滤器分类（Phase 3 加入 Hard/Soft 后）：
  Hard 过滤：进 WHERE 子句，走三层降级兜底
  Soft 过滤：不进 WHERE，传给 ScoringAgent 作加分项

空结果三级降级（只作用于 Hard 过滤器）：
  级别1：完整条件（SQL + 向量）→ 有结果直接返回
  级别2：放宽 SQL 条件（去掉限制最严的），保留向量 → 告知用户"已放宽条件"
  级别3：纯向量搜索，忽略所有 SQL 过滤 → 告知用户"未找到完全匹配"

输出（写入 AgentState）：
  hypothetical_partner_desc → 传给 ScoringAgent 做 Cross-encoder 输入
  candidate_companies       → Top-10 高质量候选公司
```

> 参考：Gao et al. 2022，"Precise Zero-Shot Dense Retrieval without Relevance Labels"（HyDE）

---

### 5.3 ResearchAgent

```
职责：补充数据库之外的公司信息，为 ScoringAgent 提供更丰富的输入

输入（来自 AgentState）：
  candidate_companies  → Top-10 公司列表

数据来源优先级（三层）：
  层1：SDGZero 数据库（content / summary / sdg_involvement）
       始终使用，免费，覆盖率100%
  层2：Tavily extract（抓取官网）
       使用数据库 website 字段的 URL
       处理反爬 + JS 渲染，合法（责任在 Tavily）
       免费额度 1000次/月
  层3：Tavily search（搜索公司名）
       官网 URL 失效时的兜底

并行处理：
  20家公司同时调研（asyncio），不串行等待

成本估算：
  每次用户搜索：20次 Tavily extract 调用
  每月50次用户搜索 ≈ 1000次调用，在免费额度内

输出（写入 AgentState）：
  research_results     → 每家公司300字调研摘要，注明数据来源
```

---

### 5.4 ScoringAgent

```
职责：对 Top-10 候选做 Cross-encoder 精排，筛出 Top-5，生成推荐理由
      不再做检索，HyDE 已在 SearchAgent 完成

输入（来自 AgentState）：
  user_company_desc & other requirements → 用户原始输入
  hypothetical_partner_desc → SearchAgent 生成的理想合作伙伴描述
  candidate_companies       → Top-10 候选公司
  research_results          → ResearchAgent 补充的官网摘要

Step 1：Cross-encoder 精排
  对 Top-10 逐一打分：
    输入：[user_company_desc] [SEP] [company.full_text]
    模型看到两段文字的完整交互，精度高于 Bi-encoder
  输出：Top-5 + 相关性分数

Step 2：LLM 生成推荐理由
  Cross-encoder 只输出分数，不解释原因
  对 Top-5 逐一调用 LLM：
    输入：user_company_desc + company 完整信息 + research_summary
          + soft_filters（用户标注为"优先"的条件，引导推荐理由的侧重）
    输出：一段推荐理由，说明合作切入点；若满足 soft 条件则在理由中体现

输出（写入 AgentState）：
  scored_companies → Top-5 + Cross-encoder 分数 + 推荐理由
```

**理论依据：**
- Cross-encoder：Reimers & Gurevych 2019（SBERT）；Nogueira & Cho 2019（MonoBERT reranking）

**为什么不在这里做 HyDE：**
```
HyDE 在 SearchAgent 已完成，hypothetical_partner_desc 直接复用
ScoringAgent 只做精排和解释，职责清晰，不重复检索
```

---

### 5.5 ReportAgent

```
职责：将 ScoringAgent 的结果格式化为可读报告，不做任何推理

输入（来自 AgentState）：
  scored_companies     → Top-5 + 分数 + 推荐理由（ScoringAgent 已生成）
  filters              → 用于报告摘要

输出（写入 AgentState）：
  report               → Markdown 格式报告
```

**报告格式：**

```
━━ 搜索摘要 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
条件：伦敦（必须）/ SDG7（优先）/ B2B（必须）/ 已认证
候选：从 47 家公司中筛选，为你推荐 5 家
[若发生降级] ⚠️ 未找到完全匹配 · 已放宽：SDG7 限制

━━ 推荐结果 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#1  GreenTech London                    匹配度 87%  ✅ 强匹配
    ████████████████████░░░░
    标签：SDG7 ✓  SDG13  B2B  认证✓  能源

    推荐理由：（ScoringAgent 生成，soft_filters 引导侧重）
    专注企业碳减排实施方案，与你的审计业务高度互补。
    他们的客户恰好是你的潜在审计对象，存在客户转介绍空间。

    合作切入点：联合投标、客户转介绍
    联系方式：LinkedIn · greentech.co.uk

    ▼ 外联消息草稿
      邮件草稿 / LinkedIn 消息 / 3个开场话题

#2  EcoConsult UK                       匹配度 71%  ◐ 部分符合
    标签：SDG7 ✗（SDG 标签缺失，但业务高度相关）B2B  认证✓

#3  SustainCo Manchester                匹配度 65%  ⚠️ 降级推荐
    ℹ️ 伦敦内无完全匹配，Manchester 公司供参考

#4-5  ...（同上格式）

━━ 可视化 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[公司位置地图]（latitude/longitude 字段）
[SDG 覆盖矩阵]（5家公司 × 17个 SDG）
```

**结果透明度分层（Result Transparency）：**

搜索路径标注（摘要区显示）：
- Level 0 完整匹配 → 不提示（正常结果）
- Level 1 放宽条件 → ⚠️ "已放宽：SDG/认证限制，以下为最接近匹配"
- Level 2 纯语义   → ⚠️ "未找到完全匹配，以下为语义相关推荐"

每家公司匹配质量标签（卡片右上角）：
- ✅ 强匹配   — Cross-encoder 高分 + 所有 hard/soft 条件均满足
- ◐ 部分符合 — 语义相关，但某个 soft 条件未满足（附说明原因）
- ⚠️ 降级推荐 — 来自 Level 1/2 fallback（附说明放宽了哪个条件）

ReportAgent 职责：从 ScoringAgent 的 scored_companies 中读取
search_fallback_level 和每家公司的 soft_filter_hit 字段，
自动生成对应的透明度标注，不需要额外 LLM 调用。

**可视化原则：**

```
保留：
  匹配度进度条（简单直观）
  SDG 标签徽章（官方配色）
  公司位置地图（lat/lng 字段）
  外联消息折叠显示（不干扰主报告）

不做：
  复杂图表（演示时分散注意力）
```

---

## 6. 搜索结果持久化（无需注册）

```
方案：URL + 随机 session_id，服务端存储

流程：
  1. 用户提交搜索表单
  2. 后端生成随机 session_id（6位，如 "x7k2m9"）
  3. Celery 异步执行 Agent Pipeline
  4. 结果写入 search_sessions 表
  5. 前端跳转到 /results/x7k2m9

优点：
  不需要注册，不需要登录
  换浏览器也能访问（服务端存储，非 localStorage）
  可分享给同事（同一 URL）
  30天后自动过期清理

未来演进（如需要）：
  邮箱 Magic Link 登录（NextAuth.js）
  绑定历史搜索记录
  现阶段不需要实现
```

---

## 7. MCP Server 设计

两个独立的 MCP Server，覆盖不同使用场景：

**Server 1：自建 MCP Server（Agent 工具入口）**

```
工具列表：

search_businesses(query, filters?)
  → 触发完整 Multi-Agent Pipeline，返回 session_id

get_business_detail(slug)
  → 获取企业完整信息

get_search_session(session_id)
  → 读取已完成的 Pipeline 结果

generate_outreach_message(business_slug, user_context?)
  → 单独生成外联消息
```

**Server 2：mcp-db-server（原始数据查询入口）**

```
直接把 PostgreSQL 暴露给 MCP 客户端
使用本地 HuggingFace NL-to-SQL 模型，零 API 成本
只允许 SELECT，有注入防护

⚠️  依赖 PostgreSQL，ChromaDB 阶段无法使用
    Phase 4 完成 pgvector 迁移后才能接入
    SearchAgent 的 SQL 生成也在此时改为调用 mcp-db-server

使用场景：
  面试演示时，面试官在 Claude Desktop 里直接问：
  "SDGZero 里有哪些伦敦的能源公司？"
  → 实时查数据库返回结果
  → 不需要走完整 Pipeline，即时响应

两个 Server 的区别：
  mcp-db-server  → 原始数据查询，快，适合探索性提问
  自建 MCP Server → AI 增强查询，慢，适合完整分析流程
```

**面试亮点：** 两种 MCP 接入方式并存——原始数据查询（mcp-db-server）和 AI 增强分析（自建 Server），覆盖不同使用场景，体现对 MCP 协议的深度理解。

---

## 8. ML 模型：SDG 分类器（SetFit）

**问题：** sdg_tags 字段填写率仅 4%（~20家），Agent 的 SDG 过滤几乎失效。

**解决方案：** SetFit 零样本分类（Tunstall et al. 2022，"Efficient Few-Shot Learning Without Prompts"）

**运行时序（重要）：**

```
SetFit 是数据管道的必要步骤，不是可选的后处理。

原因：
  sdg_tags 原始填写率仅 4%
  用户一输入 SDG 筛选条件，sql_filter 几乎查空
  → SDG 过滤完全失效，降级为纯向量搜索
  → 筛选条件形同虚设

正确的入库流程（每条数据都走这四步）：
  REST API 拉取 → 入库 → embedding 生成 → SetFit 预测 predicted_sdg_tags
  四步连续，缺一不可，数据才算"可用"
```


**完整流程：**

```
一次性初始化（Phase 1，数据入库前完成）：
  SDG 官方描述 × 17 → get_templated_dataset()
  → 生成 136 条合成训练数据（零标注）
  → 训练 SetFit（all-MiniLM-L6-v2，与 bi-encoder 一致）
  → 批量预测全部 490 家公司
  → 写入 predicted_sdg_tags 字段
  → MLflow 记录训练过程
  → 人工抽查 20-30 家验证质量

日常运行（新公司入库时立即触发）：
  新公司入库 → embedding 生成 → SetFit 预测
  → 写入 predicted_sdg_tags
  → 三步连续执行，不异步延迟
  → Evidently 监控标签分布漂移

Agent 搜索时：
  WHERE sdg_tags LIKE '%SDG7%'
     OR predicted_sdg_tags LIKE '%SDG7%'
  SDG 过滤覆盖率从 4% → ~90%
```

**为什么选 SetFit 而不是 NLI：**

| 指标 | SetFit | NLI |
|------|--------|-----|
| 准确率 | 59.1% | 37.6% |
| 推理速度 | 0.46ms | 31ms（慢67倍） |
| 训练数据需求 | 零标注 | 零标注 |

**开发优先级：**

```
Phase 1 必须完成：
  → SetFit 训练（零标注，一次性）
  → 批量预测 490 家写入 predicted_sdg_tags
  → 之后 SDG 筛选才有意义，Agent 才能正常工作

新公司日常入库：
  → 入库 → embedding → SetFit/LogReg 预测，三步连续
  → 不可跳过，否则新公司的 SDG 筛选失效
```

# Step 1：训练模型（~5分钟，CPU）
python -m ml.sdg_classifier train

# Step 2：批量写入 ChromaDB
python -m ml.sdg_classifier backfill

# 验证覆盖率
python -m ml.sdg_classifier stats

# 看效果
python inspect_db.py --sample 5

---

## 9. Airflow 调度策略

```
每天 08:00（轻量，~2分钟）
├── 爬取 listing 前 3 页
├── 对比 slug 列表，发现新企业
└── 新企业 → 解析 → 入库 → embedding 生成 → SetFit 预测
           （四步连续，数据完整后才算可用）

每周日 02:00（全量扫描，~15分钟）
├── 遍历所有 URL（约 490 条）
├── 对比 html_hash
├── 有变化 → 重新解析 + 更新 embedding + 重新预测 predicted_sdg_tags
├── 触发 Evidently 漂移检测报告
└── 记录到 scrape_log + MLflow

模型重训练（CI/CD 触发或手动）
├── 检查是否有新的 prediction_log 数据
├── 重新训练 SetFit
├── MLflow 对比新旧模型 F1
└── 自动晋升或保留 Champion 模型
```

---

## 10. 项目目录结构

```
sdgzero-intelligence/
├── .github/
│   └── workflows/
│       ├── test.yml               # PR 自动测试
│       └── ml-pipeline.yml        # 训练 + 评估 + 部署
│
├── frontend/                      # Next.js 前端
│   ├── pages/
│   │   ├── index.tsx              # 入口表单页
│   │   ├── results/[id].tsx       # 报告结果页（URL分享）
│   │   └── company/[slug].tsx     # 公司详情页
│   └── components/
│       ├── SearchForm.tsx         # 筛选表单组件
│       ├── CompanyCard.tsx        # 公司卡片组件
│       └── ReportView.tsx         # 报告展示组件
│
├── scraper/
│   ├── spider.py                  # 异步爬虫主逻辑
│   ├── parser.py                  # 字段解析
│   └── models.py                  # Pydantic 数据模型
│
├── db/
│   ├── migrations/                # Alembic 迁移文件
│   ├── schema.py                  # SQLAlchemy 模型定义
│   ├── queries.py                 # 查询方法（含三种搜索）
│   └── connection.py              # 连接池配置
│
├── pipeline/
│   ├── ingest.py                  # 清洗 + 结构化 + 入库
│   ├── embed.py                   # 向量化（bi-encoder）
│   └── update.py                  # 增量更新
│
├── ml/
│   ├── sdg_classifier.py          # SetFit 训练 + 批量预测
│   ├── clustering.py              # 企业画像聚类
│   ├── evaluate.py                # 评估报告
│   └── prompts/
│       ├── registry.py            # Prompt 版本管理
│       └── versions/              # 各版本 prompt 文件
│
├── agent/
│   ├── graph.py                   # LangGraph 流水线定义
│   ├── state.py                   # AgentState 定义
│   ├── search_agent.py            # SearchAgent（三种搜索工具）
│   ├── research_agent.py          # ResearchAgent（Tavily 三层策略）
│   ├── scoring_agent.py           # ScoringAgent（HyDE + Bi + Cross）
│   ├── report_agent.py            # ReportAgent（格式化报告）
│   ├── tools.py                   # 工具定义
│   ├── memory.py                  # 对话记忆管理
│   └── rag.py                     # RAG 检索逻辑
│
├── mcp_server/
│   └── server.py                  # MCP Server
│
├── api/
│   ├── main.py                    # FastAPI 入口
│   └── routes/
│       ├── search.py              # 搜索 + 会话管理（含 URL 分享）
│       ├── businesses.py          # 企业 CRUD
│       └── internal.py            # 运维接口
│
├── dags/
│   ├── daily_check.py             # 每日新增检测
│   ├── weekly_scan.py             # 每周全量扫描
│   └── model_retrain.py           # 模型重训练
│
├── tests/
│   ├── test_scraper.py
│   ├── test_queries.py
│   ├── test_agent.py
│   └── test_ml.py
│
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## 11. docker-compose 服务

```yaml
services:
  postgres:        # PostgreSQL 15 + pgvector 扩展
  redis:           # 缓存 + Celery broker
  airflow:         # 调度（webserver + scheduler + worker）
  mlflow:          # 实验追踪 UI → localhost:5000
  api:             # FastAPI → localhost:8000
  frontend:        # Next.js → localhost:3000
  celery-worker:   # 异步 Agent Pipeline 任务
  mcp-db-server:   # mcp-db-server → localhost:8001（直连 PostgreSQL，NL-to-SQL）
```

```bash
docker-compose up -d
```

---

## 12. Python 核心依赖

```toml
[tool.poetry.dependencies]
python = "^3.11"

# 数据采集
httpx = "^0.27"
beautifulsoup4 = "^4.12"
pydantic = "^2.0"

# 外部调研
tavily-python = "^0.3"           # Tavily extract + search

# 数据库
sqlalchemy = "^2.0"
alembic = "^1.13"
psycopg2-binary = "^2.9"
asyncpg = "^0.29"
pgvector = "^0.2"
redis = "^5.0"

# AI / ML
ollama = "^0.3"                  # 本地 LLM（Llama 3.1 8B，零成本）
groq = "^0.9"                    # Groq API（备选，极快，有免费额度）
langgraph = "^0.2"               # Multi-Agent 状态机
langchain = "^0.3"               # 工具定义等基础组件
sentence-transformers = "^3.0"   # Bi-encoder + Cross-encoder
setfit = "^1.0"                  # SDG 零样本分类器
scikit-learn = "^1.5"
numpy = "^1.26"
pandas = "^2.2"

# Agent / MCP
mcp = "^1.0"                     # MCP Server SDK

# MLOps
mlflow = "^2.15"
apache-airflow = "^2.9"
evidently = "^0.4"

# 服务
fastapi = "^0.115"
uvicorn = "^0.30"
celery = "^5.4"
structlog = "^24.0"
python-dotenv = "^1.0"
pytest = "^8.0"
pytest-asyncio = "^0.23"
```

---

## 13. 系统性能指标（目标）

| 指标 | 目标值 |
|------|--------|
| SearchAgent 响应时间 | < 500ms（缓存命中 < 50ms）|
| ResearchAgent（20家并行） | < 10s |
| ScoringAgent（HyDE + Cross-encoder） | < 5s |
| Agent 完整 Pipeline | < 20s |
| SDG 分类器 F1 | > 0.85 |
| Top-5 检索召回率 | > 90% |
| 系统可用性 | > 99% |
| Tavily 月度成本 | < $5（免费额度内） |

---

## 14. 与岗位需求的对应关系

| 技术点 | Role1 MLOps | Role2 AI Engineer | Role3 Agent |
|--------|------------|-------------------|-------------|
| MLflow + 实验追踪 | ✅ 核心 | - | - |
| Airflow Pipeline | ✅ 核心 | - | - |
| Evidently 监控 | ✅ | - | - |
| GitHub Actions CI/CD | ✅ | - | ✅ |
| Docker | ✅ | - | - |
| AWS S3 | ✅ | ✅ | - |
| RAG Architecture | - | ✅ 核心 | ✅ |
| MCP Server（自建） | - | ✅ 核心 | - |
| mcp-db-server 集成 | - | ✅ | ✅ |
| LangGraph Multi-Agent | - | ✅ | ✅ 核心 |
| AgentState 设计 | - | ✅ | ✅ 核心 |
| Tavily 外部调研集成 | - | ✅ | ✅ |
| HyDE + Schema 注入（SearchAgent） | - | ✅ 核心 | ✅ |
| Bi-encoder + Cross-encoder | - | ✅ 核心 | - |
| pgvector 语义搜索 | - | ✅ | - |
| SQL + 向量混合搜索 | - | ✅ | ✅ |
| SetFit 零样本分类 | ✅ | ✅ | - |
| Prompt 版本管理 | - | - | ✅ |
| 前端入口（Next.js） | - | - | ✅ |

---

## 15. 开发优先级（推荐顺序）

```
Phase 1（Week 1-2）：数据 + RAG 基础
  → SetFit 训练（零标注，一次性，必须先完成）
  → REST API 取数据 → 入库 → embedding → SetFit 预测 predicted_sdg_tags
  → ChromaDB → RAG 跑通
  → ChromaDB 阶段：hybrid = 语义搜索 + Python 过滤
  → 结果：SDG 筛选可用，可以演示基础语义搜索

Phase 2（Week 2-3）：Multi-Agent 核心
  → LangGraph AgentState 定义
  → SearchAgent（HyDE + Schema 注入 + 三种搜索工具 + 三级降级）
  → ResearchAgent（Tavily 三层策略）
  → ScoringAgent（Cross-encoder 精排 + LLM 理由生成）
  → ReportAgent（格式化报告）
  → 结果：完整 Agent Pipeline 跑通

Phase 3（Week 3-4）：前端 + 部署
  → Next.js 入口表单 + 结果页
  → 筛选条件 Hard/Soft 切换（每个过滤项旁加"必须/优先"按钮）
  → AgentState 新增 soft_filters 字段，ScoringAgent 读取作加分项
  → URL 分享（search_sessions 表）
  → FastAPI + Railway 部署上线
  → 结果：有可演示的线上地址

Phase 4（Week 4-5）：MLOps + pgvector 迁移
  → ChromaDB → pgvector 迁移
  → hybrid_search 升级为原生 SQL + 向量单条查询
  → MLflow 实验追踪
  → GitHub Actions CI/CD
  → Airflow DAG（入库流水线 + 模型重训练）
  → 结果：完整 MLOps 闭环

Phase 5（Week 5-6）：工程化 + MCP 完善
  → mcp-db-server 接入（依赖 pgvector 迁移完成，Phase 4 前无法使用）
  → SearchAgent SQL 生成改为调用 mcp-db-server MCP 工具
  → 自建 MCP Server 完善
  → AWS S3 存储 artifacts
  → Evidently 漂移监控
  → 结果：两种 MCP 入口均可演示，三个岗位技术栈全覆盖
```

---

## 附录A：Agent 开发顺序（详细）

```
Step 1：AgentState + SearchAgent（HyDE + 单工具）
  目标：HyDE 生成描述，LangGraph 图跑通，能返回企业列表
  (  HyDE 生成 50-120 token 短描述（采纳 ChatGPT 的建议）
     + 同时做 Query Expansion（3-5个等价表达）
     + 两者的向量做平均或分别检索取并集)
  验证：传入 user_company_desc，能拿到 hypothetical_partner_desc 和候选公司10家

Step 2：SearchAgent 三种搜索工具
  新增：sql_filter / hybrid_search
  验证：
    只有 filters → sql_filter
    只有描述    → semantic_search
    两者都有    → hybrid_search

Step 3：ResearchAgent（Tavily 三层）
  新增：Tavily extract + search 兜底
  验证：10家公司并行调研完成，每家有摘要

Step 4：ScoringAgent（Cross-encoder + 理由）
  输入：hypothetical_partner_desc（来自 SearchAgent）+ research_results
  Cross-encoder 精排 → Top-5
  LLM 生成推荐理由
  验证：Top-5 排序符合预期（人工判断）

Step 5：ReportAgent（报告格式化）
  新增：Markdown 报告 + 外联消息草稿
  验证：报告可读性（人工评估）

Step 6：前端 + URL 分享
  新增：Next.js 表单 + 结果页 + search_sessions 持久化
  验证：生成 URL，另一台设备能打开

Step 7：MCP Server（锦上添花）
  新增：标准化工具暴露
  验证：Claude Desktop 能调用
```

**开发原则：**
- 每一步都能独立演示，不要等全做完再测试
- 先用假数据验证 Agent 逻辑，再接真实数据库
- Phase 3 完成后就有完整演示链路，后续都是加分项