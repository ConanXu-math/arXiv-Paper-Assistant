# arXiv-Paper-Assistant

一个基于 [agno](https://github.com/agno-ai/agno) 框架构建的多智能体学术助手，用于管理本地 arXiv 论文 PDF，并提供基于知识库的智能问答。该项目展示了如何利用 agno 的 Agent、Team、Knowledge、Tool 等核心组件构建一个完整的 RAG 应用。

## 特性

- **自动扫描与索引**：启动时扫描本地论文文件夹，对新 PDF 进行语义分块、向量化存储，避免重复索引。
- **arXiv 实时检索**：通过 arXiv API 按关键词检索最新论文，返回标题、摘要、作者及 arXiv ID。
- **全自动下载与解析**：根据 arXiv ID 自动下载 PDF 并永久索引至知识库，随后自动执行 PDF 解析、语义分块、向量化并持久化至 LanceDB。整个过程无需人工干预。
- **智能问答**：
  - 全局检索：跨所有已索引论文回答问题。
  - 聚焦模式：指定单篇论文进行深度精读，回答严格基于原文并附引用来源。
- **多智能体协作**：团队主管负责任务委派，网络检索与本地问答由不同 Agent 分工完成，提升复杂任务处理能力。
- **终端交互友好**：CLI 使用颜色框与分区展示用户消息、Agent 回复、工具调用等，信息层次清晰，便于长对话跟踪。

## 核心架构

1. **团队主管 (arXiv Team Leader)**
   - 核心职责：意图识别、任务委派（Delegation）以及全局上下文和多轮对话状态管理。
   - 焦点机制：内置焦点论文（Focus Paper）追踪，能记住当前研究的论文，在委派时自动补充上下文，解决“这篇论文的创新点是什么？”等指代问题。
   - 工具访问：可查看已索引论文列表，辅助校验用户指代意图。

2. **外网情报员 (arXiv Researcher)**
   - 核心职责：学术前沿探索与文献检索。
   - 能力：通过 arXiv API 进行关键词检索，解析 XML，返回标题、摘要、作者及 arXiv ID 的格式化列表。

3. **本地文献专家 (Local RAG Expert)**
   - 核心职责：挂载本地向量知识库，负责文献增量解析与深度问答。
   - 安全与红线：严格基于本地知识库检索作答，必须携带页码引用，减少幻觉。

## 安装

### 前置要求

- Python 3.9+
- [Ollama](https://ollama.com/) 已安装，用于运行嵌入模型 `bge-m3`
- （可选）若使用 OpenAI 兼容的 LLM 接口，需配置 API Key 与 Base URL

### 步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/IsRivulet/arXiv-Paper-Assistant.git
   cd arXiv-Paper-Assistant
   ```

2. **创建并激活虚拟环境（推荐）**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   ```bash
   cp .env.example .env
   ```
   编辑 `.env`，填写 LLM 相关配置（二选一或按需）：
   - **OpenAI 兼容接口**：`OPENAI_API_KEY=your_key`
   - **自定义接口（如 ChatECNU）**：
     ```
     LLM_API_KEY=your_api_key
     LLM_BASE_URL=https://your-api-base/v1
     LLM_MODEL_ID=your-model-id
     ```

5. **拉取嵌入模型并启动 Ollama**
   ```bash
   ollama pull bge-m3
   ollama serve   # 建议在单独终端常驻运行
   ```

## 配置

- **数据路径**：在 `PaperDive.py` 中，数据目录基于项目根目录自动设置：
  - `arxiv_test/papers/`：存放下载的 PDF
  - `arxiv_test/state.db`：SQLite（会话与知识元数据）
  - `arxiv_test/lancedb/`：LanceDB 向量库
  如需修改，可调整 `BASE_DIR` 或其下的 `PAPERS_DIR`、`SQLITE_DB_FILE`、`LANCEDB_URI`。

## 使用方法

启动主程序：

```bash
python PaperDive.py
```

首次启动会自动扫描 `arxiv_test/papers` 中的 PDF 并索引新论文，随后进入交互式 CLI。常用指令：

| 操作           | 指令示例 |
|----------------|----------|
| 扫描新论文     | `scan` / `扫描` |
| 列出已索引论文 | `list` / `列表` |
| 检索 arXiv     | `search large language model` / `找一下 大语言模型` |
| 下载并加载论文 | `load 2301.12345` 或 `load https://arxiv.org/abs/2301.12345` |
| 聚焦单篇问答   | `研究 2301.12345`，然后直接提问 |
| 退出           | `exit` / `quit` / `bye` |

- 直接提问（如「什么是注意力机制？」）会在所有已索引论文中检索作答。
- 先输入「研究 &lt;arXiv ID&gt;」再提问，则后续回答默认针对该篇，并带引用。

## 技术栈

- [agno](https://github.com/agno-ai/agno)：智能体框架
- LanceDB：向量数据库
- Ollama（bge-m3）：本地文本嵌入
- SQLite：会话与知识元数据
- arXiv API：论文检索
- httpx：HTTP 客户端
