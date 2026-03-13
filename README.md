# arXiv-Paper-Assistant

一个基于 agno 框架构建的多智能体学术助手，用于管理本地 arXiv 论文 PDF，并提供基于知识库的智能问答。该项目展示了如何利用 agno 的 Agent、Team、Knowledge、Tool 等核心组件构建一个完整的 RAG应用。

## 特性

- **自动扫描与索引**：启动时扫描本地论文文件夹，对新 PDF 进行语义分块、向量化存储，避免重复索引。
- **arXiv 实时检索**：通过 arXiv API 按关键词检索最新论文，返回标题、摘要、作者及 arXiv ID。
- **全自动下载与解析**：根据 arXiv ID 自动下载 PDF 并永久索引至知识库，随后自动执行 PDF 解析、语义分块、向量化并持久化至 LanceDB。整个过程无需人工干预。
- **智能问答**：
  - 全局检索：跨所有已索引论文回答问题。
  - 聚焦模式：指定单篇论文进行深度精读，回答严格基于原文并附引用来源。
- **多智能体协作**：团队主管负责任务委派，网络检索与本地问答由不同 Agent 分工完成，提升复杂任务处理能力。

## 核心架构
1. **团队主管 (arXiv Team Leader)**

  - 核心职责：负责意图识别、任务委派（Delegation）以及全局上下文和多轮对话状态的管理。

  - 焦点机制：内置焦点论文（Focus Paper）追踪逻辑。能够记住用户当前正在研究的具体论文，并在委派任务时自动补充上下文，解决用户提问时“指代不明”（如“这篇论文的创新点是什么？”）的问题。

  - 工具访问：拥有查看已索引论文列表的权限，用于辅助校验用户的指代意图。

2. **外网情报员 (arXiv Researcher)**

  - 核心职责：负责学术前沿探索与文献检索。

  - 能力：通过调用 arXiv API 进行精准的高级关键词检索，解析 XML 数据，向用户提供包含标题、摘要、作者及 arXiv ID 的格式化文献列表。

3. **本地文献专家 (Local RAG Expert)**

  - 核心职责：作为唯一挂载本地向量知识库的智能体，负责文献的增量解析与深度问答。

  - 安全与红线：被严格限制在基于本地知识库检索内容作答，必须携带页码引用，杜绝大模型的“幻觉”（Hallucination）现象。

## 安装

### 前置要求
- Python 3.9+
- [Ollama](https://ollama.com/) 已安装并拉取嵌入模型 `bge-m3`（用于向量化）
- 可选：若使用 OpenAI 兼容接口，需配置 API Key

### 步骤
1. 克隆仓库
   ```bash
   git clone https://github.com/IsRivulet/arXiv-Paper-Assistant.git
   cd arxiv-paper-assistant
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 配置环境变量
   复制 `.env.example` 为 `.env`，并根据需要填写：
   ```
   OPENAI_API_KEY=your_api_key_here   # 如果使用 OpenAI 兼容接口
   ```

4. 启动 Ollama 并确保 `bge-m3` 模型可用
   ```bash
   ollama pull bge-m3
   ```

## 配置

在 `PaperDive.py` 中可以修改以下路径（默认适用于 Windows，可根据需要调整）：
- `PAPERS_DIR`：存放 PDF 论文的文件夹路径
- `SQLITE_DB_FILE`：SQLite 数据库文件路径（存储会话和知识元数据）
- `LANCEDB_URI`：LanceDB 向量数据库存储路径

## 使用方法

直接运行主程序：
```bash
python PaperDive.py
```

首次启动会自动扫描 `PAPERS_DIR` 中的 PDF 文件，并索引新论文。之后进入交互式命令行界面，支持以下指令：

- **扫描新论文**：输入 `扫描` 或 `scan` 手动触发索引（启动时已自动执行）。
- **列出已索引论文**：输入 `列表` 或 `list` 查看知识库中的所有论文。
- **检索 arXiv 论文**：输入 `找一下 大语言模型` 或 `search large language model` 查询 arXiv。
- **下载并加载指定论文**：输入 `加载 2301.12345` 或 `load https://arxiv.org/abs/2301.12345`。
- **提问**：
  - 直接问问题（如 `什么是注意力机制？`）将全局检索。
  - 先指定聚焦论文（如 `研究 2301.12345`），后续问题默认针对该论文。
- **退出**：输入 `exit`、`quit` 或 `bye`。


## 技术栈

- [agno](https://github.com/agno-ai/agno)：智能体框架
- LanceDB：向量数据库
- Ollama：本地嵌入模型（bge-m3）
- SQLite：元数据存储
- arXiv API：论文检索
- httpx：HTTP 客户端
