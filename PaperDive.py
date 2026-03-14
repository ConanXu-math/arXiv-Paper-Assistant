import os
import re
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from textwrap import dedent
from dotenv import load_dotenv
import httpx

from agno.agent import Agent
from agno.team import Team
from agno.db.sqlite import SqliteDb
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.content import ContentStatus
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.models.openai.like import OpenAILike
from agno.tools import tool
from agno.vectordb.lancedb import LanceDb, SearchType

load_dotenv()
# 将 agno 的日志级别设为 ERROR，屏蔽 WARNING 及以下
logging.getLogger("agno").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).parent
PAPERS_DIR = BASE_DIR / "arxiv_test" / "papers"
SQLITE_DB_FILE = str(BASE_DIR / "arxiv_test" / "state.db")
LANCEDB_URI = str(BASE_DIR / "arxiv_test" / "lancedb")

PAPERS_DIR.mkdir(parents=True, exist_ok=True)

agent_db = SqliteDb(
    db_file=SQLITE_DB_FILE, session_table="agent_sessions", memory_table="agent_memory"
)
knowledge_db = SqliteDb(
    db_file=SQLITE_DB_FILE,
    knowledge_table="knowledge_contents",
    session_table="knowledge_sessions",
)


vector_db = LanceDb(
    uri=LANCEDB_URI,
    table_name="arxiv_paper_chunks",
    search_type=SearchType.vector,
    embedder=OllamaEmbedder(
        id="bge-m3",
        dimensions=1024,
        timeout=120.0,
    ),
)


# 把文章按语义自然分割成多个块，每个块的字符数尽量接近 1200
semantic_chunking = SemanticChunking(chunk_size=1200, similarity_threshold=0.6)

pdf_reader = PDFReader(
    chunking_strategy=semantic_chunking,
    split_on_pages=True,
)

shared_llm = OpenAILike(
    id=os.getenv("LLM_MODEL_ID", "ecnu-plus"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://chat.ecnu.edu.cn/open/api/v1"),
    timeout=300.0,
)

shared_knowledge = Knowledge(
    name="arxiv_library",
    vector_db=vector_db,
    contents_db=knowledge_db,
    readers={"pdf": pdf_reader},
    max_results=8,
)


def _get_indexed_names() -> set:
    """查询 knowledge_db 中已成功完成索引的论文名称集合。"""
    try:
        contents, _ = shared_knowledge.get_content()
        return {
            c.name for c in contents if c.status == ContentStatus.COMPLETED and c.name
        }
    except Exception as e:
        print(f"[Debug] _get_indexed_names 出错: {e}")
        return set()


# ─────────────────────────────────────────────────────────────
# 工具定义
# ─────────────────────────────────────────────────────────────


def _perform_scan() -> str:
    """
    扫描本地论文文件夹，将尚未索引的新 PDF 写入向量知识库。
    已索引过的论文自动跳过，不重复处理，不删除历史数据。

    Returns:
        有新论文：索引结果列表。
        无新论文：现有论文列表 + 引导选项。
        文件夹为空：下载操作指引。
    """
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    # 情况一：文件夹为空
    if not pdf_files:
        return (
            f"论文文件夹目前为空。\n\n"
            f"文件夹路径：{PAPERS_DIR}\n\n"
            "请按以下步骤添加论文：\n"
            "  1. 访问 https://arxiv.org 搜索感兴趣的论文\n"
            "  2. 点击论文页面右侧的 [Download PDF] 下载\n"
            "  3. 将 PDF 文件移入上述文件夹（或告诉我 arXiv ID，我自动下载）\n\n"
            "也可告诉我感兴趣的研究方向，我来帮你找值得读的论文！"
        )

    # 情况二：差集识别新论文
    indexed_names = _get_indexed_names()
    new_files = [f for f in pdf_files if f.stem not in indexed_names]

    # 情况三：无新论文，全部已索引
    if not new_files:
        lines = [
            "扫描完成，没有发现新论文。\n",
            f"知识库中已有以下 {len(indexed_names)} 篇论文：\n",
        ]
        for i, name in enumerate(sorted(indexed_names), 1):
            lines.append(f"  [{i}] {name}")
        lines += [
            "\n您可以：",
            "  • 直接提问，我将在所有论文中检索作答",
            "  • 说「研究第N篇」聚焦单篇深度问答",
            "  • 说「帮我找 XXX 方向的新论文」在 arXiv 搜索",
            "  • 说「加载 arXiv ID」自动下载并索引新论文",
        ]
        return "\n".join(lines)

    # 情况四：有新论文，执行追加索引（不清除旧数据）
    success, failed = [], []
    for pdf_path in new_files:
        paper_id = pdf_path.stem
        try:
            shared_knowledge.insert(
                name=paper_id,
                path=str(pdf_path),
                reader=pdf_reader,
                # upsert=False,
                skip_if_exists=True,
            )
            success.append(paper_id)
        except Exception as e:
            failed.append(f"{pdf_path.name}：{e}")

    lines = [f" 已成功索引 {len(success)} 篇新论文：\n"]
    for i, pid in enumerate(success, 1):
        lines.append(f"  [{i}] {pid}")

    if failed:
        lines.append(f"\n以下 {len(failed)} 篇索引失败：")
        for err in failed:
            lines.append(f"  - {err}")

    all_indexed = _get_indexed_names()
    if len(all_indexed) > len(success):
        lines.append(f"\n知识库合计现有 {len(all_indexed)} 篇论文（含历史积累）。")

    lines.append("\n现在可以就任意论文提问了！")
    return "\n".join(lines)


@tool
def scan_and_index_new_papers() -> str:
    return _perform_scan()


@tool
def list_indexed_papers() -> str:
    """
    列出知识库中所有已成功索引的论文。

    【调用时机】：
    - 用户询问"有哪些论文"、"知识库里有什么"时

    Returns:
        已索引论文列表，含编号、文件名和本地文件状态。
    """
    indexed_names = _get_indexed_names()

    if not indexed_names:
        return (
            f"知识库目前为空。\n"
            f"请告诉我 arXiv ID，我自动下载；或将 PDF 放入 {PAPERS_DIR} 后说「扫描新论文」。"
        )

    lines = [f"知识库共有 {len(indexed_names)} 篇论文：\n"]
    for i, name in enumerate(sorted(indexed_names), 1):
        pdf_path = PAPERS_DIR / f"{name}.pdf"
        status = (
            "本地缓存存在" if pdf_path.exists() else " 源文件已移除（向量仍可检索）"
        )
        lines.append(f"  [{i}] {name}　{status}")

    lines += [
        "\n使用方式：",
        "  • 直接提问 → 跨所有论文检索作答",
        "  • 「研究第N篇」→ 定向深度问答",
    ]
    return "\n".join(lines)


def _fetch_arxiv_title(arxiv_id: str) -> str | None:
    """获取 arXiv 论文的标题，失败时返回 None。"""
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(abs_url, headers=headers)
            if resp.status_code != 200:
                return None
            # 解析 HTML 中的标题（简单正则）
            import re

            match = re.search(
                r"<title>arXiv:[\w.]+\s+(.*?)</title>", resp.text, re.DOTALL
            )
            if match:
                title = match.group(1).strip()
                # 移除可能的多余空白和换行
                title = re.sub(r"\s+", " ", title)
                return title
    except Exception:
        pass
    return None


@tool
def load_paper_for_deep_analysis(
    arxiv_url_or_id: str | float | int, expected_title: str | None = None
) -> str:
    """
    自动下载指定 arXiv 论文的 PDF，永久索引到向量知识库，支持后续深度精准问答。

    【调用时机】：
    - 用户明确说"读这篇"、"分析这篇"、"加载这篇"时
    - 用户通过指代（如"第一篇"、"那篇关于 Agent 的"）引用检索结果时

    【执行流程】：
    1. 校验 arXiv ID 格式（防止路径穿越攻击）
    2. 若知识库已有该论文 → 直接告知，跳过索引
    3. 若本地缓存已有 PDF → 跳过下载，直接索引
    4. 否则：伪装浏览器 UA 下载 → 持久化保存 → 语义分块 → 向量化 → 写入 LanceDB
    5. 额外步骤：获取 arXiv 标题并验证，避免 LLM 幻觉导致的 ID‑标题不匹配

    Args:
        arxiv_url_or_id: arXiv 完整 URL 或 arXiv ID（如 2301.12345 或 2301.12345v2）
        expected_title: 可选的预期标题，若提供则进行简单关键词匹配校验

    Returns:
        索引成功确认，或失败原因说明。
    """
    # ── 归一化 ID ────────────────────────────────────────────
    raw = str(arxiv_url_or_id).strip()
    if raw.startswith("http"):
        arxiv_id = raw.split("/abs/")[-1].split("/pdf/")[-1].removesuffix(".pdf")
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    else:
        arxiv_id = raw.removesuffix(".pdf").strip()
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"

    # [Fix-4] 路径穿越防御：仅允许字母、数字、点号、横线
    # 拒绝 "../etc/passwd" 或包含斜杠、空格的恶意输入
    if not re.match(r"^[\w.\-]+$", arxiv_id):
        return f"非法的 arXiv ID 格式：「{arxiv_id}」\n合法示例：2301.12345 或 2301.12345v2"

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    local_pdf_path = PAPERS_DIR / f"{arxiv_id}.pdf"

    try:
        # ── 已在知识库中？直接告知 ───────────────────────────
        if arxiv_id in _get_indexed_names():
            return (
                f"论文 **{arxiv_id}** 已在知识库中，无需重新索引。\n\n"
                f"可以直接提问，我将严格基于原文给出带引用的精准回答。"
            )

        # ── 本地无缓存时才下载 ───────────────────────────────
        if not local_pdf_path.exists():
            # 伪装真实浏览器 UA，绕过 arXiv 反爬拦截
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            print(f"[下载] 正在从 {pdf_url} 下载 PDF...")
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                resp = client.get(pdf_url, headers=headers)

            if resp.status_code != 200:
                return (
                    f"无法下载论文 PDF（状态码 {resp.status_code}）：{pdf_url}\n"
                    f"请检查 arXiv ID 是否正确，或网络是否可访问 arxiv.org。"
                )

            local_pdf_path.write_bytes(resp.content)
            print(f"[下载] 已保存至 {local_pdf_path}（{len(resp.content) // 1024} KB）")
        else:
            print(f"[缓存] 命中本地文件 {local_pdf_path}，跳过下载")

        # ── 获取 arXiv 标题并校验（避免 LLM 幻觉）─────────────
        title = _fetch_arxiv_title(arxiv_id)
        if title:
            print(f"[校验] 论文标题：{title}")
            # 简单关键词匹配（若用户提供了预期标题）
            if expected_title:
                import difflib

                ratio = difflib.SequenceMatcher(
                    None, expected_title.lower(), title.lower()
                ).ratio()
                if ratio < 0.6:  # 相似度阈值
                    print(
                        f"[警告] 下载的论文标题与预期差异较大，请确认是否为同一篇论文。"
                    )
                    print(f"      预期：{expected_title}")
                    print(f"      实际：{title}")
        else:
            print(f"[警告] 无法获取论文标题，请确认 arXiv ID {arxiv_id} 有效。")

        print(f"[索引] 正在向量化 {arxiv_id}，请稍候（大论文可能需要几分钟）...")
        shared_knowledge.insert(
            name=arxiv_id,
            path=str(local_pdf_path),
            reader=pdf_reader,
            skip_if_exists=True,
        )

        title_line = f"\n论文标题：{title}" if title else ""
        return (
            f"论文 **{arxiv_id}** 已成功下载并索引至知识库！{title_line}\n\n"
            f"原文链接：{abs_url}\n"
            f"本地缓存：{local_pdf_path}\n\n"
            f"现在可以就该论文的任何内容提问，"
            f"我将严格基于论文原文给出带引用的精准回答。\n"
            f"（若论文中未涉及您的问题，我会明确告知。）"
        )

    except Exception as e:
        return f"论文加载失败（{arxiv_id}）：{e}"


@tool
def search_arxiv_papers(query: str, max_results: int = 3) -> str:
    """
    在 arXiv 上检索最新学术论文，返回带 arXiv ID 的论文列表。

    【调用时机】：
    - 用户想探索某研究方向时
    - 用户说"帮我找 XXX 方向的论文"时
    - 知识库无论文，用户希望寻找新方向时

    检索后用户说「加载第N篇」，系统自动调用 load_paper_for_deep_analysis 完成下载索引。

    Args:
        query:       检索关键词（英文效果更佳）
        max_results: 返回数量（默认 3，最大 10）
    """
    import urllib.parse

    encoded_query = urllib.parse.quote(query.strip())
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=all:{encoded_query}"
        f"&start=0&max_results={min(max_results, 10)}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url)
            resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        if not entries:
            return f"未找到与「{query}」相关的论文，建议换用英文关键词重试。"

        lines = [f"arXiv 检索结果：「{query}」\n"]
        for i, entry in enumerate(entries, 1):
            title_elem = entry.find("atom:title", ns)
            title = (
                title_elem.text.strip().replace("\n", " ")
                if title_elem is not None and title_elem.text
                else "无标题"
            )
            summary_elem = entry.find("atom:summary", ns)
            summary = (
                summary_elem.text.strip().replace("\n", " ")
                if summary_elem is not None and summary_elem.text
                else "无摘要"
            )
            abs_elem = entry.find("atom:id", ns)
            abs_url = (
                abs_elem.text.strip() if abs_elem is not None and abs_elem.text else ""
            )
            arxiv_id = abs_url.split("/abs/")[-1] if abs_url else "未知ID"
            author_elems = entry.findall("atom:author", ns)
            authors = []
            for a in author_elems:
                name_elem = a.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)
            author_str = ", ".join(authors[:3]) if authors else "未知作者"
            if len(authors) > 3:
                author_str += "等"
            lines.append(
                f"**[{i}] {title}**\n"
                f"  - arXiv ID：`{arxiv_id}`\n"
                f"  - 作者：{author_str}\n"
                f"  - 摘要：{summary[:280]}…\n"
                f"  - 链接：{abs_url if abs_url else '无链接'}\n"
            )

        lines += [
            "─" * 48,
            " 说「加载第N篇」，我自动下载 PDF 并完成索引，全程无需手动操作。",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"arXiv 检索失败：{e}"


# ─────────────────────────────────────────────────────────────
# Agent 构建
# ─────────────────────────────────────────────────────────────

# 1. 角色 A：外网情报员
arxiv_researcher = Agent(
    name="arXiv Researcher",
    role="负责在 arXiv 上检索前沿学术论文",
    model=shared_llm,
    tools=[search_arxiv_papers],
    instructions="当用户需要寻找新方向、推荐论文时，你负责调用工具检索 arXiv，返回论文标题、摘要和 ID。",
    markdown=True,
)

# 2. 角色 B：本地文献专家（拥有知识库）
rag_expert = Agent(
    name="Local RAG Expert",
    role="负责本地 PDF 管理与基于知识库的深度精读问答",
    model=shared_llm,
    tools=[
        scan_and_index_new_papers,
        list_indexed_papers,
        load_paper_for_deep_analysis,
    ],
    knowledge=shared_knowledge,
    search_knowledge=True,
    add_knowledge_to_context=True,
    instructions=dedent("""
        你是本地文献精读专家。
        绝对红线：回答论文内容必须且只能基于知识库检索结果！
        
        【工具调用严格规范 - 必读】：
        你的可用工具包括 `search_knowledge_base` (用于检索论文内容) 和 `list_indexed_papers` (用于查看论文列表)。
        严禁将工具名称拼接或修改！必须准确使用上述工具名。
        
        【高级检索技巧 - 必读】：
        当用户或主管要求你“研究”、“总结”某篇论文（而不是问某个具体细节）时，不要拿指令去搜索！
        你应该将检索关键词（Query）设置为该论文核心章节的英文词汇，例如调用搜索时输入：
        "Abstract, Introduction, main contribution, conclusion" 
        这样才能命中论文的核心段落，从而给出总结。

        如果检索不到，明确回答"知识库中未找到相关内容"，绝不允许自己编造！
        每次陈述必须带上引用标识（如 [第3页]）。
    """),
    markdown=True,
)

# 3. 角色 C：团队主管 (Team Leader)
arxiv_team = Team(
    name="arXiv Team",
    model=shared_llm,
    members=[arxiv_researcher, rag_expert],
    # 主管掌控记忆和对话历史
    db=agent_db,
    num_history_runs=10,
    add_history_to_context=True,
    enable_agentic_memory=True,
    tools=[list_indexed_papers],
    instructions=dedent("""
        你是 arXiv 学术助理团队的主管。你的任务是将用户需求委派（Delegate）给最合适的专家。
        
        【核心状态管理：当前研究论文（Focus Paper）】
        1. 你必须通过对话历史记住用户当前正在“研究”、“阅读”或“聚焦”的某篇论文（Focus Paper）。
        2. 当用户明确表示要研究某篇论文时（例如“我要研究 2401.12345”或“看看第一篇”），将其设为当前的 Focus Paper。
        3. 之后，只要用户提问（如“它的创新点是什么？”、“讲了什么？”），就算他们没提论文名，你也要**默认**他们是在问这篇 Focus Paper。在委派给 `Local RAG Expert` 时，**必须在指令中强行加上这篇论文的名称或ID**（例如：“请在 2401.12345 中检索创新点”）。
        4. **关键拦截（强制）**：如果用户问了一个关于论文的具体问题，但当前上下文中**没有** Focus Paper，你也完全**不知道**他在说哪篇，**绝对不允许盲目搜索或瞎猜！** 你必须直接反问用户：“请问您具体想问哪篇文章？”
        
        【常规任务委派】：
        - 【探索/找新论文】：委派给 `arXiv Researcher`。
        - 【下载指定论文/深度问答某篇论文】：委派给 `Local RAG Expert`。
        
        【重要容错机制】：
        1. 如果用户提问存在指代不明（只说“这篇论文”且上下文中无记录），必须先调用 `list_indexed_papers` 辅助判断，若仍不确定，触发上述“关键拦截”反问。
        2. 当用户要求“研究”或“总结”某篇论文时，你在委派给 RAG Expert 时必须明确下达具体指令。例如：“请检索 2603.11046v1 的 Abstract 和 Conclusion，并用中文总结核心创新点”。不要只泛泛地说“研究这篇”。
    """),
    markdown=True,
    stream=True,
    session_id="arxiv-team-v2",
    user_id="researcher",
    show_members_responses=True,
)


def interactive_cli():
    """同步命令行交互入口。"""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    print("正在扫描论文文件夹...\n")
    scan_result = _perform_scan()
    print(scan_result)
    print()

    # ── 主交互循环 ──────────────────────────────────────────
    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！祝学术研究顺利 ")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "bye", "退出"):
            print("感谢使用，再见！")
            break

        arxiv_team.print_response(raw, stream=True)
        print()


if __name__ == "__main__":
    interactive_cli()
