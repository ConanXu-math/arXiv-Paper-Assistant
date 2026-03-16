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
from agno.knowledge.reader.pdf_reader import PDFReader  # kept as fallback
from ocr_pdf_reader import OcrPDFReader
from structure_extractor import (
    extract_paper_structure,
    extract_paper_summary,
    find_section_for_page,
    find_elements_on_page,
    format_structure_for_display,
)
from agno.models.openai.like import OpenAILike
from agno.tools import tool
from agno.vectordb.lancedb import LanceDb, SearchType

load_dotenv()
# 抑制 numpy 除以零警告
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
# 将 agno 的日志级别设为 ERROR，屏蔽 WARNING 及以下
logging.getLogger("agno").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).parent
PAPERS_DIR = BASE_DIR / "arxiv_test" / "papers"
NOTES_DIR = BASE_DIR / "notes"
SQLITE_DB_FILE = str(BASE_DIR / "arxiv_test" / "state.db")
LANCEDB_URI = str(BASE_DIR / "arxiv_test" / "lancedb")
TEAM_SESSION_ID = "arxiv-team-v4"

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
# embedder 必须显式指定，否则默认用 OpenAIEmbedder（需要 OPENAI_API_KEY）
semantic_chunking = SemanticChunking(
    embedder="minishlab/potion-base-32M",
    chunk_size=1200,
    similarity_threshold=0.6,
)

pdf_reader = OcrPDFReader(
    ocr_url="https://edusys5.sii.edu.cn/ocr",
    dpi=150,
    max_workers=8,
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


def _cleanup_stuck_processing():
    """清理卡在 processing 状态的记录，避免重启后无法重新索引。"""
    try:
        conn = sqlite3.connect(SQLITE_DB_FILE)
        cursor = conn.execute(
            "DELETE FROM knowledge_contents WHERE status = 'processing'"
        )
        n = cursor.rowcount
        conn.commit()
        conn.close()
        if n > 0:
            print(f"[清理] 已移除 {n} 条卡住的索引记录，将重新索引。", flush=True)
    except Exception:
        pass


_POLLUTION_PATTERNS = ["<function=", "</tool_call>", "<parameter="]


def _cleanup_polluted_session():
    """检测并清除 session 历史中被污染的工具调用记录。

    当 LLM 尝试调用未注册的工具时会输出原始文本（如 <function=...>），
    这些错误模式被持久化到 session 后会导致后续对话持续模仿。
    """
    try:
        conn = sqlite3.connect(SQLITE_DB_FILE)
        row = conn.execute(
            "SELECT runs FROM agent_sessions WHERE session_id = ?",
            (TEAM_SESSION_ID,),
        ).fetchone()
        if not row or not row[0]:
            conn.close()
            return
        runs_text = row[0]
        if any(p in runs_text for p in _POLLUTION_PATTERNS):
            conn.execute(
                "DELETE FROM agent_sessions WHERE session_id = ?",
                (TEAM_SESSION_ID,),
            )
            conn.commit()
            print(
                f"[清理] 检测到对话历史包含异常工具调用模式，已自动清除以确保工具正常执行。",
                flush=True,
            )
        conn.close()
    except Exception as e:
        print(f"[清理] session 检测失败（不影响使用）: {e}", flush=True)


# ─────────────────────────────────────────────────────────────
# 论文结构存储（SQLite）
# ─────────────────────────────────────────────────────────────

import json
import sqlite3

def _init_tables():
    conn = sqlite3.connect(SQLITE_DB_FILE)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS paper_structures ("
        "  paper_id TEXT PRIMARY KEY,"
        "  structure_json TEXT,"
        "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS paper_pages ("
        "  paper_id TEXT,"
        "  page_num INTEGER,"
        "  content TEXT,"
        "  PRIMARY KEY (paper_id, page_num)"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS paper_summaries ("
        "  paper_id TEXT PRIMARY KEY,"
        "  title TEXT,"
        "  abstract TEXT,"
        "  proof_approaches TEXT,"
        "  core_techniques TEXT,"
        "  field_tags TEXT,"
        "  content_tags TEXT,"
        "  technique_tags TEXT,"
        "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    conn.commit()
    conn.close()

_init_tables()


def save_paper_structure(paper_id: str, structure: dict) -> None:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    conn.execute(
        "INSERT OR REPLACE INTO paper_structures (paper_id, structure_json) VALUES (?, ?)",
        (paper_id, json.dumps(structure, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def load_paper_structure(paper_id: str) -> dict | None:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    row = conn.execute(
        "SELECT structure_json FROM paper_structures WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


# ── paper_pages CRUD ──────────────────────────────────────────

def save_paper_pages(paper_id: str, pages: list[str]) -> None:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    for i, content in enumerate(pages):
        conn.execute(
            "INSERT OR REPLACE INTO paper_pages (paper_id, page_num, content) VALUES (?, ?, ?)",
            (paper_id, i + 1, content),
        )
    conn.commit()
    conn.close()


def load_paper_pages(paper_id: str, start: int = 1, end: int = 0) -> list[tuple[int, str]]:
    """Return list of (page_num, content). end=0 means start page only."""
    if end <= 0:
        end = start
    conn = sqlite3.connect(SQLITE_DB_FILE)
    rows = conn.execute(
        "SELECT page_num, content FROM paper_pages "
        "WHERE paper_id = ? AND page_num >= ? AND page_num <= ? ORDER BY page_num",
        (paper_id, start, end),
    ).fetchall()
    conn.close()
    return rows


def get_paper_page_count(paper_id: str) -> int:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    row = conn.execute(
        "SELECT MAX(page_num) FROM paper_pages WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else 0


# ── paper_summaries CRUD ──────────────────────────────────────

def save_paper_summary(paper_id: str, summary: dict) -> None:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    conn.execute(
        "INSERT OR REPLACE INTO paper_summaries "
        "(paper_id, title, abstract, proof_approaches, core_techniques, "
        " field_tags, content_tags, technique_tags) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            paper_id,
            summary.get("title", ""),
            summary.get("abstract", ""),
            json.dumps(summary.get("proof_approaches", {}), ensure_ascii=False),
            json.dumps(summary.get("core_techniques", []), ensure_ascii=False),
            json.dumps(summary.get("field_tags", []), ensure_ascii=False),
            json.dumps(summary.get("content_tags", []), ensure_ascii=False),
            json.dumps(summary.get("technique_tags", []), ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def load_paper_summary(paper_id: str) -> dict | None:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    row = conn.execute(
        "SELECT title, abstract, proof_approaches, core_techniques, "
        "field_tags, content_tags, technique_tags FROM paper_summaries WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "title": row[0] or "",
        "abstract": row[1] or "",
        "proof_approaches": json.loads(row[2]) if row[2] else {},
        "core_techniques": json.loads(row[3]) if row[3] else [],
        "field_tags": json.loads(row[4]) if row[4] else [],
        "content_tags": json.loads(row[5]) if row[5] else [],
        "technique_tags": json.loads(row[6]) if row[6] else [],
    }


def load_all_paper_summaries() -> list[dict]:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    rows = conn.execute(
        "SELECT paper_id, title, field_tags, content_tags, technique_tags FROM paper_summaries"
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "paper_id": r[0],
            "title": r[1] or "",
            "field_tags": json.loads(r[2]) if r[2] else [],
            "content_tags": json.loads(r[3]) if r[3] else [],
            "technique_tags": json.loads(r[4]) if r[4] else [],
        })
    return results


def diagnose_paper(paper_id: str) -> str:
    """
    诊断某篇论文在本地库中的结构/摘要/原文状态（用于排查「结构数据暂时无法获取」等问题）。
    可在项目根目录运行: python -c "from PaperDive import diagnose_paper; print(diagnose_paper('sat-matching'))"
    """
    lines = [f"论文 ID: {paper_id}", "─" * 40]
    pdf_path = PAPERS_DIR / f"{paper_id}.pdf"
    lines.append(f"PDF 文件: {'存在' if pdf_path.exists() else '不存在'}")
    # 知识库向量
    try:
        contents, _ = shared_knowledge.get_content()
        vec_count = sum(1 for c in contents if c.name == paper_id)
        lines.append(f"向量块数: {vec_count}")
    except Exception as e:
        lines.append(f"向量: 查询异常 — {e}")
    # 结构
    st = load_paper_structure(paper_id)
    if st is None:
        lines.append("结构: 无")
    else:
        n_sec = len(st.get("sections", []))
        n_thm = len(st.get("theorems", []))
        n_def = len(st.get("definitions", []))
        lines.append(f"结构: 有 — {n_sec} 章节, {n_thm} 定理/引理, {n_def} 定义")
        raw = json.dumps(st, ensure_ascii=False)[:300]
        lines.append(f"结构预览: {raw}...")
    # 摘要
    sm = load_paper_summary(paper_id)
    if sm is None:
        lines.append("摘要: 无")
    else:
        title = (sm.get("title") or "")[:60]
        lines.append(f"摘要: 有 — title: {title}...")
    # 原文页数
    n_pages = get_paper_page_count(paper_id)
    lines.append(f"原文页数: {n_pages}")
    lines.append("─" * 40)
    return "\n".join(lines)


def _build_page_metadata(structure: dict, total_pages: int) -> dict[int, dict]:
    """Build per-page metadata dict from a structure for OcrPDFReader."""
    metadata: dict[int, dict] = {}
    for i in range(total_pages):
        page_num = i + 1
        section = find_section_for_page(structure, page_num)
        elements = find_elements_on_page(structure, page_num)
        meta: dict[str, str] = {}
        if section:
            meta["section"] = section
        if elements:
            meta["element_types"] = ",".join(elements)
        if meta:
            metadata[i] = meta
    return metadata


def _extract_and_store_structure(paper_id: str, pages: list[str]) -> dict:
    """Run two-phase structure extraction, store result, return structure.
    On any failure (LLM/timeout/JSON), falls back to regex-only and still saves.
    """
    print(f"[结构] 正在提取 {paper_id} 的论文结构...", flush=True)
    structure = None
    try:
        structure = extract_paper_structure(pages, llm=shared_llm)
    except Exception as e:
        print(f"[结构] 完整提取异常，改用仅正则结果: {e}", flush=True)
        try:
            structure = extract_paper_structure(pages, llm=None)
        except Exception as e2:
            print(f"[结构] 正则提取也失败: {e2}", flush=True)
            structure = {"sections": [], "theorems": [], "proofs": [], "definitions": [], "key_equations": []}
    if structure is None:
        structure = {"sections": [], "theorems": [], "proofs": [], "definitions": [], "key_equations": []}
    try:
        save_paper_structure(paper_id, structure)
    except Exception as se:
        print(f"[结构] 写入数据库失败: {se}", flush=True)
        raise
    n_thm = len(structure.get("theorems", []))
    n_def = len(structure.get("definitions", []))
    n_sec = len(structure.get("sections", []))
    print(
        f"[结构] 完成: {n_sec} 个章节, {n_thm} 个定理/引理, {n_def} 个定义",
        flush=True,
    )
    return structure


def _extract_and_store_summary(paper_id: str, pages: list[str], structure: dict) -> dict | None:
    """Run LLM summary extraction, store result, return summary dict."""
    print(f"[摘要] 正在为 {paper_id} 生成高层摘要...", flush=True)
    summary = extract_paper_summary(pages, structure, llm=shared_llm)
    if summary:
        save_paper_summary(paper_id, summary)
        tags = (
            summary.get("field_tags", [])
            + summary.get("content_tags", [])
            + summary.get("technique_tags", [])
        )
        print(f"[摘要] 完成: 标签={', '.join(tags[:6])}", flush=True)
    else:
        print(f"[摘要] LLM 摘要提取失败，跳过", flush=True)
    return summary


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
    _cleanup_stuck_processing()
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
    total = len(new_files)
    print(f"发现 {total} 篇新论文，开始索引（大论文可能需数分钟）...\n", flush=True)
    for i, pdf_path in enumerate(new_files, 1):
        paper_id = pdf_path.stem
        print(f"[{i}/{total}] 正在索引: {paper_id} ...", flush=True)
        try:
            shared_knowledge.insert(
                name=paper_id,
                path=str(pdf_path),
                reader=pdf_reader,
                # upsert=False,
                skip_if_exists=True,
            )
            success.append(paper_id)
            print(f"[{i}/{total}] 索引完成: {paper_id}", flush=True)
            # 持久化 OCR 原文 + 结构/摘要提取（可选增强，失败不影响索引）
            if pdf_reader.last_ocr_pages:
                try:
                    save_paper_pages(paper_id, pdf_reader.last_ocr_pages)
                    print(f"[{i}/{total}] OCR 原文已存储 ({len(pdf_reader.last_ocr_pages)} 页)", flush=True)
                except Exception as pe:
                    print(f"[{i}/{total}] 原文存储失败: {pe}", flush=True)
                structure = None
                try:
                    structure = _extract_and_store_structure(paper_id, pdf_reader.last_ocr_pages)
                except Exception as se:
                    print(f"[{i}/{total}] 结构提取失败（不影响检索）: {se}", flush=True)
                try:
                    _extract_and_store_summary(
                        paper_id,
                        pdf_reader.last_ocr_pages,
                        structure or {},
                    )
                except Exception as sme:
                    print(f"[{i}/{total}] 摘要提取失败（不影响检索）: {sme}", flush=True)
        except Exception as e:
            failed.append(f"{pdf_path.name}：{e}")
            print(f"[{i}/{total}] 失败: {paper_id} — {e}", flush=True)

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
        _cleanup_stuck_processing()
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

        structure_info = ""
        summary_info = ""
        if pdf_reader.last_ocr_pages:
            try:
                save_paper_pages(arxiv_id, pdf_reader.last_ocr_pages)
            except Exception as pe:
                print(f"[原文] 存储失败: {pe}", flush=True)
            structure = None
            try:
                structure = _extract_and_store_structure(arxiv_id, pdf_reader.last_ocr_pages)
                n_thm = len(structure.get("theorems", []))
                n_def = len(structure.get("definitions", []))
                n_sec = len(structure.get("sections", []))
                structure_info = f"\n结构提取：{n_sec} 个章节, {n_thm} 个定理/引理, {n_def} 个定义"
            except Exception as se:
                print(f"[结构] 提取失败（不影响检索）: {se}", flush=True)
            try:
                sm = _extract_and_store_summary(arxiv_id, pdf_reader.last_ocr_pages, structure or {})
                if sm:
                    tags = sm.get("field_tags", []) + sm.get("technique_tags", [])
                    summary_info = f"\n摘要标签：{', '.join(tags[:4])}"
            except Exception as sme:
                print(f"[摘要] 提取失败（不影响检索）: {sme}", flush=True)

        title_line = f"\n论文标题：{title}" if title else ""
        return (
            f"论文 **{arxiv_id}** 已成功下载并索引至知识库！{title_line}"
            f"{structure_info}{summary_info}\n\n"
            f"原文链接：{abs_url}\n"
            f"本地缓存：{local_pdf_path}\n\n"
            f"现在可以就该论文的任何内容提问，"
            f"我将严格基于论文原文给出带引用的精准回答。\n"
            f"可使用 `browse_paper_catalog` 浏览论文索引，"
            f"或 `get_paper_overview` 查看详细概要。\n"
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
# 工具定义：笔记保存
# ─────────────────────────────────────────────────────────────


@tool
def save_note(filename: str, content: str) -> str:
    """
    将笔记内容保存到本地文件（Markdown 格式）。

    【调用时机】：
    - 用户要求保存总结、笔记或任何文本内容时。

    Args:
        filename: 文件名（不含扩展名），将自动添加 .md 后缀。
        content: 要保存的文本内容。

    Returns:
        成功或失败消息。
    """
    import os
    from pathlib import Path

    if not filename.endswith(".md"):
        filename += ".md"
    path = NOTES_DIR / filename
    try:
        path.write_text(content, encoding="utf-8")
        return f"笔记已保存至：{path}"
    except Exception as e:
        return f"保存失败：{e}"


@tool
def list_notes() -> str:
    """
    列出所有已保存的笔记文件。

    【调用时机】：
    - 用户询问“有哪些笔记”或“查看已保存的笔记”时。

    Returns:
        笔记文件列表，包含文件名和大小。
    """
    import os
    from pathlib import Path

    files = list(NOTES_DIR.glob("*.md"))
    if not files:
        return "暂无笔记文件。"
    lines = [f"共有 {len(files)} 个笔记文件："]
    for i, f in enumerate(sorted(files), 1):
        size = f.stat().st_size
        lines.append(f"  [{i}] {f.name} ({size} 字节)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 工具定义：结构化检索
# ─────────────────────────────────────────────────────────────


@tool
def get_paper_structure(paper_id: str) -> str:
    """
    获取论文的结构化大纲：章节层级、定理/引理列表、定义、证明、关键公式。

    【调用时机】：
    - 用户说"论文结构是什么"、"有哪些定理"、"列出定义"时
    - 在深入提问某篇论文的具体定理/证明前，先调用此工具了解全貌

    Args:
        paper_id: 论文在知识库中的名称（通常是 arXiv ID，如 2301.12345）

    Returns:
        格式化的论文结构大纲（Markdown），包含章节、定理、定义、证明列表。
        若未找到结构数据，返回提示信息。
    """
    structure = load_paper_structure(paper_id)
    if structure is None:
        return (
            f"未找到论文 {paper_id} 的结构数据。\n"
            f"可能原因：该论文在结构提取功能上线前已索引。\n"
            f"建议：删除后重新加载该论文以触发结构提取。"
        )
    return format_structure_for_display(structure)


@tool
def search_structured(
    query: str,
    paper_id: str = "",
    element_type: str = "",
) -> str:
    """
    在知识库中按结构类型精准检索论文内容。

    【调用时机】：
    - 用户问"定理3.1是什么"→ element_type="theorem"
    - 用户问"定理3.1怎么证明的"→ element_type="proof"
    - 用户问"XX的定义"→ element_type="definition"
    - 需要精确定位特定类型内容时

    Args:
        query: 检索关键词（如 "theorem 3.1 statement", "proof of main theorem"）
        paper_id: 可选，限定在某篇论文内检索（论文名称/arXiv ID）
        element_type: 可选，按结构类型过滤。
                     可选值: theorem, proof, definition, equation

    Returns:
        匹配的文本块列表，带页码和结构类型标注。
    """
    filters: dict[str, any] = {}
    if paper_id:
        filters["name"] = paper_id
    if element_type:
        filters["element_types"] = element_type

    results = vector_db.search(query, limit=8, filters=filters if filters else None)

    if not results:
        filter_desc = ""
        if paper_id:
            filter_desc += f" 论文={paper_id}"
        if element_type:
            filter_desc += f" 类型={element_type}"
        return f"未找到匹配结果。查询: '{query}'{filter_desc}\n建议：尝试放宽过滤条件或换用不同关键词。"

    lines = [f"找到 {len(results)} 条结果：\n"]
    for i, doc in enumerate(results, 1):
        meta = doc.meta_data or {}
        page = meta.get("page", "?")
        section = meta.get("section", "")
        etypes = meta.get("element_types", "")
        header_parts = [f"p.{page}"]
        if section:
            header_parts.append(section)
        if etypes:
            header_parts.append(f"[{etypes}]")
        header = " | ".join(header_parts)

        content_preview = (doc.content or "")[:400]
        lines.append(f"**[{i}]** ({header})\n{content_preview}\n")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 工具定义：三级递进（全局索引 → 概要 → 深读）
# ─────────────────────────────────────────────────────────────


@tool
def browse_paper_catalog() -> str:
    """
    返回所有已索引论文的紧凑索引：每篇含标题和三维标签（领域/内容/技巧）。

    【调用时机】：
    - 用户提问时，先调用此工具浏览全局索引，挑出可能相关的论文
    - 用户问"有哪些论文"、"知识库里有什么"时

    Returns:
        紧凑论文目录，每篇 2 行（标题 + 三维标签），适合一次性通读筛选。
    """
    summaries = load_all_paper_summaries()
    if not summaries:
        indexed = _get_indexed_names()
        if not indexed:
            return "知识库为空，请先添加论文。"
        lines = [f"知识库有 {len(indexed)} 篇论文（尚无摘要信息，可能需要重新索引）："]
        for i, name in enumerate(sorted(indexed), 1):
            lines.append(f"  {i}. [{name}]")
        return "\n".join(lines)

    lines = [f"论文索引（共 {len(summaries)} 篇）：\n"]
    for i, s in enumerate(summaries, 1):
        pid = s["paper_id"]
        title = s["title"] or pid
        field = ", ".join(s.get("field_tags", [])) or "—"
        content = ", ".join(s.get("content_tags", [])) or "—"
        technique = ", ".join(s.get("technique_tags", [])) or "—"
        lines.append(f"{i}. [{pid}] {title}")
        lines.append(f"   领域: {field} | 内容: {content} | 技巧: {technique}")
    lines.append(f"\n提示：对感兴趣的论文调用 get_paper_overview(paper_id) 查看详细概要。")
    return "\n".join(lines)


@tool
def get_paper_overview(paper_id: str) -> str:
    """
    获取论文的详细概要：主要内容、章节结构、证明思路、核心技巧。

    【调用时机】：
    - 通过 browse_paper_catalog 筛出候选后，逐篇调用此工具精选
    - 用户问某篇论文"讲了什么"、"主要内容"、"创新点"时

    Args:
        paper_id: 论文在知识库中的名称（通常是 arXiv ID 或文件名）

    Returns:
        格式化的论文详细概要（Markdown）。
    """
    summary = load_paper_summary(paper_id)
    structure = load_paper_structure(paper_id)

    if not summary and not structure:
        return (
            f"未找到论文 {paper_id} 的概要或结构数据。\n"
            f"可能原因：该论文在此功能上线前已索引。\n"
            f"建议：删除后重新加载以触发摘要提取。"
        )

    lines: list[str] = []

    if summary:
        title = summary.get("title", paper_id)
        lines.append(f"# {title}\n")

        field_tags = summary.get("field_tags", [])
        content_tags = summary.get("content_tags", [])
        technique_tags = summary.get("technique_tags", [])
        if field_tags or content_tags or technique_tags:
            lines.append(f"**领域**: {', '.join(field_tags) if field_tags else '—'}")
            lines.append(f"**内容**: {', '.join(content_tags) if content_tags else '—'}")
            lines.append(f"**技巧**: {', '.join(technique_tags) if technique_tags else '—'}\n")

        abstract = summary.get("abstract", "")
        if abstract:
            lines.append(f"## 主要内容\n{abstract}\n")

        core_techniques = summary.get("core_techniques", [])
        if core_techniques:
            lines.append("## 核心技巧")
            for t in core_techniques:
                lines.append(f"- {t}")
            lines.append("")

        proof_approaches = summary.get("proof_approaches", {})
        if proof_approaches:
            lines.append("## 证明思路")
            for thm, approach in proof_approaches.items():
                lines.append(f"- **{thm}**: {approach}")
            lines.append("")

    if structure:
        sections = structure.get("sections", [])
        if sections:
            lines.append("## 章节大纲")
            for sec in sections:
                indent = "  " * (sec.get("level", 1) - 1)
                lines.append(f"{indent}- **{sec['id']}** {sec.get('title', '')} (p.{sec.get('page', '?')})")
            lines.append("")

        theorems = structure.get("theorems", [])
        if theorems:
            lines.append("## 定理/引理")
            for thm in theorems:
                stmt = thm.get("statement", "")[:120]
                lines.append(f"- **{thm.get('label', thm['id'])}** (p.{thm.get('page', '?')}): {stmt}")
            lines.append("")

    page_count = get_paper_page_count(paper_id)
    if page_count:
        lines.append(f"---\n全文共 {page_count} 页，可用 `read_paper_pages` 或 `read_paper_section` 深读原文。")

    return "\n".join(lines) if lines else f"论文 {paper_id} 的概要信息为空。"


@tool
def read_paper_pages(paper_id: str, start_page: int, end_page: int = 0) -> str:
    """
    读取论文指定页的 OCR 原文。

    【调用时机】：
    - 看完概要后需要深读某几页原文时
    - 需要查看某个证明、定义的完整内容时

    Args:
        paper_id: 论文 ID
        start_page: 起始页码（从 1 开始）
        end_page: 结束页码（包含），为 0 时只读 start_page 一页

    Returns:
        指定页的 OCR 原文。
    """
    if end_page <= 0:
        end_page = start_page
    if start_page < 1:
        return "页码必须从 1 开始。"
    if end_page - start_page > 10:
        return "一次最多读取 10 页，请缩小范围。"

    total = get_paper_page_count(paper_id)
    if total == 0:
        return f"未找到论文 {paper_id} 的原文数据。可能需要重新索引。"

    rows = load_paper_pages(paper_id, start_page, end_page)
    if not rows:
        return f"论文 {paper_id} 没有第 {start_page}-{end_page} 页的数据（全文共 {total} 页）。"

    lines = [f"论文 {paper_id} 第 {start_page}-{end_page} 页（共 {total} 页）：\n"]
    for page_num, content in rows:
        lines.append(f"--- 第 {page_num} 页 ---")
        lines.append(content if content else "[空白页]")
        lines.append("")

    return "\n".join(lines)


@tool
def read_paper_section(paper_id: str, section_id: str) -> str:
    """
    读取论文指定章节的原文（根据结构自动定位页码范围）。

    【调用时机】：
    - 看完概要后需要深读某个章节时
    - 用户说"读第3章"、"看 section 2.1"时

    Args:
        paper_id: 论文 ID
        section_id: 章节 ID（如 "sec1", "sec2.1"），可从 get_paper_overview 获取

    Returns:
        该章节覆盖页的 OCR 原文。
    """
    structure = load_paper_structure(paper_id)
    if not structure:
        return f"未找到论文 {paper_id} 的结构数据。"

    sections = structure.get("sections", [])
    target = None
    target_idx = -1
    for idx, sec in enumerate(sections):
        if sec["id"] == section_id or sec.get("title", "").lower() == section_id.lower():
            target = sec
            target_idx = idx
            break

    if not target:
        available = ", ".join(f"{s['id']}({s.get('title', '')})" for s in sections)
        return f"未找到章节 {section_id}。可用章节：{available}"

    start_page = target.get("page", 1)
    if target_idx + 1 < len(sections):
        end_page = sections[target_idx + 1].get("page", start_page)
        if end_page > start_page:
            end_page -= 1
    else:
        total = get_paper_page_count(paper_id)
        end_page = total if total > 0 else start_page

    end_page = min(end_page, start_page + 9)

    rows = load_paper_pages(paper_id, start_page, end_page)
    if not rows:
        return f"未找到论文 {paper_id} 第 {start_page}-{end_page} 页的原文数据。"

    sec_title = target.get("title", section_id)
    lines = [f"章节 [{section_id}] {sec_title}（p.{start_page}-{end_page}）：\n"]
    for page_num, content in rows:
        lines.append(f"--- 第 {page_num} 页 ---")
        lines.append(content if content else "[空白页]")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 工具定义：删除 / 重索引
# ─────────────────────────────────────────────────────────────

_ALL_TARGETS = ["vector", "structure", "summary", "pages"]


def _delete_paper_data(paper_id: str, targets: list[str]) -> list[str]:
    """Delete specified data for a paper. Returns list of successfully deleted targets."""
    if "all" in targets:
        targets = list(_ALL_TARGETS)

    deleted: list[str] = []

    if "vector" in targets:
        try:
            contents, _ = shared_knowledge.get_content()
            for c in contents:
                if c.name == paper_id and c.id is not None:
                    shared_knowledge.remove_content_by_id(c.id)
            deleted.append("vector")
        except Exception as e:
            print(f"[删除] 向量数据删除失败: {e}", flush=True)

    conn = sqlite3.connect(SQLITE_DB_FILE)
    try:
        if "structure" in targets:
            conn.execute("DELETE FROM paper_structures WHERE paper_id = ?", (paper_id,))
            deleted.append("structure")
        if "summary" in targets:
            conn.execute("DELETE FROM paper_summaries WHERE paper_id = ?", (paper_id,))
            deleted.append("summary")
        if "pages" in targets:
            conn.execute("DELETE FROM paper_pages WHERE paper_id = ?", (paper_id,))
            deleted.append("pages")
        conn.commit()
    except Exception as e:
        print(f"[删除] SQLite 数据删除失败: {e}", flush=True)
    finally:
        conn.close()

    return deleted


@tool
def delete_paper_data(paper_id: str, targets: str = "all") -> str:
    """
    选择性删除论文的索引数据。

    【调用时机】：
    - 用户说"删除某篇论文"、"清除索引"时
    - 需要只更新部分数据（如重新生成摘要）时

    Args:
        paper_id: 论文 ID（如 sat-matching 或 2301.12345）
        targets: 要删除的数据类型，逗号分隔。可选值：
                 vector（向量嵌入）、structure（结构）、summary（摘要标签）、
                 pages（OCR原文）、all（全部，默认）
                 例如："summary,pages" 只删摘要和原文

    Returns:
        删除结果说明。
    """
    target_list = [t.strip() for t in targets.split(",") if t.strip()]
    if not target_list:
        target_list = ["all"]

    invalid = [t for t in target_list if t not in _ALL_TARGETS + ["all"]]
    if invalid:
        return f"无效的 targets: {invalid}。可选值: vector, structure, summary, pages, all"

    deleted = _delete_paper_data(paper_id, target_list)
    if not deleted:
        return f"论文 {paper_id} 没有找到可删除的数据。"

    return f"已删除论文 **{paper_id}** 的以下数据：{', '.join(deleted)}。\n可调用 `reindex_paper` 重新索引，或 `scan_and_index_new_papers` 扫描。"


@tool
def reindex_paper(paper_id: str) -> str:
    """
    删除论文全部索引数据并重新执行完整索引流程（OCR → 存原文 → 结构提取 → 摘要生成 → 向量嵌入）。

    【调用时机】：
    - 用户说"重新索引"、"重建索引"时
    - 旧论文缺少摘要/原文数据，需要走新流程时

    Args:
        paper_id: 论文 ID（如 sat-matching 或 2301.12345）

    Returns:
        重索引结果。
    """
    pdf_path = PAPERS_DIR / f"{paper_id}.pdf"
    if not pdf_path.exists():
        return f"未找到论文 PDF 文件：{pdf_path}\n请确认 paper_id 正确，或先下载论文。"

    print(f"[重索引] 正在删除 {paper_id} 的旧数据...", flush=True)
    deleted = _delete_paper_data(paper_id, ["all"])
    print(f"[重索引] 已清除: {', '.join(deleted) if deleted else '无旧数据'}", flush=True)

    print(f"[重索引] 开始重新索引 {paper_id}...", flush=True)
    try:
        shared_knowledge.insert(
            name=paper_id,
            path=str(pdf_path),
            reader=pdf_reader,
            skip_if_exists=True,
        )
        print(f"[重索引] 向量嵌入完成", flush=True)

        structure_info = ""
        summary_info = ""
        if pdf_reader.last_ocr_pages:
            try:
                save_paper_pages(paper_id, pdf_reader.last_ocr_pages)
                print(f"[重索引] OCR 原文已存储 ({len(pdf_reader.last_ocr_pages)} 页)", flush=True)
            except Exception as pe:
                print(f"[重索引] 原文存储失败: {pe}", flush=True)

            structure = None
            try:
                structure = _extract_and_store_structure(paper_id, pdf_reader.last_ocr_pages)
                n_sec = len(structure.get("sections", []))
                n_thm = len(structure.get("theorems", []))
                n_def = len(structure.get("definitions", []))
                structure_info = f"\n结构提取：{n_sec} 个章节, {n_thm} 个定理/引理, {n_def} 个定义"
            except Exception as se:
                print(f"[重索引] 结构提取失败: {se}", flush=True)

            try:
                sm = _extract_and_store_summary(paper_id, pdf_reader.last_ocr_pages, structure or {})
                if sm:
                    tags = sm.get("field_tags", []) + sm.get("technique_tags", [])
                    summary_info = f"\n摘要标签：{', '.join(tags[:4])}"
            except Exception as sme:
                print(f"[重索引] 摘要提取失败: {sme}", flush=True)

        return (
            f"论文 **{paper_id}** 重新索引完成！{structure_info}{summary_info}\n\n"
            f"可用 `browse_paper_catalog` 查看索引，或 `get_paper_overview('{paper_id}')` 查看概要。"
        )
    except Exception as e:
        return f"重索引失败：{e}"


# ─────────────────────────────────────────────────────────────
# Agent 构建（4 角色架构）
# ─────────────────────────────────────────────────────────────

# 1. arXiv Researcher — 外网检索 + 下载
arxiv_researcher = Agent(
    name="arXiv Researcher",
    role="负责在 arXiv 上检索、下载论文并触发索引",
    model=shared_llm,
    tools=[search_arxiv_papers, load_paper_for_deep_analysis],
    instructions=dedent("""
        你是外网论文情报员。职责：
        1. 用户需要找新论文时，调用 `search_arxiv_papers` 检索 arXiv
        2. 用户指定要下载/加载某篇时，调用 `load_paper_for_deep_analysis` 下载并索引
        3. 返回论文标题、摘要、arXiv ID，让用户选择

        注意：
        - 下载后论文会自动完成索引（OCR + 结构提取 + 摘要生成 + 向量嵌入）
        - 若用户说"加载第N篇"，从你之前的搜索结果中找到对应 ID 再调用下载
    """),
    markdown=True,
)

# 2. Paper Librarian — 索引管理专员
paper_librarian = Agent(
    name="Paper Librarian",
    role="负责论文索引的扫描、删除和重建",
    model=shared_llm,
    tools=[
        scan_and_index_new_papers,
        delete_paper_data,
        reindex_paper,
        list_indexed_papers,
    ],
    instructions=dedent("""
        你是论文管理员，负责知识库的索引维护。职责：

        【扫描索引】：
        - 调用 `scan_and_index_new_papers` 扫描新论文并自动索引

        【删除数据】：
        - 调用 `delete_paper_data(paper_id, targets)` 选择性删除
        - targets 可选: vector, structure, summary, pages, all
        - 执行前简要说明将删除什么，执行后报告结果

        【重新索引】：
        - 调用 `reindex_paper(paper_id)` 删除旧数据并重跑完整索引
        - 适用于旧论文缺少摘要/结构数据的情况

        【列出论文】：
        - 调用 `list_indexed_papers` 查看知识库中所有论文

        操作原则：先报告当前状态，再执行操作，最后确认结果。
    """),
    markdown=True,
)

# 3. Deep Reader — 精读 + 数学推理专家
deep_reader = Agent(
    name="Deep Reader",
    role="负责论文深度精读、数学内容解析与推理",
    model=shared_llm,
    tools=[
        browse_paper_catalog,
        get_paper_overview,
        read_paper_pages,
        read_paper_section,
        get_paper_structure,
        search_structured,
    ],
    knowledge=shared_knowledge,
    search_knowledge=True,
    add_knowledge_to_context=True,
    instructions=dedent("""
        你是数学论文精读与推理专家。
        绝对红线：回答论文内容必须且只能基于工具返回的结果，绝不编造！

        【检索策略 — 三级递进】：
        1. 第一级：调用 `browse_paper_catalog()` 浏览全局索引，按领域/内容/技巧三维标签挑候选
        2. 第二级：调用 `get_paper_overview(paper_id)` 看摘要、证明思路、核心技巧
        3. 第三级：深读原文：
           - 按章节 → `read_paper_section(paper_id, section_id)`
           - 按页码 → `read_paper_pages(paper_id, start_page, end_page)`
        4. 兜底：用 `search_structured` 或知识库向量检索

        【定理/证明定位】：
        - 问定理内容 → 从 overview 定理列表定位页码，再 read_paper_pages 读原文
        - 问证明思路 → 先看 overview 的 proof_approaches，需细节再 read_paper_section
        - 也可用 `search_structured(query, element_type="theorem/proof/definition")` 辅助

        【数学推理能力 — 核心差异化能力】：
        读取原文后，不要只复制粘贴！你必须：
        1. 用自然语言解释证明的核心思路（"为什么这样做"、"关键洞察是什么"）
        2. 指出证明中的关键转折步骤和技巧
        3. 对比不同定理的证明方法异同（如果涉及多个定理）
        4. 简化复杂表达式时补充中间推导步骤
        5. LaTeX 公式保持原样，但用括号注释说明关键符号含义

        【回答层次（必须遵循）】：
        1. 先给出 1-2 句直觉性回答（让用户快速理解要点）
        2. 再给出严谨的数学细节（带 [Theorem X, p.Y] 格式引用）
        3. 如有必要，补充"直觉理解"或"类比"帮助消化

        【回答格式】：
        - 引用格式：[Theorem 3.1, p.5] 或 [第3页]
        - 公式保留 LaTeX（$ 或 $$ 包裹）
        - 检索不到则明确说"知识库中未找到"
    """),
    markdown=True,
)

# 4. Team Leader — 增强版主管
arxiv_team = Team(
    name="arXiv Team",
    model=shared_llm,
    members=[arxiv_researcher, paper_librarian, deep_reader],
    # 主管掌控记忆和对话历史
    db=agent_db,
    num_history_runs=10,
    add_history_to_context=True,
    enable_agentic_memory=True,
    tools=[browse_paper_catalog, get_paper_overview, get_paper_structure, list_indexed_papers, save_note, list_notes],
    instructions=dedent("""
        你是 arXiv 学术助理团队的主管，管理三位专家：
        - `arXiv Researcher`：外网检索和下载论文
        - `Paper Librarian`：索引管理（扫描、删除、重建）
        - `Deep Reader`：论文精读和数学推理

        ═══════════════════════════════════════
        【最高优先级：自主处理原则】
        ═══════════════════════════════════════
        能用自己的工具解决的问题，绝不委派！你有以下工具：
        - `browse_paper_catalog()` — 浏览所有论文索引（标题+三维标签）
        - `get_paper_overview(paper_id)` — 查看论文详细概要（摘要/证明思路/技巧/章节）
        - `get_paper_structure(paper_id)` — 查看论文结构（章节/定理/定义列表）
        - `list_indexed_papers()` — 列出所有已索引论文
        - `save_note(filename, content)` — 保存笔记
        - `list_notes()` — 列出笔记

        自主处理场景：
        - "标题是什么" / "这篇论文讲什么" → 调 get_paper_overview，直接回答
        - "有哪些论文" / "知识库里有什么" → 调 browse_paper_catalog 或 list_indexed_papers
        - "论文结构" / "有哪些定理" → 调 get_paper_structure，直接回答
        - "保存笔记" / "有哪些笔记" → 调 save_note / list_notes
        - 不需要读原文就能回答的概要性问题 → 用 get_paper_overview 即可

        ═══════════════════════════════════════
        【委派规则】
        ═══════════════════════════════════════
        只在以下情况委派：

        → 委派 `arXiv Researcher`：
          - 用户要在 arXiv 搜索新论文
          - 用户要下载/加载某篇论文

        → 委派 `Paper Librarian`：
          - 用户要扫描新论文、删除索引、重建索引
          - 用户说"扫描"、"删除"、"重新索引"

        → 委派 `Deep Reader`：
          - 用户需要读论文原文细节（某个证明、某段推导）
          - 用户问数学证明的思路、步骤、直觉理解
          - 用户要求解释/简化某个定理的证明
          - 用户问"为什么"类需要深度推理的问题

        委派时必须下达具体指令，包含 paper_id 和具体要求。
        例如："请读取论文 'sat-matching' 第3章原文，解释 Theorem 3.1 的证明思路，
        特别说明为什么使用了鸽巢原理。"

        ═══════════════════════════════════════
        【Focus Paper 状态管理】
        ═══════════════════════════════════════
        1. 通过对话历史记住用户当前聚焦的论文（Focus Paper）
        2. 用户表示要研究某篇时，设为 Focus Paper
        3. 后续提问未指明论文时，默认问 Focus Paper
        4. 委派时必须带上 paper_id
        5. 无 Focus Paper 且指代不明时，反问确认

        ═══════════════════════════════════════
        【跨论文综合】
        ═══════════════════════════════════════
        当用户问涉及多篇论文的问题（对比、综述）：
        1. 自己调 browse_paper_catalog 确定相关论文
        2. 逐篇调 get_paper_overview 获取各篇概要
        3. 在概要层面综合回答（对比方法、总结共性与差异）
        4. 如需原文细节佐证，再委派 Deep Reader 精读特定章节
    """),
    markdown=True,
    stream=True,
    session_id=TEAM_SESSION_ID,
    user_id="researcher",
    show_members_responses=True,
)


def interactive_cli():
    """同步命令行交互入口。"""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_polluted_session()

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
