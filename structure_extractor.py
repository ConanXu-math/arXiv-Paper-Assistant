"""Two-phase structure extraction for math papers: regex + LLM refinement.

Phase 1: Fast regex scanning to identify sections, theorems, proofs,
         definitions, and display equations from OCR text.
Phase 2: LLM call to validate, fix, and enrich the regex results
         (e.g. theorem-proof linkage, section assignment, summary).
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _fix_json_escapes(s: str) -> str:
    """Fix invalid JSON escape sequences caused by LaTeX backslashes.

    LLM output often contains raw LaTeX like \\subseteq or \\mathbb inside
    JSON strings.  These are not valid JSON escapes and cause json.loads()
    to fail.  This function doubles any backslash that is NOT already part
    of a legal JSON escape (\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX).
    """
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)


# ── Phase 1: Regex patterns ───────────────────────────────────────

_SECTION_RE = re.compile(
    r"^(?:#{1,3}\s+)?"
    r"(\d+(?:\.\d+)*)"
    r"\.?\s+"
    r"(.+)",
    re.MULTILINE,
)

_THEOREM_RE = re.compile(
    r"(?P<type>Theorem|Lemma|Proposition|Corollary|定理|引理|命题|推论)"
    r"\s+(?P<label>[\d.]+)"
    r"[.:\s]*(?P<statement>[^\n]*(?:\n(?!Proof|证明|Definition|定义|Theorem|Lemma|Proposition|Corollary)[^\n]*){0,5})",
    re.IGNORECASE,
)

_PROOF_RE = re.compile(
    r"(?P<keyword>Proof|证明)[.\s:]*",
    re.IGNORECASE,
)

_DEFINITION_RE = re.compile(
    r"(?P<type>Definition|定义)\s+(?P<label>[\d.]+)"
    r"[.:\s]*(?P<content>[^\n]*(?:\n(?!Theorem|Lemma|Proof|Definition|定义|定理|引理)[^\n]*){0,4})",
    re.IGNORECASE,
)

_DISPLAY_MATH_RE = re.compile(
    r"\$\$(.+?)\$\$|\\\[(.+?)\\\]",
    re.DOTALL,
)


def _level_from_label(label: str) -> int:
    return label.count(".") + 1


def _regex_extract(pages: list[str]) -> dict:
    """Phase 1: extract raw structure using regex."""
    sections: list[dict] = []
    theorems: list[dict] = []
    proofs: list[dict] = []
    definitions: list[dict] = []
    key_equations: list[dict] = []

    for page_idx, text in enumerate(pages):
        page_num = page_idx + 1

        for m in _SECTION_RE.finditer(text):
            label = m.group(1)
            title = m.group(2).strip()
            if len(title) > 200 or len(title) < 2:
                continue
            sections.append({
                "id": f"sec{label}",
                "title": title,
                "level": _level_from_label(label),
                "page": page_num,
            })

        for m in _THEOREM_RE.finditer(text):
            label = m.group("label")
            theorems.append({
                "id": f"thm{label}",
                "label": f"{m.group('type')} {label}",
                "type": m.group("type").lower(),
                "statement": m.group("statement").strip()[:500],
                "section_id": "",
                "page": page_num,
            })

        for m in _PROOF_RE.finditer(text):
            proofs.append({
                "id": f"prf_p{page_num}_{m.start()}",
                "proves": "",
                "page_start": page_num,
                "page_end": page_num,
            })

        for m in _DEFINITION_RE.finditer(text):
            label = m.group("label")
            definitions.append({
                "id": f"def{label}",
                "label": f"{m.group('type')} {label}",
                "content": m.group("content").strip()[:500],
                "section_id": "",
                "page": page_num,
            })

        for m in _DISPLAY_MATH_RE.finditer(text):
            latex = (m.group(1) or m.group(2) or "").strip()
            if len(latex) > 10:
                key_equations.append({
                    "id": f"eq_p{page_num}_{m.start()}",
                    "latex": latex[:300],
                    "page": page_num,
                })

    # De-duplicate sections by id
    seen_sec = set()
    unique_sections = []
    for s in sections:
        if s["id"] not in seen_sec:
            seen_sec.add(s["id"])
            unique_sections.append(s)

    return {
        "title": "",
        "summary": "",
        "sections": unique_sections,
        "theorems": theorems,
        "proofs": proofs,
        "definitions": definitions,
        "key_equations": key_equations[:20],
    }


# ── Phase 2: LLM refinement ──────────────────────────────────────

_LLM_PROMPT = """\
你是一个数学论文结构分析器。下面是一篇论文的 OCR 文本（已用正则做了初步结构提取）。

请你：
1. 修正和补充正则结果中遗漏或错误的条目
2. 为每个 theorem/definition 标注它所属的 section_id
3. 为每个 proof 标注它证明了哪个 theorem（填 proves 字段）
4. 填写 title（论文标题）和 summary（1-2 句话概括论文核心内容）
5. 若正则结果中某些条目明显是误识别（如把正文段落当成 section），请删除

严格输出 JSON，不要加 ```json 或其他标记，格式如下：
{
  "title": "...",
  "summary": "...",
  "sections": [{"id": "sec1", "title": "...", "level": 1, "page": 1}, ...],
  "theorems": [{"id": "thm3.1", "label": "Theorem 3.1", "type": "theorem", "statement": "...", "section_id": "sec3", "page": 5}, ...],
  "proofs": [{"id": "prf1", "proves": "thm3.1", "page_start": 5, "page_end": 6}, ...],
  "definitions": [{"id": "def2.1", "label": "Definition 2.1", "content": "...", "section_id": "sec2", "page": 3}, ...],
  "key_equations": [{"id": "eq1", "latex": "...", "page": 5}, ...]
}

=== 正则提取结果 ===
{regex_json}

=== 论文全文（前 6000 字） ===
{paper_text}
"""


def _call_llm_for_refinement(
    regex_result: dict,
    pages: list[str],
    llm: Any,
) -> Optional[dict]:
    """Use LLM to refine the regex-extracted structure."""
    full_text = "\n\n".join(
        f"[第{i+1}页]\n{t}" for i, t in enumerate(pages) if t.strip()
    )
    truncated = full_text[:6000]

    prompt = _LLM_PROMPT.format(
        regex_json=json.dumps(regex_result, ensure_ascii=False, indent=2),
        paper_text=truncated,
    )

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=llm.api_key,
            base_url=str(llm.base_url),
        )
        response = client.chat.completions.create(
            model=llm.id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            logger.warning("LLM returned empty response")
            return None
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        # Try to extract JSON object even if surrounded by extra text
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            raw = raw[brace_start : brace_end + 1]
        return json.loads(_fix_json_escapes(raw))
    except Exception as e:
        logger.warning("LLM refinement failed, using regex results: %s", e)
        return None


# ── Public API ────────────────────────────────────────────────────

def extract_paper_structure(
    pages: list[str],
    llm: Any = None,
) -> dict:
    """Extract structured outline from OCR page texts.

    Args:
        pages: list of per-page OCR text strings.
        llm: an agno OpenAILike model instance (optional; skips Phase 2 if None).

    Returns:
        Standardised structure dict.
    """
    regex_result = _regex_extract(pages)

    if llm is not None:
        refined = _call_llm_for_refinement(regex_result, pages, llm)
        if refined is not None:
            for key in ("title", "summary", "sections", "theorems",
                        "proofs", "definitions", "key_equations"):
                if key in refined:
                    regex_result[key] = refined[key]

    return regex_result


def find_section_for_page(structure: dict, page_num: int) -> str:
    """Return the section title that covers the given page."""
    sections = structure.get("sections", [])
    current = ""
    for sec in sorted(sections, key=lambda s: s.get("page", 0)):
        if sec.get("page", 0) <= page_num:
            current = sec.get("title", "")
        else:
            break
    return current


def find_elements_on_page(structure: dict, page_num: int) -> list[str]:
    """Return a list of element types present on the given page."""
    types = set()
    for thm in structure.get("theorems", []):
        if thm.get("page") == page_num:
            types.add("theorem")
    for prf in structure.get("proofs", []):
        if prf.get("page_start", 0) <= page_num <= prf.get("page_end", 0):
            types.add("proof")
    for defn in structure.get("definitions", []):
        if defn.get("page") == page_num:
            types.add("definition")
    for eq in structure.get("key_equations", []):
        if eq.get("page") == page_num:
            types.add("equation")
    return sorted(types)


# ── Paper summary extraction (LLM) ────────────────────────────────

_SUMMARY_PROMPT = """\
你是一个数学论文分析专家。请阅读以下论文的结构骨架和 OCR 原文，生成一份高层概要。

要求：
1. title: 论文标题
2. abstract: 1-2 段概述论文的主要研究内容和贡献（中文）
3. proof_approaches: 每个主要定理的证明思路（1-2句），格式为 {{"Theorem 3.1": "通过构造...", ...}}
4. core_techniques: 论文用到的核心方法/技巧列表（如 ["鸽巢原理", "概率方法"]）
5. field_tags: 论文所属数学领域，选 2-3 个（如 ["图论", "组合优化"]）
6. content_tags: 论文研究内容的关键词，选 2-4 个（如 ["匹配存在性条件", "Hall定理推广"]）
7. technique_tags: 论文用到的方法/技巧关键词，选 2-4 个（如 ["构造性证明", "鸽巢原理"]）

注意：
- field_tags 描述论文属于哪个数学分支/领域
- content_tags 描述论文"做了什么"（研究的具体问题）
- technique_tags 描述论文"怎么做的"（用了什么方法/工具）
- 三类标签不要重复，各自侧重不同维度

严格输出 JSON，不要加 ```json 或其他标记：
{{
  "title": "...",
  "abstract": "...",
  "proof_approaches": {{"Theorem X": "...", ...}},
  "core_techniques": ["...", ...],
  "field_tags": ["...", ...],
  "content_tags": ["...", ...],
  "technique_tags": ["...", ...]
}}

=== 论文结构骨架 ===
{structure_skeleton}

=== 论文原文（前 8000 字） ===
{paper_text}
"""


def extract_paper_summary(
    pages: list[str],
    structure: dict,
    llm: Any,
) -> Optional[dict]:
    """Use LLM to generate a high-level summary with multi-dimensional tags.

    Args:
        pages: list of per-page OCR text strings.
        structure: structure dict from extract_paper_structure().
        llm: an agno OpenAILike model instance.

    Returns:
        Summary dict with title, abstract, proof_approaches, core_techniques,
        field_tags, content_tags, technique_tags. None on failure.
    """
    skeleton_parts = []
    if structure.get("title"):
        skeleton_parts.append(f"标题: {structure['title']}")
    if structure.get("summary"):
        skeleton_parts.append(f"概要: {structure['summary']}")
    for sec in structure.get("sections", []):
        indent = "  " * (sec.get("level", 1) - 1)
        skeleton_parts.append(f"{indent}[{sec['id']}] {sec.get('title', '')} (p.{sec.get('page', '?')})")
    for thm in structure.get("theorems", []):
        skeleton_parts.append(f"  {thm.get('label', thm['id'])}: {thm.get('statement', '')[:100]}")
    for defn in structure.get("definitions", []):
        skeleton_parts.append(f"  {defn.get('label', defn['id'])}: {defn.get('content', '')[:80]}")
    skeleton_text = "\n".join(skeleton_parts) if skeleton_parts else "(无结构信息)"

    full_text = "\n\n".join(
        f"[第{i+1}页]\n{t}" for i, t in enumerate(pages) if t.strip()
    )
    truncated = full_text[:8000]

    prompt = _SUMMARY_PROMPT.format(
        structure_skeleton=skeleton_text,
        paper_text=truncated,
    )

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=llm.api_key,
            base_url=str(llm.base_url),
        )
        response = client.chat.completions.create(
            model=llm.id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            logger.warning("LLM summary returned empty response")
            return None
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            raw = raw[brace_start : brace_end + 1]
        result = json.loads(_fix_json_escapes(raw))
        expected_keys = {"title", "abstract", "proof_approaches", "core_techniques",
                         "field_tags", "content_tags", "technique_tags"}
        for key in expected_keys:
            if key not in result:
                result[key] = {} if key == "proof_approaches" else ([] if key != "title" and key != "abstract" else "")
        return result
    except Exception as e:
        logger.warning("LLM summary extraction failed: %s", e)
        return None


def format_structure_for_display(structure: dict) -> str:
    """Format structure dict into readable markdown text for the agent."""
    lines: list[str] = []

    title = structure.get("title", "")
    if title:
        lines.append(f"# {title}\n")

    summary = structure.get("summary", "")
    if summary:
        lines.append(f"**摘要**: {summary}\n")

    sections = structure.get("sections", [])
    if sections:
        lines.append("## 章节结构\n")
        for sec in sections:
            indent = "  " * (sec.get("level", 1) - 1)
            lines.append(f"{indent}- **{sec['id']}** {sec.get('title', '')} (p.{sec.get('page', '?')})")

    theorems = structure.get("theorems", [])
    if theorems:
        lines.append("\n## 定理/引理\n")
        for thm in theorems:
            stmt = thm.get("statement", "")[:120]
            sec = f" [{thm.get('section_id', '')}]" if thm.get("section_id") else ""
            lines.append(f"- **{thm.get('label', thm['id'])}** (p.{thm.get('page', '?')}){sec}: {stmt}")

    definitions = structure.get("definitions", [])
    if definitions:
        lines.append("\n## 定义\n")
        for d in definitions:
            content = d.get("content", "")[:120]
            lines.append(f"- **{d.get('label', d['id'])}** (p.{d.get('page', '?')}): {content}")

    proofs = structure.get("proofs", [])
    if proofs:
        lines.append("\n## 证明\n")
        for p in proofs:
            proves = f" → {p['proves']}" if p.get("proves") else ""
            lines.append(f"- {p['id']} (p.{p.get('page_start', '?')}-{p.get('page_end', '?')}){proves}")

    key_eqs = structure.get("key_equations", [])
    if key_eqs:
        lines.append("\n## 关键公式\n")
        for eq in key_eqs[:10]:
            lines.append(f"- (p.{eq.get('page', '?')}) ${eq.get('latex', '')[:80]}$")

    return "\n".join(lines) if lines else "未提取到结构化信息。"
