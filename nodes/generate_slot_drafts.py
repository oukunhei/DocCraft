from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from nodes.slot_skills import SLOT_TYPE_REFERENCE, get_generation_skill, get_skill_snapshot, infer_slot_type
    from state import ReportState
except Exception:  # noqa: BLE001
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from nodes.slot_skills import SLOT_TYPE_REFERENCE, get_generation_skill, get_skill_snapshot, infer_slot_type
    from state import ReportState


GENERATION_PROMPT_BASE = """你是 Agent A，负责为项目技术报告生成高质量草稿。

## 核心任务
请仅针对当前输入的单个槽位(slot)进行深入分析并生成文本。
写作要求：
- 创新性：思考该槽位可体现的独创性
- 技术深度：说明技术路径与选型理由
- 应用价值：明确实际问题与可推广价值
- 引用规范：所有结论应有证据来源
- 语言风格：客观正式、术语统一、避免口语化
- 段落长度：150-400字
- 事实约束：非参考文献槽位中，禁止引入未出现在 project_evidence 的具体数据、数据集名、版本号、平台参数。
- 证据不足时：如果 project_evidence 为空或不足，必须明确输出“[待补证据]”而不是编造细节。

## 引用输出约束（必须遵守）
- 当前槽位类型：{slot_type}
- 若不是参考文献槽位：只能使用文内引用格式（如“(Author, Year)”或“[1]”），禁止在本段末尾输出参考文献条目列表。
- 若是参考文献槽位：输出文末参考文献列表（APA格式），可结合联网检索补全缺失信息。

## 当前槽位专项规则
{slot_skill}

请严格遵守输出格式：
<output>
{{
  "slots": [
    {{
      "slot_id": "...",
      "draft_text": "...",
      "source_refs": ["project_1"],
      "confidence": 0.0,
      "risk_notes": ["..."]
    }}
  ]
}}
</output>
"""

_EVIDENCE_MIN_ITEMS = 1


def _llm(state: ReportState, temperature: float = 0.2) -> ChatOpenAI:
    api_key = state.get("reasoning_api_key")
    base_url = state.get("reasoning_url") or state.get("reasoning_base_url")
    if not api_key or not base_url:
        raise ValueError("missing reasoning_api_key / reasoning_base_url in state")
    model = state.get("reasoning_model_name") or state.get("llm_model_name") or "deepseek-chat"
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=temperature)


def _decode_first_json_object(text: str) -> Dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _extract_output_block(raw: str) -> str:
    m = re.search(r"<output>\s*([\s\S]*?)\s*</output>", raw, re.IGNORECASE)
    return m.group(1).strip() if m else raw


def _safe_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    extracted = _extract_output_block(raw)
    try:
        obj = json.loads(extracted)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", extracted, re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    scanned = _decode_first_json_object(extracted)
    if scanned is not None:
        return scanned

    scanned_raw = _decode_first_json_object(raw)
    if scanned_raw is not None:
        return scanned_raw

    raise ValueError(f"cannot parse JSON from: {raw[:300]}")


def _invoke(llm: ChatOpenAI, sys_prompt: str, payload: Any) -> Dict[str, Any]:
    resp = llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return _safe_json(content)


def _slots(template_slots: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        s for s in (template_slots.get("core_slots") or []) + (template_slots.get("other_technical_slots") or [])
        if s.get("slot_id")
    ]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return default


def _slist(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _fallback_seed_terms(raw_docs: List[Document]) -> Dict[str, List[str]]:
    """当 seed_terms 缺失时，从论文前若干块提取英文关键词，避免证据召回为0。"""
    text = " ".join((d.page_content or "")[:500] for d in raw_docs[:8])
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", text.lower())
    stop = {
        "this", "that", "with", "from", "were", "been", "their", "they", "into", "about", "through",
        "which", "using", "based", "than", "have", "has", "also", "into", "over", "between", "such",
    }
    keywords = [w for w in words if w not in stop]
    keywords = list(dict.fromkeys(keywords))[:20]
    return {"keywords": keywords, "domains": keywords[:6]}


def _build_evidence(
    slot: Dict[str, Any],
    raw_docs: List[Document],
    seed_terms: Dict[str, Any],
) -> Dict[str, Any]:
    terms = re.findall(
        r"[\u4e00-\u9fff]{2,}|[a-zA-Z][a-zA-Z0-9\-]{2,}",
        f"{slot.get('title', '')} {slot.get('description', '')} "
        f"{' '.join(map(str, (seed_terms.get('keywords') or [])[:6]))} "
        f"{' '.join(map(str, (seed_terms.get('domains') or [])[:3]))}".lower(),
    )
    # slot_id 往往更稳定，补充进去可减少分类文本与正文语言不一致造成的漏召回。
    terms.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", str(slot.get("slot_id") or "").replace("_", " ").lower()))
    terms = list(dict.fromkeys(terms))[:14]

    def score_docs(docs: List[Document], max_n: int, max_len: int, prefix: str) -> List[Dict[str, Any]]:
        scored = []
        for doc in docs:
            cnt = sum(1 for t in terms if t and t in (doc.page_content or "").lower())
            if cnt > 0:
                scored.append((cnt, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored and docs:
            # 最小止血：若关键词完全未命中，回退到前若干块，确保每槽位至少可追溯到项目正文。
            return [
                {
                    "id": f"{prefix}_{i}",
                    "content": doc.page_content[:max_len],
                    "metadata": {**dict(doc.metadata or {}), "fallback": "front_chunks"},
                }
                for i, doc in enumerate(docs[:max_n], 1)
            ]
        return [
            {
                "id": f"{prefix}_{i}",
                "content": doc.page_content[:max_len],
                "metadata": dict(doc.metadata or {}),
            }
            for i, (_, doc) in enumerate(scored[:max_n], 1)
        ]

    return {
        "slot_id": slot.get("slot_id", ""),
        "slot_title": slot.get("title", ""),
        "slot_description": str(slot.get("description", "")),
        "terms": terms,
        "project_evidence": score_docs(raw_docs, 4, 700, "proj"),
        "external_evidence": [],
    }


def _extract_dois(text: str) -> List[str]:
    pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
    dois = [m.group(0).strip(" .,;:()[]{}") for m in pattern.finditer(text or "")]
    return list(dict.fromkeys(dois))[:6]


def _infer_paper_title(raw_docs: List[Document]) -> str:
    if not raw_docs:
        return ""
    sample = "\n".join((raw_docs[i].page_content or "")[:240] for i in range(min(2, len(raw_docs))))
    lines = [ln.strip() for ln in sample.splitlines() if ln.strip()]
    for line in lines:
        if 6 <= len(line) <= 150 and not re.search(r"摘要|abstract|关键词|keyword|目录", line, re.IGNORECASE):
            return line
    return ""


def _url_json(url: str, timeout: float = 8.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "DocCraft/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def _crossref_item_to_evidence(item: Dict[str, Any], prefix: str, idx: int) -> Dict[str, Any]:
    title = ""
    if isinstance(item.get("title"), list) and item.get("title"):
        title = str(item["title"][0])
    doi = str(item.get("DOI") or "")
    year = ""
    for yk in ("issued", "published-print", "published-online"):
        yv = item.get(yk) or {}
        parts = (yv.get("date-parts") or [[]])[0]
        if parts:
            year = str(parts[0])
            break

    authors: List[str] = []
    for a in item.get("author") or []:
        family = str(a.get("family") or "").strip()
        given = str(a.get("given") or "").strip()
        if family or given:
            authors.append(f"{family} {given}".strip())
    author_text = ", ".join(authors[:5]) if authors else "Unknown"

    container = ""
    if isinstance(item.get("container-title"), list) and item.get("container-title"):
        container = str(item["container-title"][0])

    content = f"{author_text} ({year or 'n.d.'}). {title}. {container}. DOI: {doi}".strip()
    return {
        "id": f"{prefix}_{idx}",
        "content": content,
        "metadata": {
            "source": "crossref",
            "title": title,
            "doi": doi,
            "year": year,
            "container": container,
            "url": str(item.get("URL") or ""),
        },
    }


def _search_reference_web_evidence(slot_evidence: Dict[str, Any], raw_docs: List[Document]) -> List[Dict[str, Any]]:
    corpus = " ".join(
        [str(slot_evidence.get("slot_title") or ""), str(slot_evidence.get("slot_description") or "")]
        + [str(x.get("content") or "") for x in slot_evidence.get("project_evidence") or []]
    )
    dois = _extract_dois(corpus)
    paper_title = _infer_paper_title(raw_docs)
    results: List[Dict[str, Any]] = []

    for i, doi in enumerate(dois, 1):
        try:
            qdoi = urllib.parse.quote(doi, safe="")
            data = _url_json(f"https://api.crossref.org/works/{qdoi}")
            item = data.get("message") or {}
            if item:
                results.append(_crossref_item_to_evidence(item, "web_doi", i))
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            continue

    if paper_title:
        try:
            qtitle = urllib.parse.quote(paper_title)
            data = _url_json(f"https://api.crossref.org/works?query.title={qtitle}&rows=3")
            items = ((data.get("message") or {}).get("items") or [])[:3]
            for i, item in enumerate(items, 1):
                results.append(_crossref_item_to_evidence(item, "web_title", i))
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            pass

    dedup: Dict[str, Dict[str, Any]] = {}
    for item in results:
        key = str((item.get("metadata") or {}).get("doi") or item.get("content") or "")
        if key and key not in dedup:
            dedup[key] = item
    return list(dedup.values())[:6]


def _fallback_draft(slot: Dict[str, Any], error: str) -> Dict[str, Any]:
    sid = str(slot.get("slot_id") or "")
    title = str(slot.get("title") or sid or "该部分").strip()
    return {
        "slot_id": sid,
        "draft_text": (
            f"围绕{title}，当前证据可支持的结论主要涉及问题定义、方法流程与阶段性结果。"
            "由于本槽位生成阶段发生异常，尚未形成高置信度文本，"
            "请在后续复核中重点核实关键论断并补充量化证据。"
            "[需补充：本槽位自动生成失败后的人工作业内容]"
        ),
        "source_refs": [],
        "confidence": 0.2,
        "risk_notes": [f"slot_generate_failed: {error}"],
    }


def _insufficient_evidence_draft(slot: Dict[str, Any]) -> Dict[str, Any]:
    sid = str(slot.get("slot_id") or "")
    title = str(slot.get("title") or sid or "该部分").strip()
    return {
        "slot_id": sid,
        "draft_text": (
            f"[{title}] 当前自动证据检索不足，无法生成可验证的完整技术段落。"
            "请先补充本槽位对应的项目原文证据、实验记录或实现说明后再生成。"
            "[待补证据]"
        ),
        "source_refs": ["project_1"],
        "confidence": 0.1,
        "risk_notes": ["insufficient_project_evidence"],
    }


def _normalize_slot_output(slot: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
    sid = str(slot.get("slot_id") or "")
    first = {}
    if isinstance(out.get("slots"), list) and out["slots"]:
        first = out["slots"][0] or {}

    resolved_id = str(first.get("slot_id") or sid).strip() or sid
    return {
        "slot_id": resolved_id,
        "draft_text": str(first.get("draft_text") or "").strip(),
        "source_refs": _slist(first.get("source_refs")),
        "confidence": _f(first.get("confidence"), 0.5),
        "risk_notes": _slist(first.get("risk_notes")),
    }


def generate_slot_drafts(state: ReportState) -> Dict[str, Any]:
    """节点4A：逐槽位调用Agent A生成草稿并显式落盘。"""

    raw_documents: List[Document] = state.get("raw_documents") or []
    template_slots: Dict[str, Any] = state.get("template_slots") or {}
    seed_terms: Dict[str, Any] = state.get("seed_terms") or {}
    if not (seed_terms.get("keywords") or seed_terms.get("domains")):
        seed_terms = _fallback_seed_terms(raw_documents)

    if not raw_documents:
        return {"errors": {"generate_slot_drafts": "missing raw_documents"}}
    if not template_slots:
        return {"errors": {"generate_slot_drafts": "missing template_slots"}}

    slots = _slots(template_slots)
    if not slots:
        return {"errors": {"generate_slot_drafts": "template_slots has no valid slots"}}

    run_id = state.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(state.get("intermediate_dir") or "artifacts/intermediate") / "node4a" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[node4a] start per-slot generation, slots={len(slots)}", flush=True)

    slot_evidence: Dict[str, Any] = {}
    reference_web_search_used: Dict[str, bool] = {}
    for slot in slots:
        sid = str(slot.get("slot_id") or "").strip()
        if not sid:
            continue
        evidence = _build_evidence(slot, raw_documents, seed_terms)
        slot_type = infer_slot_type(slot)
        if slot_type == SLOT_TYPE_REFERENCE:
            web_items = _search_reference_web_evidence(evidence, raw_documents)
            evidence["external_evidence"] = web_items
            reference_web_search_used[sid] = bool(web_items)
        else:
            evidence["external_evidence"] = []
            reference_web_search_used[sid] = False
        slot_evidence[sid] = evidence
    _write_json(out_dir / "slot_evidence_map.json", slot_evidence)

    llm = _llm(state, temperature=0.2)
    generated_slots: List[Dict[str, Any]] = []
    slot_errors: List[Dict[str, str]] = []
    skill_map: Dict[str, Any] = {}

    for idx, slot in enumerate(slots, 1):
        sid = str(slot.get("slot_id") or "").strip()
        if not sid:
            continue

        slot_type = infer_slot_type(slot)
        skill_snapshot = get_skill_snapshot(slot)
        skill_map[sid] = skill_snapshot
        sys_prompt = GENERATION_PROMPT_BASE.format(
            slot_type=slot_type,
            slot_skill=get_generation_skill(slot),
        )
        payload = {
            "slot": slot,
            "slot_evidence": slot_evidence.get(sid, {}),
            "constraints": {
                "prefer_project_paper": True,
                "external_is_supporting": slot_type == SLOT_TYPE_REFERENCE,
                "language": "zh",
                "style": "project_report",
                "depth": "deep",
                "allow_reference_web_search": slot_type == SLOT_TYPE_REFERENCE,
                "non_reference_strict_project_only": slot_type != SLOT_TYPE_REFERENCE,
                "no_unseen_facts_without_project_evidence": slot_type != SLOT_TYPE_REFERENCE,
            },
        }

        project_evidence = (slot_evidence.get(sid) or {}).get("project_evidence") or []
        if slot_type != SLOT_TYPE_REFERENCE and len(project_evidence) < _EVIDENCE_MIN_ITEMS:
            generated_slots.append(_insufficient_evidence_draft(slot))
            print(f"[node4a] generated slot {idx}/{len(slots)}: {sid} (insufficient evidence)", flush=True)
            continue

        try:
            out = _invoke(llm, sys_prompt, payload)
            normalized = _normalize_slot_output(slot, out)
            if not normalized.get("draft_text"):
                normalized["risk_notes"] = normalized.get("risk_notes", []) + ["empty_draft_text"]
                normalized["draft_text"] = _fallback_draft(slot, "empty_draft_text")["draft_text"]
                normalized["confidence"] = min(_f(normalized.get("confidence"), 0.5), 0.4)
            generated_slots.append(normalized)
        except Exception as exc:  # noqa: BLE001
            slot_errors.append({"slot_id": sid, "error": str(exc)})
            generated_slots.append(_fallback_draft(slot, str(exc)))

        print(f"[node4a] generated slot {idx}/{len(slots)}: {sid}", flush=True)

    a_out = {"slots": generated_slots}
    metadata = {
        "run_id": run_id,
        "total_slots": len(slots),
        "success_slots": len(slots) - len(slot_errors),
        "failed_slots": len(slot_errors),
        "slot_errors": slot_errors,
        "slot_skill_map": skill_map,
        "reference_web_search_used": reference_web_search_used,
    }

    _write_json(out_dir / "agent_a_output.json", a_out)
    _write_json(out_dir / "metadata.json", metadata)

    return {
        "run_id": run_id,
        "slot_evidence_map": slot_evidence,
        "agent_a_output": a_out,
        "crosscheck_disputes": [],
        "node4a_completed": True,
        "node5_completed": False,
        "node4_current_round": 1,
        "node4_max_rounds": 1,
        "node4_discussion_rounds": [{"round": 1, "disputes_count": 0}],
        "node4_feedback_mode": "none",
        "node4_feedback_pending": False,
        "node4_feedback_request_file": "",
        "node4_human_feedback": {},
    }
