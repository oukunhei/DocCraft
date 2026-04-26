"""Slot Subgraph：单个槽位的生成-评审-迭代闭环子图。

内部流程：
  gather_evidence → generate_draft → self_review
  self_review ──[confidence < 0.7 && iteration < max]──→ refine_draft ──→ self_review
  self_review ──[else]──→ finalize_slot

通过 langgraph.types.Send 由主编排图并行派发。
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from nodes.slot_skills import SLOT_TYPE_REFERENCE, get_generation_skill, get_skill_snapshot, infer_slot_type
    from nodes.semantic_retrieval import hybrid_evidence
    from nodes.schemas import SlotDraft, SlotRefine, SlotReview
except Exception:  # noqa: BLE001
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from nodes.slot_skills import SLOT_TYPE_REFERENCE, get_generation_skill, get_skill_snapshot, infer_slot_type
    from nodes.semantic_retrieval import hybrid_evidence
    from nodes.schemas import SlotDraft, SlotRefine, SlotReview


# ──────────────────────── SlotState ────────────────────────

class SlotState(TypedDict, total=False):
    # 输入（由 Send 传入）
    slot_id: str
    slot_definition: Dict[str, Any]
    raw_documents: List[Document]
    seed_terms: Dict[str, Any]
    previous_work_docs: List[Document]
    reasoning_model_name: str
    reasoning_api_key: str
    reasoning_base_url: str
    llm_model_name: str
    tool_model_name: str
    tool_api_key: str
    tool_base_url: str
    run_id: str
    iteration: int
    max_iterations: int

    # 中间状态
    evidence: Dict[str, Any]
    draft_text: str
    source_refs: List[str]
    confidence: float
    risk_notes: List[str]
    review_text: str
    review_issues: List[str]

    # 输出
    slot_output: Dict[str, Any]


# ──────────────────────── Prompts ────────────────────────

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

REVIEW_PROMPT_BASE = """你是 Agent B（Evidence-Skeptic Reviewer），负责复核项目技术报告中的单槽位草稿。

## 核心要求
- 你只做复查，不负责定稿改写。
- 请重点识别：幻觉数据、证据不足、术语不一致、论证跳跃、引用缺失。
- 若发现问题，必须在 disagreements 中写出可执行的问题描述。
- 若当前不是参考文献槽位但出现了文末参考文献列表，必须在 disagreements 明确写出“非参考文献槽位出现参考文献列表”并要求改为文内引用。
- 若当前是参考文献槽位，重点检查APA格式完整性与文内引用-文末条目的一致性。
- 若草稿出现具体数值、专有名词或实验结论，但无法由 source_refs 对应到项目证据，必须标记为“不可追溯事实”。

## 当前槽位专项复查规则
{slot_skill}

## 输出要求
- review_text: 2-4句具体问题说明
- revised_text: 可以给建议性修订文本，但下游不会直接采用
- disagreements: 列出明确分歧/风险点

请严格遵守输出格式：
<output>
{{
  "slots": [
    {{
      "slot_id": "...",
      "review_text": "...",
      "revised_text": "...",
      "source_refs": ["project_1"],
      "confidence": 0.0,
      "disagreements": ["..."]
    }}
  ]
}}
</output>
"""

REFINE_PROMPT_BASE = """你是 Agent A-Refine，负责根据复查意见改写草稿。

## 任务
基于原草稿和复查意见，输出改进后的完整段落（150-400字）。

## 改写原则
1. 删除或替换所有被标记为幻觉的数据/事实
2. 补充被指出缺失的证据引用
3. 修正术语不一致
4. 若复查指出证据不足，保留现有内容但标注「[需补充: xxx]」
5. 保持原段落的核心论点和结构

## 输入
- slot: 槽位定义
- original_draft: 原草稿
- review_issues: 复查指出的问题列表
- evidence: 可用证据

请输出严格 JSON：
<output>
{{
  "slot_id": "...",
  "draft_text": "...",
  "source_refs": ["..."],
  "confidence": 0.0,
  "risk_notes": ["..."]
}}
</output>
"""


# ──────────────────────── Helpers ────────────────────────

def _llm_from_state(state: SlotState, temperature: float = 0.2) -> ChatOpenAI:
    api_key = state.get("reasoning_api_key")
    base_url = state.get("reasoning_base_url") or state.get("reasoning_url")
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


def _invoke_structured(llm: ChatOpenAI, sys_prompt: str, payload: Any, schema_cls: type):
    """使用 LangChain structured output 调用 LLM，失败时回退到 JSON 解析。"""
    try:
        structured_llm = llm.with_structured_output(schema_cls)
        return structured_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
    except Exception as exc:  # noqa: BLE001
        print(f"[slot_subgraph] structured output failed, fallback to JSON parse: {exc}", flush=True)
        raw = _invoke(llm, sys_prompt, payload)
        # 将 dict 转换为 schema_cls
        try:
            return schema_cls(**raw)
        except Exception:
            # 尝试从嵌套的 slots 列表中提取
            slots = raw.get("slots") or []
            if slots and isinstance(slots, list):
                return schema_cls(**slots[0])
            raise


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
    text = " ".join((d.page_content or "")[:500] for d in raw_docs[:8])
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", text.lower())
    stop = {
        "this", "that", "with", "from", "were", "been", "their", "they", "into", "about", "through",
        "which", "using", "based", "than", "have", "has", "also", "over", "between", "such",
    }
    keywords = [w for w in words if w not in stop]
    keywords = list(dict.fromkeys(keywords))[:20]
    return {"keywords": keywords, "domains": keywords[:6]}


# ──────────────────────── Evidence Builder ────────────────────────

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


# ──────────────────────── Normalize Outputs ────────────────────────

def _normalize_slot_output(slot_id: str, out: Dict[str, Any]) -> Dict[str, Any]:
    first = {}
    if isinstance(out.get("slots"), list) and out["slots"]:
        first = out["slots"][0] or {}
    return {
        "slot_id": str(first.get("slot_id") or slot_id).strip() or slot_id,
        "draft_text": str(first.get("draft_text") or "").strip(),
        "source_refs": _slist(first.get("source_refs")),
        "confidence": _f(first.get("confidence"), 0.5),
        "risk_notes": _slist(first.get("risk_notes")),
    }


def _normalize_review_output(slot_id: str, out: Dict[str, Any]) -> Dict[str, Any]:
    first = {}
    if isinstance(out.get("slots"), list) and out["slots"]:
        first = out["slots"][0] or {}
    return {
        "slot_id": str(first.get("slot_id") or slot_id).strip() or slot_id,
        "review_text": str(first.get("review_text") or "").strip(),
        "revised_text": str(first.get("revised_text") or "").strip(),
        "source_refs": _slist(first.get("source_refs")),
        "confidence": _f(first.get("confidence"), 0.5),
        "disagreements": _slist(first.get("disagreements")),
    }


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


# ──────────────────────── Subgraph Nodes ────────────────────────

def gather_evidence(state: SlotState) -> SlotState:
    """为当前 slot 构建证据包（混合检索：语义 + 关键词）。"""
    raw_docs = state.get("raw_documents") or []
    seed_terms = state.get("seed_terms") or {}
    if not (seed_terms.get("keywords") or seed_terms.get("domains")):
        seed_terms = _fallback_seed_terms(raw_docs)

    slot = state.get("slot_definition") or {}
    run_id = state.get("run_id") or ""

    # 优先尝试语义混合检索
    if run_id:
        try:
            project_evidence = hybrid_evidence(
                run_id=run_id,
                slot=slot,
                raw_docs=raw_docs,
                seed_terms=seed_terms,
                semantic_k=6,
                keyword_k=4,
                max_len=700,
            )
            evidence = {
                "slot_id": slot.get("slot_id", ""),
                "slot_title": slot.get("title", ""),
                "slot_description": str(slot.get("description", "")),
                "terms": seed_terms.get("keywords", []),
                "project_evidence": project_evidence,
                "external_evidence": [],
                "retrieval_method": "hybrid",
            }
            return {"evidence": evidence}
        except Exception as exc:  # noqa: BLE001
            print(f"[slot_subgraph] hybrid_evidence failed for {slot.get('slot_id')}, fallback to keyword: {exc}", flush=True)

    # Fallback：原有纯关键词匹配
    evidence = _build_evidence(slot, raw_docs, seed_terms)
    evidence["retrieval_method"] = "keyword"
    return {"evidence": evidence}


def generate_draft(state: SlotState) -> SlotState:
    """基于证据生成草稿（首次生成）。"""
    slot = state.get("slot_definition") or {}
    evidence = state.get("evidence") or {}
    sid = state.get("slot_id") or ""

    if not sid:
        return {"draft_text": "", "source_refs": [], "confidence": 0.0, "risk_notes": ["missing slot_id"]}

    slot_type = infer_slot_type(slot)
    sys_prompt = GENERATION_PROMPT_BASE.format(
        slot_type=slot_type,
        slot_skill=get_generation_skill(slot),
    )
    payload = {
        "slot": slot,
        "slot_evidence": evidence,
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

    project_evidence = evidence.get("project_evidence") or []
    if slot_type != SLOT_TYPE_REFERENCE and len(project_evidence) < 1:
        result = _insufficient_evidence_draft(slot)
        return {
            "draft_text": result["draft_text"],
            "source_refs": result["source_refs"],
            "confidence": result["confidence"],
            "risk_notes": result["risk_notes"],
        }

    try:
        llm = _llm_from_state(state, temperature=0.2)
        draft: SlotDraft = _invoke_structured(llm, sys_prompt, payload, SlotDraft)
        if not draft.draft_text:
            draft.risk_notes.append("empty_draft_text")
            draft.draft_text = _fallback_draft(slot, "empty_draft_text")["draft_text"]
            draft.confidence = min(draft.confidence, 0.4)
        return {
            "draft_text": draft.draft_text,
            "source_refs": draft.source_refs,
            "confidence": draft.confidence,
            "risk_notes": draft.risk_notes,
        }
    except Exception as exc:  # noqa: BLE001
        result = _fallback_draft(slot, str(exc))
        return {
            "draft_text": result["draft_text"],
            "source_refs": result["source_refs"],
            "confidence": result["confidence"],
            "risk_notes": result["risk_notes"],
        }


def self_review(state: SlotState) -> SlotState:
    """对当前草稿进行自我评审。"""
    slot = state.get("slot_definition") or {}
    evidence = state.get("evidence") or {}
    sid = state.get("slot_id") or ""
    draft_text = state.get("draft_text") or ""
    source_refs = state.get("source_refs") or []

    if not draft_text:
        return {
            "review_text": "草稿为空，无需复查。",
            "review_issues": ["empty_draft"],
        }

    sys_prompt = REVIEW_PROMPT_BASE.format(slot_skill=get_generation_skill(slot))
    payload = {
        "slot": slot,
        "slot_evidence": evidence,
        "agent_a_slot": {
            "slot_id": sid,
            "draft_text": draft_text,
            "source_refs": source_refs,
            "confidence": state.get("confidence", 0.5),
            "risk_notes": state.get("risk_notes", []),
        },
        "constraints": {
            "language": "zh",
            "style": "project_report",
            "strict_evidence": True,
        },
    }

    try:
        llm = _llm_from_state(state, temperature=0.15)
        review: SlotReview = _invoke_structured(llm, sys_prompt, payload, SlotReview)
        issues = _slist(review.disagreements)
        if not issues and review.review_text:
            issues = [review.review_text]
        return {
            "review_text": review.review_text,
            "review_issues": issues,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "review_text": f"复查调用失败: {exc}",
            "review_issues": [f"review_failed: {exc}"],
        }


def should_refine(state: SlotState) -> str:
    """条件路由：判断是否需要改写。"""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    confidence = _f(state.get("confidence"), 0.0)
    issues = _slist(state.get("review_issues"))

    if iteration >= max_iter:
        return "finalize"
    if confidence >= 0.75 and not issues:
        return "finalize"
    if iteration >= max_iter - 1 and confidence >= 0.6 and len(issues) <= 1:
        return "finalize"
    return "refine"


def refine_draft(state: SlotState) -> SlotState:
    """根据评审意见改写草稿。"""
    slot = state.get("slot_definition") or {}
    evidence = state.get("evidence") or {}
    sid = state.get("slot_id") or ""
    draft_text = state.get("draft_text") or ""
    issues = state.get("review_issues") or []

    if not issues:
        # 没有问题也增加 iteration，防止无限循环
        return {"iteration": (state.get("iteration") or 0) + 1}

    sys_prompt = REFINE_PROMPT_BASE
    payload = {
        "slot": slot,
        "original_draft": draft_text,
        "review_issues": issues,
        "evidence": evidence,
        "iteration": state.get("iteration", 0),
    }

    try:
        llm = _llm_from_state(state, temperature=0.25)
        refined: SlotRefine = _invoke_structured(llm, sys_prompt, payload, SlotRefine)
        if not refined.draft_text:
            # 改写失败，保持原稿但降低置信度
            return {
                "iteration": (state.get("iteration") or 0) + 1,
                "confidence": min(_f(state.get("confidence"), 0.5), 0.4),
                "risk_notes": (state.get("risk_notes") or []) + ["refine_failed: empty output"],
            }
        return {
            "draft_text": refined.draft_text,
            "source_refs": refined.source_refs or state.get("source_refs", []),
            "confidence": refined.confidence,
            "risk_notes": refined.risk_notes or state.get("risk_notes", []),
            "iteration": (state.get("iteration") or 0) + 1,
            "review_issues": [],  # 清空问题，下一轮 self_review 重新评估
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "iteration": (state.get("iteration") or 0) + 1,
            "confidence": min(_f(state.get("confidence"), 0.5), 0.4),
            "risk_notes": (state.get("risk_notes") or []) + [f"refine_failed: {exc}"],
        }


def finalize_slot(state: SlotState) -> SlotState:
    """整理最终输出，将结果写入 slot_output（供主状态聚合）。"""
    sid = state.get("slot_id") or ""
    draft_text = state.get("draft_text") or ""
    confidence = _f(state.get("confidence"), 0.0)
    source_refs = _slist(state.get("source_refs"))
    risk_notes = _slist(state.get("risk_notes"))
    review_issues = _slist(state.get("review_issues"))

    # 若有未解决的 review_issues，降低置信度并加入 risk_notes
    if review_issues:
        confidence = min(confidence, 0.55)
        risk_notes = list(dict.fromkeys(risk_notes + review_issues))

    needs_review = bool(risk_notes) or confidence < 0.6

    return {
        "slot_output": {
            sid: {
                "final_text": draft_text,
                "confidence": confidence,
                "source_refs": source_refs,
                "risk_notes": risk_notes,
                "needs_review": needs_review,
                "iterations_used": state.get("iteration", 0),
            }
        }
    }


# ──────────────────────── Subgraph Builder ────────────────────────

def build_slot_subgraph():
    """构建并返回槽位子图（StateGraph 实例）。"""
    from langgraph.graph import END, StateGraph

    graph = StateGraph(SlotState)

    graph.add_node("gather_evidence", gather_evidence)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("self_review", self_review)
    graph.add_node("refine_draft", refine_draft)
    graph.add_node("finalize_slot", finalize_slot)

    graph.set_entry_point("gather_evidence")
    graph.add_edge("gather_evidence", "generate_draft")
    graph.add_edge("generate_draft", "self_review")
    graph.add_conditional_edges(
        "self_review",
        should_refine,
        {"refine": "refine_draft", "finalize": "finalize_slot"},
    )
    graph.add_edge("refine_draft", "self_review")
    graph.add_edge("finalize_slot", END)

    return graph
