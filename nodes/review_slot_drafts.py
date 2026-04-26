from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from nodes.slot_skills import get_review_skill, get_skill_snapshot
    from state import ReportState
except Exception:  # noqa: BLE001
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from nodes.slot_skills import get_review_skill, get_skill_snapshot
    from state import ReportState


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


def _llm(state: ReportState, temperature: float = 0.15) -> ChatOpenAI:
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


def _slot_map(slots: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(s.get("slot_id") or ""): s for s in slots if str(s.get("slot_id") or "").strip()}


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


def _default_review(slot_id: str, error: str) -> Dict[str, Any]:
    return {
        "slot_id": slot_id,
        "review_text": "复查调用失败，建议人工逐条核验关键结论与引用。",
        "revised_text": "",
        "source_refs": [],
        "confidence": 0.2,
        "disagreements": [f"review_failed: {error}"],
    }


def _has_untraceable_claim(draft_text: str, refs: List[str]) -> bool:
    if not draft_text:
        return False
    lower = draft_text.lower()
    has_placeholder = bool(re.search(r"\[待补证据\]|\[需补充|请在此处|待填充|x\.xx", lower, re.IGNORECASE))
    has_numeric_claim = bool(re.search(r"\d+(?:\.\d+)?\s*(%|ms|s|x|倍|fps|f1|mape|mae)?", draft_text, re.IGNORECASE))
    has_named_entities = bool(re.search(r"\b(coco|mot|yolo|bert|transformer|resnet|pytorch|cuda|docker|kubernetes)\b", lower))
    if has_placeholder:
        return True
    if (has_numeric_claim or has_named_entities) and not refs:
        return True
    return False


def review_slot_drafts(state: ReportState) -> Dict[str, Any]:
    """节点5：逐槽位复查Agent A草稿，只记录问题不改写生成内容。"""

    template_slots: Dict[str, Any] = state.get("template_slots") or {}
    a_out: Dict[str, Any] = state.get("agent_a_output") or {}
    slot_evidence_map: Dict[str, Any] = state.get("slot_evidence_map") or {}

    if not template_slots:
        return {"errors": {"review_slot_drafts": "missing template_slots"}}
    if not a_out.get("slots"):
        return {"errors": {"review_slot_drafts": "missing agent_a_output.slots"}}

    slots = _slots(template_slots)
    if not slots:
        return {"errors": {"review_slot_drafts": "template_slots has no valid slots"}}

    run_id = state.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(state.get("intermediate_dir") or "artifacts/intermediate") / "node5" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[node5] start per-slot review, slots={len(slots)}", flush=True)

    llm = _llm(state, temperature=0.15)
    a_map = _slot_map(a_out.get("slots", []))

    b_slots: List[Dict[str, Any]] = []
    review_errors: List[Dict[str, str]] = []
    crosscheck_disputes: List[Dict[str, Any]] = []
    review_summary: Dict[str, str] = {}
    skill_map: Dict[str, Any] = {}

    for idx, slot in enumerate(slots, 1):
        sid = str(slot.get("slot_id") or "").strip()
        if not sid:
            continue

        a_slot = a_map.get(sid, {})
        skill_snapshot = get_skill_snapshot(slot)
        skill_map[sid] = skill_snapshot
        sys_prompt = REVIEW_PROMPT_BASE.format(slot_skill=get_review_skill(slot))
        payload = {
            "slot": slot,
            "slot_evidence": slot_evidence_map.get(sid, {}),
            "agent_a_slot": a_slot,
            "constraints": {
                "language": "zh",
                "style": "project_report",
                "strict_evidence": True,
            },
        }

        try:
            b_out = _invoke(llm, sys_prompt, payload)
            b_slot = _normalize_review_output(sid, b_out)
        except Exception as exc:  # noqa: BLE001
            review_errors.append({"slot_id": sid, "error": str(exc)})
            b_slot = _default_review(sid, str(exc))

        b_slots.append(b_slot)
        review_summary[sid] = str(b_slot.get("review_text") or "")

        reasons: List[str] = []
        a_conf = _f(a_slot.get("confidence"), 0.5)
        b_conf = _f(b_slot.get("confidence"), 0.5)
        if abs(a_conf - b_conf) >= 0.25:
            reasons.append("confidence_gap")
        a_refs = set(_slist(a_slot.get("source_refs")))
        b_refs = set(_slist(b_slot.get("source_refs")))
        if a_refs and b_refs and a_refs.isdisjoint(b_refs):
            reasons.append("evidence_ref_disjoint")
        if _slist(b_slot.get("disagreements")):
            reasons.append("review_disagreement")
        if reasons:
            crosscheck_disputes.append(
                {
                    "slot_id": sid,
                    "reasons": reasons,
                    "a_confidence": a_conf,
                    "b_confidence": b_conf,
                }
            )

        print(f"[node5] reviewed slot {idx}/{len(slots)}: {sid}", flush=True)

    b_out = {"slots": b_slots}
    b_map = _slot_map(b_slots)
    dispute_map = {str(x.get("slot_id") or ""): _slist(x.get("reasons")) for x in crosscheck_disputes}

    filled_slots: Dict[str, str] = {}
    slot_confidence: Dict[str, float] = {}
    slot_sources: Dict[str, List[str]] = {}
    review_flags: List[str] = []
    uncertainty_map: Dict[str, Any] = {}

    for slot in slots:
        sid = str(slot.get("slot_id") or "").strip()
        if not sid:
            continue

        a_slot = a_map.get(sid, {})
        b_slot = b_map.get(sid, {})

        draft_text = str(a_slot.get("draft_text") or "").strip()
        if draft_text:
            filled_slots[sid] = draft_text

        a_conf = _f(a_slot.get("confidence"), 0.5)
        b_conf = _f(b_slot.get("confidence"), a_conf)
        final_conf = min(a_conf, b_conf)

        refs = list(dict.fromkeys(_slist(a_slot.get("source_refs")) + _slist(b_slot.get("source_refs"))))
        slot_sources[sid] = refs

        points = _slist(b_slot.get("disagreements")) + _slist(a_slot.get("risk_notes"))
        points = list(dict.fromkeys(points))

        if _has_untraceable_claim(draft_text, refs):
            final_conf = min(final_conf, 0.3)
            points.append("检测到不可追溯事实：文本包含具体结论但缺少可对应的项目证据")
            points = list(dict.fromkeys(points))

        slot_confidence[sid] = final_conf

        needs_review = bool(points) or final_conf < 0.6
        if sid in dispute_map:
            needs_review = True
            points.extend([f"A/B存在分歧: {reason}" for reason in dispute_map[sid]])
            points = list(dict.fromkeys(points))

        if needs_review:
            review_flags.append(sid)

        if final_conf < 0.6 or len(points) >= 2:
            level = "high"
        elif needs_review:
            level = "medium"
        else:
            level = "low"

        human_needed: List[str] = []
        if needs_review:
            human_needed = [
                "核对本槽位关键结论与原文证据是否逐条对应",
                "补充缺失的实验数据、统计口径或引用来源",
            ]

        uncertainty_map[sid] = {
            "level": level,
            "points": points,
            "human_needed": human_needed,
        }

    analysis_notes = {
        "filled_slots": filled_slots,
        "slot_confidence": slot_confidence,
        "slot_sources": slot_sources,
        "review_flags": review_flags,
        "missing_info_slots": [sid for sid, conf in slot_confidence.items() if conf < 0.6],
        "uncertainty_map": uncertainty_map,
        "slot_review_summary": review_summary,
    }

    metadata = {
        "run_id": run_id,
        "total_slots": len(slots),
        "review_errors": review_errors,
        "slot_skill_map": skill_map,
    }

    _write_json(out_dir / "agent_b_output.json", b_out)
    _write_json(out_dir / "crosscheck_disputes.json", crosscheck_disputes)
    _write_json(out_dir / "analysis_notes.json", analysis_notes)
    _write_json(out_dir / "metadata.json", metadata)

    return {
        "run_id": run_id,
        "agent_b_output": b_out,
        "crosscheck_disputes": crosscheck_disputes,
        "analysis_notes": analysis_notes,
        "node4a_completed": bool(state.get("node4a_completed")),
        "node5_completed": True,
        "node4_current_round": 1,
        "node4_max_rounds": 1,
        "node4_discussion_rounds": [{"round": 1, "disputes_count": len(crosscheck_disputes)}],
        "node4_unresolved_slots": review_flags,
        "node4_feedback_mode": "none",
        "node4_feedback_pending": False,
        "node4_feedback_request_file": "",
        "node4_human_feedback": {},
    }
