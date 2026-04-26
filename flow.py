from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from nodes.preprocess_paper import preprocess_project_paper
from nodes.extract_templete_requirements import extract_template_requirements
from nodes.build_report_docx import build_report_docx
from nodes.slot_subgraph import build_slot_subgraph
from nodes.consistency_check import consistency_check
from state import ReportState


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DocCraft workflow from project paper to final report docx."
    )
    parser.add_argument(
        "--source-file",
        default="project_doc.pdf",
        help="Path to the source project paper (default: project_doc.pdf).",
    )
    parser.add_argument(
        "--template-file",
        default="template.docx",
        help="Path to the report template docx (default: template.docx).",
    )
    parser.add_argument(
        "--intermediate-dir",
        default="artifacts/intermediate",
        help="Directory for intermediate artifacts (default: artifacts/intermediate).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID used for artifact grouping.",
    )
    parser.add_argument(
        "--max-slot-iterations",
        type=int,
        default=3,
        help="Max iterations per slot in the refine loop (default: 3).",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_input_file(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        return cwd_path

    return (_project_root() / candidate).resolve()


def _resolve_output_dir(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    return (_project_root() / candidate).resolve()


def _build_initial_state(args: argparse.Namespace) -> ReportState:
    reasoning_model_name = _env_first("REASONING_MODEL_NAME") or "deepseek-chat"
    tool_model_name = _env_first("TOOL_MODEL_NAME") or "step-3.5-flash"
    reasoning_api_key = _env_first("REASONING_API_KEY")
    tool_api_key = _env_first("TOOL_API_KEY")
    reasoning_base_url = _env_first("REASONING_BASE_URL", "REASONING_URL")
    tool_base_url = _env_first("TOOL_BASE_URL", "TOOL_URL")
    run_id = args.run_id or _env_first("RUN_ID")

    missing = []
    if not reasoning_api_key:
        missing.append("REASONING_API_KEY")
    if not reasoning_base_url:
        missing.append("REASONING_BASE_URL or REASONING_URL")
    if not tool_api_key:
        missing.append("TOOL_API_KEY")
    if not tool_base_url:
        missing.append("TOOL_BASE_URL or TOOL_URL")

    if missing:
        raise RuntimeError(f"missing required env config: {', '.join(missing)}")

    return {
        "source_file": args.source_file,
        "template_file": args.template_file,
        "reasoning_model_name": reasoning_model_name,
        "reasoning_api_key": reasoning_api_key,
        "reasoning_base_url": reasoning_base_url,
        "reasoning_url": reasoning_base_url,
        "tool_model_name": tool_model_name,
        "tool_api_key": tool_api_key,
        "tool_base_url": tool_base_url,
        "tool_url": tool_base_url,
        "node4_current_round": 1,
        "node4_max_rounds": 1,
        "node4_feedback_mode": "none",
        "node4_cli_detail": "concise",
        "node4_log_mode": "minimal",
        "node4_feedback_pending": False,
        "llm_model_name": reasoning_model_name,
        "intermediate_dir": args.intermediate_dir,
        "max_slot_iterations": args.max_slot_iterations,
        **({"run_id": run_id} if run_id else {}),
    }


def _validate_paths(args: argparse.Namespace) -> None:
    args.source_file = str(_resolve_input_file(args.source_file))
    args.template_file = str(_resolve_input_file(args.template_file))
    args.intermediate_dir = str(_resolve_output_dir(args.intermediate_dir))

    if not os.path.isfile(args.source_file):
        raise FileNotFoundError(f"source file not found: {args.source_file}")
    if not os.path.isfile(args.template_file):
        raise FileNotFoundError(f"template file not found: {args.template_file}")
    os.makedirs(args.intermediate_dir, exist_ok=True)


# ──────────────────────── New Orchestration Nodes ────────────────────────

def _collect_slot_defs(template_slots: Dict[str, Any]) -> List[Dict[str, Any]]:
    core = template_slots.get("core_slots") or []
    other = template_slots.get("other_technical_slots") or []
    return [s for s in core + other if s.get("slot_id")]


PLAN_OUTLINE_PROMPT = """你是项目技术报告的大纲规划助手。

基于模板槽位定义和论文摘要，输出一份全局写作计划，确保各槽位之间的逻辑一致性。

输出要求：
1. slot_order: 建议的槽位生成顺序（考虑依赖关系，如"测试分析"应在"技术方案"之后）
2. terminology_glossary: 术语统一表（识别可能的不一致术语并给出统一用法）
3. cross_slot_refs: 跨槽位引用关系（哪些槽位会引用其他槽位的内容）
4. data_reference_rules: 数据引用一致性规则（如"表2必须在测试分析和创新点中保持一致"）
5. key_messages: 全文必须贯穿的 3-5 个核心信息点

严格返回 JSON，不要有多余文本。
"""


def plan_outline(state: ReportState) -> Dict[str, Any]:
    """在逐槽位生成前输出全局写作计划，确保跨槽位一致性。"""
    template_slots = state.get("template_slots") or {}
    slots = _collect_slot_defs(template_slots)
    doc_summary = state.get("doc_summary") or {}

    if not slots:
        return {"outline_plan": {}}

    # 构建轻量大纲（保底）
    fallback_outline = {
        "slot_order": [s["slot_id"] for s in slots],
        "terminology_glossary": {},
        "cross_slot_refs": {},
        "data_reference_rules": [],
        "key_messages": [],
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "doc_language": doc_summary.get("language", "zh"),
        "total_slots": len(slots),
    }

    # 尝试 LLM-based 大纲规划
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        from nodes.schemas import OutlinePlan

        api_key = state.get("reasoning_api_key")
        base_url = state.get("reasoning_url") or state.get("reasoning_base_url")
        model = state.get("reasoning_model_name") or state.get("llm_model_name") or "deepseek-chat"
        llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0.3)

        payload = {
            "doc_summary": doc_summary,
            "template_slots": template_slots,
        }
        structured_llm = llm.with_structured_output(OutlinePlan)
        plan: OutlinePlan = structured_llm.invoke([
            SystemMessage(content=PLAN_OUTLINE_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])

        outline = {
            "slot_order": plan.slot_order or fallback_outline["slot_order"],
            "terminology_glossary": plan.terminology_glossary,
            "cross_slot_refs": plan.cross_slot_refs,
            "data_reference_rules": plan.data_reference_rules,
            "key_messages": plan.key_messages,
            "generated_at": fallback_outline["generated_at"],
            "doc_language": fallback_outline["doc_language"],
            "total_slots": fallback_outline["total_slots"],
            "llm_generated": True,
        }
        print(f"[plan_outline] LLM-generated outline OK, slots={len(slots)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        outline = fallback_outline
        outline["llm_generated"] = False
        outline["llm_error"] = str(exc)
        print(f"[plan_outline] fallback to simple outline: {exc}", flush=True)

    return {"outline_plan": outline}


def dispatch_slots(state: ReportState):
    """使用 Send 并行派发所有 slot 到 slot_subgraph。"""
    template_slots = state.get("template_slots") or {}
    slots = _collect_slot_defs(template_slots)
    if not slots:
        print("[dispatch_slots] no slots found, ending workflow", flush=True)
        return "__end__"

    max_iterations = state.get("max_slot_iterations", 3)
    print(f"[dispatch_slots] dispatching {len(slots)} slots in parallel (max_iter={max_iterations})", flush=True)

    return [
        Send(
            "slot_subgraph",
            {
                "slot_id": s["slot_id"],
                "slot_definition": s,
                "raw_documents": state.get("raw_documents", []),
                "seed_terms": state.get("seed_terms", {}),
                "previous_work_docs": state.get("previous_work_docs", []),
                "reasoning_model_name": state.get("reasoning_model_name"),
                "reasoning_api_key": state.get("reasoning_api_key"),
                "reasoning_base_url": state.get("reasoning_base_url") or state.get("reasoning_url"),
                "llm_model_name": state.get("llm_model_name"),
                "tool_model_name": state.get("tool_model_name"),
                "tool_api_key": state.get("tool_api_key"),
                "tool_base_url": state.get("tool_base_url") or state.get("tool_url"),
                "run_id": state.get("run_id") or __import__("datetime").datetime.now(__import__("datetime").UTC).strftime("%Y%m%dT%H%M%SZ"),
                "iteration": 0,
                "max_iterations": max_iterations,
            },
        )
        for s in slots
    ]


def aggregate_slots(state: ReportState) -> Dict[str, Any]:
    """聚合所有并行子图的 slot_output，生成 analysis_notes 供 build_report 使用。"""
    slot_outputs = state.get("slot_outputs") or {}
    print(f"[aggregate_slots] aggregating {len(slot_outputs)} slot results", flush=True)

    filled_slots: Dict[str, str] = {}
    slot_confidence: Dict[str, float] = {}
    slot_sources: Dict[str, List[str]] = {}
    review_flags: List[str] = []
    uncertainty_map: Dict[str, Any] = {}
    slot_review_summary: Dict[str, str] = {}

    for sid, out in slot_outputs.items():
        final_text = str(out.get("final_text") or "").strip()
        confidence = float(out.get("confidence") or 0.0)
        sources = [str(x) for x in (out.get("source_refs") or []) if str(x).strip()]
        risk_notes = [str(x) for x in (out.get("risk_notes") or []) if str(x).strip()]
        needs_review = bool(out.get("needs_review"))
        iterations = int(out.get("iterations_used") or 0)

        if final_text:
            filled_slots[sid] = final_text
        slot_confidence[sid] = confidence
        slot_sources[sid] = sources
        slot_review_summary[sid] = f"iterations={iterations}, confidence={confidence:.2f}"

        if needs_review or confidence < 0.6 or risk_notes:
            review_flags.append(sid)

        if confidence < 0.6 or len(risk_notes) >= 2:
            level = "high"
        elif needs_review or risk_notes:
            level = "medium"
        else:
            level = "low"

        human_needed: List[str] = []
        if needs_review or confidence < 0.6:
            human_needed = [
                "核对本槽位关键结论与原文证据是否逐条对应",
                "补充缺失的实验数据、统计口径或引用来源",
            ]

        uncertainty_map[sid] = {
            "level": level,
            "points": risk_notes,
            "human_needed": human_needed,
        }

    analysis_notes = {
        "filled_slots": filled_slots,
        "slot_confidence": slot_confidence,
        "slot_sources": slot_sources,
        "review_flags": review_flags,
        "missing_info_slots": [sid for sid, conf in slot_confidence.items() if conf < 0.6],
        "uncertainty_map": uncertainty_map,
        "slot_review_summary": slot_review_summary,
    }

    total_slots = len(slot_outputs)
    reviewed_slots = len(review_flags)
    print(
        f"[aggregate_slots] done: total={total_slots}, reviewed={reviewed_slots}, "
        f"avg_conf={sum(slot_confidence.values()) / max(len(slot_confidence), 1):.2f}",
        flush=True,
    )

    return {
        "analysis_notes": analysis_notes,
        "node4a_completed": True,
        "node5_completed": True,
        "node4_current_round": 1,
        "node4_max_rounds": 1,
        "node4_discussion_rounds": [{"round": 1, "disputes_count": reviewed_slots}],
        "node4_unresolved_slots": review_flags,
        "node4_feedback_mode": "none",
        "node4_feedback_pending": False,
        "node4_feedback_request_file": "",
        "node4_human_feedback": {},
    }


def _route_after_aggregate(state: ReportState) -> str:
    errors = state.get("errors") or {}
    if "aggregate_slots" in errors:
        return "end"
    analysis_notes = state.get("analysis_notes") or {}
    filled_slots = analysis_notes.get("filled_slots") or {}
    if not filled_slots:
        return "end"
    return "build_report_docx"


# ──────────────────────── Graph Builder ────────────────────────

def build_graph() -> StateGraph[ReportState]:
    """构建新版 DocCraft 工作流图。

    架构：
    - 线性预处理：preprocess → extract → plan_outline
    - 并行生成：plan_outline ──Send──→ slot_subgraph (每个槽位独立子图，内部可迭代)
    - 聚合与输出：slot_subgraph ──→ aggregate_slots ──→ build_report_docx
    - 状态持久化：MemorySaver checkpoint
    """
    graph = StateGraph(ReportState)

    # 节点注册
    graph.add_node("preprocess_project_paper", preprocess_project_paper)
    graph.add_node("extract_template_requirements", extract_template_requirements)
    graph.add_node("plan_outline", plan_outline)
    graph.add_node("slot_subgraph", build_slot_subgraph().compile())
    graph.add_node("aggregate_slots", aggregate_slots)
    graph.add_node("consistency_check", consistency_check)
    graph.add_node("build_report_docx", build_report_docx)

    # 边定义
    graph.set_entry_point("preprocess_project_paper")
    graph.add_edge("preprocess_project_paper", "extract_template_requirements")
    graph.add_edge("extract_template_requirements", "plan_outline")

    # 动态并行派发
    graph.add_conditional_edges(
        "plan_outline",
        dispatch_slots,
        {"slot_subgraph": "slot_subgraph", "__end__": END},
    )

    # 聚合后路由
    graph.add_edge("slot_subgraph", "aggregate_slots")
    graph.add_edge("aggregate_slots", "consistency_check")
    graph.add_conditional_edges(
        "consistency_check",
        _route_after_aggregate,
        {"build_report_docx": "build_report_docx", "end": END},
    )
    graph.add_edge("build_report_docx", END)

    return graph


# ──────────────────────── Main ────────────────────────

def main() -> None:
    load_dotenv()
    args = _parse_args()
    _validate_paths(args)

    print("[flow] starting workflow...", flush=True)
    graph = build_graph()

    # 接入 MemorySaver 实现状态持久化与可恢复执行
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    initial_state = _build_initial_state(args)
    config = {"configurable": {"thread_id": args.run_id or "default"}}

    print("[flow] invoking graph (this may take 20-90s depending on API/network)...", flush=True)
    final_state = compiled.invoke(initial_state, config=config)
    print("[flow] graph finished", flush=True)

    print("Doc summary:", final_state.get("doc_summary"))
    print("Template slots:", final_state.get("template_slots"))
    notes = final_state.get("analysis_notes") or {}
    print("Filled slots count:", len((notes.get("filled_slots") or {}).keys()))
    print("Review flags:", notes.get("review_flags"))
    print("Final report docx:", final_state.get("final_report_docx"))
    print("Review checklist docx:", final_state.get("final_review_checklist_docx"))
    print("Report build summary:", final_state.get("report_build_summary"))
    if "errors" in final_state:
        print("Errors:", final_state["errors"])


if __name__ == "__main__":
    main()
