from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from nodes.preprocess_paper import preprocess_project_paper
from nodes.extract_templete_requirements import extract_template_requirements
from nodes.search_related import search_related
from nodes.analyze_slots_with_crosscheck import analyze_slots_with_crosscheck
from nodes.build_report_docx import build_report_docx
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
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_input_file(path_str: str) -> Path:
    """Resolve user input file path in a tolerant order.

    1) absolute / home-expanded path
    2) current working directory relative path
    3) project-root relative path
    """
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
    """集中读取环境变量并构造初始状态，避免节点内配置分叉。"""

    reasoning_model_name = _env_first("REASONING_MODEL_NAME") or "deepseek-chat"
    tool_model_name = _env_first("TOOL_MODEL_NAME") or "step-3.5-flash"
    reasoning_api_key = _env_first("REASONING_API_KEY")
    tool_api_key = _env_first("TOOL_API_KEY")
    reasoning_base_url = _env_first("REASONING_BASE_URL", "REASONING_URL")
    tool_base_url = _env_first("TOOL_BASE_URL", "TOOL_URL")
    run_id = args.run_id or _env_first("RUN_ID")
    node4_feedback_mode = "none"
    node4_cli_detail = "concise"
    node4_log_mode = "minimal"

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
        "node4_max_rounds": 3,
        "node4_feedback_mode": node4_feedback_mode,
        "node4_cli_detail": node4_cli_detail,
        "node4_log_mode": node4_log_mode,
        "node4_feedback_pending": False,
        "llm_model_name": reasoning_model_name,
        "intermediate_dir": args.intermediate_dir,
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


def _route_after_node4(state: ReportState) -> str:
    if state.get("node4_feedback_pending"):
        return "end"

    errors = state.get("errors") or {}
    if "analyze_slots_with_crosscheck" in errors:
        return "end"

    analysis_notes = state.get("analysis_notes") or {}
    filled_slots = analysis_notes.get("filled_slots") or {}
    if not filled_slots:
        return "end"

    return "build_report_docx"


def build_graph() -> StateGraph[ReportState]:
    """构建整个报告写作的工作流图.

    当前接入：
    - 节点1：预处理原项目论文（preprocess_project_paper）
    - 节点1内置：落盘 node1 中间结果
    - 节点2：解析报告模板技术内容 slot（extract_template_requirements）
    - 节点2内置：落盘 node2 中间结果
    - 节点3：检索 related work 和相关新闻（search_related）
    - 节点4：双Agent交叉验证分析（analyze_slots_with_crosscheck）
    - 节点5：生成新的报告docx（build_report_docx）
    """

    graph = StateGraph(ReportState)

    graph.add_node("preprocess_project_paper", preprocess_project_paper)
    graph.add_node("extract_template_requirements", extract_template_requirements)
    graph.add_node("search_related", search_related)
    graph.add_node("analyze_slots_with_crosscheck", analyze_slots_with_crosscheck)
    graph.add_node("build_report_docx", build_report_docx)

    graph.set_entry_point("preprocess_project_paper")
    graph.add_edge("preprocess_project_paper", "extract_template_requirements")
    graph.add_edge("extract_template_requirements", "search_related")
    graph.add_edge("search_related", "analyze_slots_with_crosscheck")
    graph.add_conditional_edges(
        "analyze_slots_with_crosscheck",
        _route_after_node4,
        {
            "build_report_docx": "build_report_docx",
            "end": END,
        },
    )
    graph.add_edge("build_report_docx", END)

    return graph


def main() -> None:
    load_dotenv()
    args = _parse_args()
    _validate_paths(args)

    print("[flow] starting workflow...", flush=True)
    graph = build_graph().compile()

    # 在入口统一加载模型/密钥/URL配置，节点内只消费 state。
    initial_state = _build_initial_state(args)
    print("[flow] invoking graph (this may take 20-90s depending on API/network)...", flush=True)
    final_state = graph.invoke(initial_state)
    print("[flow] graph finished", flush=True)

    print("Doc summary:", final_state.get("doc_summary"))
    print("Template slots:", final_state.get("template_slots"))
    prev_docs = final_state.get("previous_work_docs") or []
    print("Previous work docs count:", len(prev_docs))
    print("Search debug:", final_state.get("search_debug"))
    notes = final_state.get("analysis_notes") or {}
    print("Node4 filled slots count:", len((notes.get("filled_slots") or {}).keys()))
    print("Node4 review flags:", notes.get("review_flags"))
    print("Node4 round:", final_state.get("node4_current_round"))
    print("Node4 unresolved slots:", final_state.get("node4_unresolved_slots"))
    print("Node4 feedback pending:", final_state.get("node4_feedback_pending"))
    print("Node4 feedback request file:", final_state.get("node4_feedback_request_file"))
    print("Final report docx:", final_state.get("final_report_docx"))
    print("Review checklist docx:", final_state.get("final_review_checklist_docx"))
    print("Report build summary:", final_state.get("report_build_summary"))
    if "errors" in final_state:
        print("Errors:", final_state["errors"])


if __name__ == "__main__":
    main()