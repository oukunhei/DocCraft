from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.documents import Document


class ReportState(TypedDict, total=False):
    """全局工作流状态.

    这里只定义目前需要的字段，后续节点可以在此基础上扩展。
    """

    # 输入：原项目论文文件路径（或标识）
    source_file: str

    # 输入：报告模板文件路径（例如 template.docx）
    template_file: str

    # 模型与运行参数
    llm_model_name: str
    tool_model_name: str
    reasoning_model_name: str
    tool_base_url: str
    reasoning_base_url: str
    # 兼容旧字段命名，避免历史状态恢复时报类型缺失
    tool_url: str
    reasoning_url: str
    tool_api_key: str
    reasoning_api_key: str
    intermediate_dir: str
    run_id: str

    # 节点1输出：预处理后的文档块
    raw_documents: List[Document]

    # 节点1输出：论文整体摘要信息
    doc_summary: Dict[str, Any]

    # 预留给后续节点使用的字段
    # 模板要点抽取结果
    template_slots: Dict[str, Any]
    # Previous work 搜索得到的资料
    previous_work_docs: List[Document]
    # 节点3调试信息
    search_queries: Dict[str, Any]
    search_debug: Dict[str, Any]
    seed_terms: Dict[str, Any]
    paper_has_related_work: bool

    # 节点4A：逐槽位生成
    slot_evidence_map: Dict[str, Any]
    agent_a_output: Dict[str, Any]
    node4a_completed: bool

    # 节点5：逐槽位复查（仅记录问题，不改写A草稿）
    agent_b_output: Dict[str, Any]
    slot_review_summary: Dict[str, str]
    node5_completed: bool

    # 兼容旧流程字段（已废弃，保留用于历史状态兼容）
    crosscheck_disputes: List[Dict[str, Any]]
    agent_judge_output: Dict[str, Any]
    # 交叉分析笔记
    analysis_notes: Dict[str, Any]
    # analysis_notes 扩展字段：uncertainty_map[slot_id] = {level, points, human_needed}
    node4_current_round: int
    node4_max_rounds: int
    node4_discussion_rounds: List[Dict[str, Any]]
    node4_unresolved_slots: List[str]
    node4_feedback_mode: str
    node4_cli_detail: str
    node4_log_mode: str
    node4_feedback_pending: bool
    node4_feedback_request_file: str
    node4_human_feedback: Dict[str, Any]
    # 最终生成的报告
    final_report: str
    final_report_docx: str
    final_review_checklist_docx: str
    report_build_summary: Dict[str, Any]

    # 新增：并行子图聚合输出（slot_id -> 最终输出）
    slot_outputs: Annotated[Dict[str, Any], operator.ior]

    # 新增：大纲规划结果
    outline_plan: Dict[str, Any]

    # 新增：一致性审查记录
    consistency_issues: List[Dict[str, Any]]

    # 错误信息
    errors: Dict[str, str]
