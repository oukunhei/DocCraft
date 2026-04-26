"""跨槽位一致性审查节点

在 aggregate_slots 之后运行，检查所有槽位之间的术语一致性、
数据引用一致性和逻辑矛盾。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from state import ReportState
    from nodes.schemas import ConsistencyReport
except Exception:  # noqa: BLE001
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from state import ReportState
    from nodes.schemas import ConsistencyReport


CONSISTENCY_PROMPT = """你是项目技术报告的一致性审查专家。

请审查以下各槽位的最终文本，检查是否存在以下问题：
1. 术语不一致：同一概念在不同槽位使用了不同词汇（如时而叫"节点"，时而叫"服务器"）
2. 数据引用矛盾：同一实验数据在不同槽位描述不一致（如测试分析中说准确率 92%，创新点中说 93%）
3. 时间线/逻辑矛盾：技术方案与系统实现的时间线或逻辑冲突
4. 引用缺失：某槽位引用了另一槽位的内容但另一槽位未出现
5. 章节重复：两个槽位的内容大面积机械重复

输出严格 JSON，符合以下结构：
{
  "issues": [
    {"slot_ids": ["slot_a", "slot_b"], "issue_type": "terminology|timeline|data|contradiction|redundancy", "description": "问题描述", "suggestion": "修改建议"}
  ],
  "overall_score": 0.0-1.0,
  "summary": "审查总结"
}
"""


def _llm(state: ReportState, temperature: float = 0.15) -> ChatOpenAI:
    api_key = state.get("reasoning_api_key")
    base_url = state.get("reasoning_url") or state.get("reasoning_base_url")
    if not api_key or not base_url:
        raise ValueError("missing reasoning_api_key / reasoning_base_url in state")
    model = state.get("reasoning_model_name") or state.get("llm_model_name") or "deepseek-chat"
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=temperature)


def consistency_check(state: ReportState) -> Dict[str, Any]:
    """审查跨槽位一致性，返回 issues 列表和更新后的 analysis_notes。"""
    analysis_notes = state.get("analysis_notes") or {}
    filled_slots = analysis_notes.get("filled_slots") or {}

    if len(filled_slots) < 2:
        print("[consistency_check] skipped: less than 2 slots", flush=True)
        return {"consistency_issues": []}

    # 只提交有效文本（避免过长）
    payload = {
        "slots": {
            sid: text[:800] for sid, text in filled_slots.items() if text.strip()
        }
    }

    try:
        llm = _llm(state, temperature=0.15)
        structured_llm = llm.with_structured_output(ConsistencyReport)
        report: ConsistencyReport = structured_llm.invoke([
            SystemMessage(content=CONSISTENCY_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])

        issues = [i.model_dump() for i in report.issues]
        print(
            f"[consistency_check] done: issues={len(issues)}, score={report.overall_score:.2f}",
            flush=True,
        )

        # 将一致性分数合并到 analysis_notes
        analysis_notes["consistency_score"] = report.overall_score
        analysis_notes["consistency_summary"] = report.summary
        analysis_notes["consistency_issues"] = issues

        return {
            "consistency_issues": issues,
            "analysis_notes": analysis_notes,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[consistency_check] failed: {exc}", flush=True)
        return {
            "consistency_issues": [{"error": str(exc)}],
            "errors": {"consistency_check": str(exc)},
        }
