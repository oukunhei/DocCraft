from __future__ import annotations
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from state import ReportState
except Exception:  # noqa: BLE001
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from state import ReportState


TEMPLATE_SYSTEM_PROMPT = """你是中国大学生计算机设计大赛（4C）报告模板分析助手。
你的任务是：
1. 阅读给定的中文/中英混合报告模板内容（通常来自 Word 模板）。
2. 忽略作品编号、名称、填写日期等和技术内容无关的基础信息填空，找出与"技术内容"相关、需要由 AI 帮忙补充完整的部分。
3. 请务必把参考文献版块包含在内。

【竞赛报告核心关注点（评审权重由高到低）】：
- 创新点分析：技术/应用层面的独创性，是评分权重最高的板块（约30%），必须提炼 1-3 个具体可验证的创新点
- 技术方案与实现：系统架构、算法选型、关键技术原理，需图文并茂展示技术深度（约30%）
- 问题背景与现有方案对比：说明 why、引出创新必要性，需有参考文献支撑（约20%）
- 测试分析：用数据/实验证明系统有效性，所有宣称必须有测试数据背书
- 应用价值与推广：实用性、社会意义、落地场景

3. 把需要 AI 生成的技术内容概括为若干个 slot，每个 slot 给出：
   - 规范化的 slot_id（使用小写下划线命名）
   - 中文标题（title），使用模板里的小节名称
   - 说明（description）：这一部分希望填入什么内容、重点有哪些

输出时严格返回 JSON 格式，结构固定为：
{
  "core_slots": [{"slot_id": "...", "title": "...", "description": "...", "priority": "..."}],
  "other_technical_slots": [...]
}
"""


def _load_template_text(path: Path) -> str:
    """读取模板文件的纯文本内容。"""
    if path.suffix.lower() == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    return "\n".join(d.page_content for d in docs)


def _safe_json(text: str) -> Dict[str, Any]:
    """从模型返回的文本中提取 JSON。"""
    raw = text.strip()
    if not raw:
        raise ValueError("empty response")
    # 直接尝试解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 尝试提取代码块中的 JSON
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, re.IGNORECASE)
    if m:
        return json.loads(m.group(1))
    # 尝试提取大括号包围的内容
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e > s:
        return json.loads(raw[s:e+1])
    raise ValueError(f"Cannot parse JSON from: {raw[:200]}")


def _normalize_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
    """规范化 slot 结构，确保字段齐全且类型正确。"""
    core = slots.get("core_slots", [])
    other = slots.get("other_technical_slots", [])
    if not isinstance(core, list):
        core = []
    if not isinstance(other, list):
        other = []

    def _normalize_item(item: Any) -> Dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        sid = str(item.get("slot_id", "")).strip()
        if not sid:
            return None
        title = str(item.get("title", sid)).strip()
        desc = str(item.get("description", "需要补充该技术内容。"))
        priority = str(item.get("priority", "medium")).lower()
        if priority not in {"high", "medium", "low"}:
            priority = "medium"
        return {"slot_id": sid, "title": title, "description": desc, "priority": priority}

    return {
        "core_slots": [x for x in (_normalize_item(i) for i in core) if x],
        "other_technical_slots": [x for x in (_normalize_item(i) for i in other) if x],
    }


def _fallback_slots_from_text(template_text: str) -> Dict[str, Any]:
    """当 LLM 不可用时，基于关键词生成默认 slot 列表。"""
    text = template_text.lower()
    core, other = [], []
    patterns = [
        ("problem_background", "问题背景", ["背景", "问题来源"]),
        ("existing_solutions", "现有方案", ["现有", "相关工作"]),
        ("core_pain_points", "痛点问题", ["痛点", "要解决"]),
        ("solution_approach", "解决思路", ["思路", "需求"]),
        ("technical_scheme", "技术方案", ["技术方案", "技术路线"]),
        ("system_implementation", "系统实现", ["系统实现", "实现"]),
        ("testing_analysis", "测试分析", ["测试", "分析"]),
        ("innovation_analysis", "创新点", ["创新", "特色"]),
        ("application_promotion", "应用推广", ["应用", "推广"]),
        ("future_work", "展望", ["展望", "未来"]),
    ]
    for sid, title, kws in patterns:
        if any(kw in text for kw in kws):
            item = {
                "slot_id": sid,
                "title": title,
                "description": f"根据模板要求补充{title}内容。",
                "priority": "high" if sid in {"innovation_analysis", "technical_scheme", "testing_analysis"} else "medium",
            }
            if sid in {"problem_background", "existing_solutions", "core_pain_points", "solution_approach",
                       "technical_scheme", "system_implementation", "testing_analysis"}:
                core.append(item)
            else:
                other.append(item)

    if not core and not other:
        # 保底默认
        core = [
            {"slot_id": "technical_scheme", "title": "技术方案", "description": "说明方案设计、关键技术。", "priority": "high"},
            {"slot_id": "system_implementation", "title": "系统实现", "description": "说明工程实现过程。", "priority": "medium"},
            {"slot_id": "testing_analysis", "title": "测试分析", "description": "测试设置、结果与对比。", "priority": "high"},
        ]
    return {"core_slots": core, "other_technical_slots": other}


def _persist_node2_outputs(
    intermediate_dir: str,
    run_id: str,
    template_slots: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    node2_dir = Path(intermediate_dir) / "node2" / run_id
    node2_dir.mkdir(parents=True, exist_ok=True)
    (node2_dir / "template_slots.json").write_text(
        json.dumps(template_slots, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (node2_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_template_requirements(state: ReportState) -> Dict[str, Any]:
    """从模板中提取技术内容 slot，返回包含 template_slots 的字典。"""
    template_path = Path(state.get("template_file", "template.docx"))
    run_id = state.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    intermediate_dir = str(state.get("intermediate_dir") or "artifacts/intermediate")

    print(
        f"[node2] start extract_template_requirements, template={template_path}, run_id={run_id}",
        flush=True,
    )

    template_text = _load_template_text(template_path)

    model_name = state.get("reasoning_model_name") or state.get("llm_model_name") or "deepseek-chat"
    api_key = state.get("reasoning_api_key")
    base_url = state.get("reasoning_base_url") or state.get("reasoning_url")
    meta: Dict[str, Any] = {
        "model_name": model_name,
        "used_fallback": False,
        "fallback_reason": "",
    }

    llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.7)
    messages = [
        SystemMessage(content=TEMPLATE_SYSTEM_PROMPT),
        HumanMessage(content=f"模板内容如下：\n\n{template_text}"),
    ]

    try:
        response = llm.invoke(messages)
        content = str(response.content)
        slots = _normalize_slots(_safe_json(content))
        if not (slots.get("core_slots") or slots.get("other_technical_slots")):
            raise ValueError("空 slots")
    except Exception as exc:
        # 解析失败时回退
        slots = _normalize_slots(_fallback_slots_from_text(template_text))
        meta["used_fallback"] = True
        meta["fallback_reason"] = "llm_invoke_or_parse_failed"
        meta["error"] = str(exc)

    _persist_node2_outputs(intermediate_dir, run_id, slots, meta)

    node2_dir = Path(intermediate_dir) / "node2" / run_id
    total_slots = len(slots.get("core_slots") or []) + len(slots.get("other_technical_slots") or [])
    if meta["used_fallback"]:
        print(
            (
                f"[node2] fallback: reason={meta['fallback_reason']}, slots={total_slots}, "
                f"artifacts={node2_dir}"
            ),
            flush=True,
        )
    else:
        print(
            f"[node2] done: slots={total_slots}, model={model_name}, artifacts={node2_dir}",
            flush=True,
        )

    return {"run_id": run_id, "template_slots": slots}


if __name__ == "__main__":
    # 简单测试入口
    state = ReportState(template_file="template.docx")
    result = extract_template_requirements(state)
    print(json.dumps(result["template_slots"], indent=2, ensure_ascii=False))