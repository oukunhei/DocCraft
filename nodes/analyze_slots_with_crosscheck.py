"""节点4：单轮双Agent交叉验证

- Agent A：基于证据生成草案
- Agent B：独立复核并给出修订建议
- Judge：综合裁决，直接输出最终文本
- 无多轮循环、无CLI阻塞、无人工反馈环节
- 有争议或低置信度的槽位由 Judge 直接选最优版本
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from state import ReportState
except Exception:  # noqa: BLE001
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from state import ReportState


# ──────────────────────── Prompts ────────────────────────

AGENT_A_PROMPT = """你是 Agent A，专门为中国大学生计算机设计大赛（4C）撰写技术作品报告。

## 核心任务
基于每个槽位(slot)的证据包，生成可直接写入参赛报告的高质量段落。

## 竞赛报告评审权重（对应体现）
- 创新性（约30%）：每个槽位都要思考：这里能体现什么独创性？
- 技术实现难度与完整性（约30%）：技术描述要有深度，展示技术选型的合理性
- 应用价值（约20%）：解决什么实际问题，有何推广价值
- 报告质量（约20%）：逻辑清晰，术语规范

## 各类槽位专项写作规范

**问题背景/来源类槽位**：
- 先描述宏观背景（行业痛点/社会需求），再聚焦到具体问题
- 用具体数据或现象支撑问题的严重性/重要性
- 结尾引出本作品解决这一问题的思路

**现有解决方案/相关工作类槽位**：
- 结构：「现有方案 → 优点 → 缺陷 → 本作品改进方向」
- 至少分析 2-3 类不同的现有方案，多维度横向对比（准确率/速度/成本/可用性等）
- 语气客观，对现有工作要有公正评价，不能一味批评

**技术方案类槽位**：
- 先给出整体架构概述，再分模块介绍（总-分结构）
- 原创工作详述，非原创工作简述并注明来源
- 技术选型要说明「为什么选这个而不是别的」

**系统实现类槽位**：
- 描述开发环境、工具链、关键实现细节
- 重点记录遇到的技术难点和解决方案（体现工作量和能力）

**测试分析类槽位**：
- 必须包含：测试数据集说明 + 测试指标定义 + 定量结果 + 与基线对比
- 所有"准确率高""效果好"等宣称必须有具体数字支撑
- 若证据中已有数据，直接引用；若无数据，写「[需补充实验数据]」，不得编造

**创新点类槽位**：
- 列举 1-3 个具体、可验证的创新点
- 每个创新点格式：「创新点名称 + 技术实现方式 + 与现有方法的区别 + 带来的改进」
- 避免空泛表述，需具体到算法/架构/数据处理层面

## 通用写作规范
- 语言：客观正式，无口语化，无"我们认为""非常好"等主观表达
- 段落：主题句 → 证据/数据 → 解释推论；每段 150-400 字
- 术语：与证据包保持一致；首次出现非通用缩写给出全称

严格输出 JSON（禁止额外说明文字）:
{
  "slots": [
    {
      "slot_id": "...",
      "draft_text": "（150-400字完整段落，可直接用于报告）",
      "source_refs": ["project_1"],
      "confidence": 0.0,
      "risk_notes": ["潜在风险（如：缺乏测试数据、证据不足等）"]
    }
  ]
}"""

AGENT_B_PROMPT = """你是 Agent B（Evidence-Skeptic Reviewer），专门对标中国大学生计算机设计大赛评审标准，复核 Agent A 的报告草案。

## 复核清单（逐项检查）

**通用检查**：
- 每个论断是否有证据直接支持（是否过度推断）
- 描述是否客观正式（有无口语化/主观化表达）
- 术语是否前后一致

**创新性检查**（最重要）：
- 创新点是否具体可验证（而非空泛声称）
- 是否清楚说明了与现有方法的区别
- 若有数据对比，数据是否真实来自证据

**技术深度检查**：
- 技术方案是否有「为什么这样选」的论证
- 关键算法/模型是否有原理级的解释（而非只说"使用了XXX"）

**测试分析检查**（高风险区）：
- 宣称的性能数字是否在证据中存在（严防幻觉数据！）
- 若无实验数据，是否已标注「[需补充实验数据]」
- 对比基线是否公平合理

## 输出要求
- 若发现幻觉数据（编造的数字/引用），必须在 disagreements 中明确指出并在 revised_text 中删除或替换为「[需补充实验数据]」
- revised_text 必须是可直接写入报告的完整段落（150-400字）

严格输出 JSON（禁止额外说明文字）：
{
  "slots": [
    {
      "slot_id": "...",
      "review_text": "复核意见（2-4句，指出具体问题）",
      "revised_text": "（修订后的完整段落，150-400字）",
      "source_refs": ["project_1"],
      "confidence": 0.0,
      "disagreements": ["具体分歧点，如：A编造了准确率数据XX%，证据中不存在"]
    }
  ]
}"""

JUDGE_PROMPT = """你是仲裁 Agent（Judge），对标中国大学生计算机设计大赛评审标准做最终裁决。

## 裁决原则
1. **反幻觉优先**：若 A 或 B 中存在编造的数据/事实，必须采用不含该幻觉的版本，或删除相关内容
2. **评审得分最大化**：在保证真实性前提下，选择能更好体现「创新性+技术深度+应用价值」的版本
3. **可用性要求**：final_text 必须是可直接写入报告的完整段落，无需人工大幅修改
4. 若 A/B 各有优劣，取长补短，综合改写（推荐策略，能得到最高质量输出）
5. 证据确实不足时，保留现有内容但标注「[需补充：xxx]」，设置 needs_review=true

## 质量检查（裁决前必做）
- final_text 中是否含有无法从证据包验证的数字/结论 → 若有，删除或标注
- 是否有具体的创新点表述 → 若无，从 A/B 中提炼
- 是否满足 150-400 字的段落长度要求

严格输出 JSON（禁止额外说明文字）：
{
  "slots": [
    {
      "slot_id": "...",
      "final_text": "（完整可用段落，150-400字，无幻觉数据）",
      "final_confidence": 0.0,
      "source_refs": ["project_1"],
      "why": "裁决理由（说明选了哪个版本/如何综合，以及对评审维度的判断）",
      "needs_review": false
    }
  ]
}"""


# ──────────────────────── Helpers ────────────────────────

def _llm(state: ReportState, temperature: float = 0.2) -> ChatOpenAI:
    api_key = state.get("reasoning_api_key")
    base_url = state.get("reasoning_url") or state.get("reasoning_base_url")
    if not api_key or not base_url:
        raise ValueError("missing reasoning_api_key / reasoning_base_url in state")
    model = state.get("reasoning_model_name") or state.get("llm_model_name") or "deepseek-chat"
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=temperature)


def _safe_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, re.IGNORECASE)
    if m:
        return json.loads(m.group(1))
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e > s:
        return json.loads(raw[s:e + 1])
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
    return {s["slot_id"]: s for s in slots if "slot_id" in s}


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


def _fallback_text(slot: Dict[str, Any], a_slot: Dict[str, Any], b_slot: Dict[str, Any]) -> str:
    title = str(slot.get("title") or slot.get("slot_id") or "该部分").strip()
    text = str(a_slot.get("draft_text") or b_slot.get("revised_text") or "").strip()
    if text:
        return text
    return (
        f"围绕{title}，现有证据可支持的结论主要包括研究对象、方法流程与阶段性结果。"
        "由于当前证据包中的可量化数据与对比实验细节有限，相关效果指标仅能进行保守表述。"
        "因此，本节先给出基于已检索材料的初步结论，后续需补充实验统计口径、关键参数与对照结果后再完成定稿。"
    )


def _with_highlight(text: str, points: List[str]) -> str:
    content = (text or "").strip()
    if not content:
        content = "本节内容基于当前证据形成初稿，后续需结合补充材料进行核实与完善。"
    if points and "【待核实:" not in content:
        content = f"{content}【待核实: {points[0]}】"
    return content


def _normalize_judge_slots(
    slots: List[Dict[str, Any]],
    slot_defs: List[Dict[str, Any]],
    a_map: Dict[str, Dict[str, Any]],
    b_map: Dict[str, Dict[str, Any]],
    dispute_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    in_map = _slot_map(slots)
    normalized: List[Dict[str, Any]] = []

    for slot in slot_defs:
        sid = str(slot.get("slot_id") or "").strip()
        if not sid:
            continue

        src = in_map.get(sid, {})
        a_slot = a_map.get(sid, {})
        b_slot = b_map.get(sid, {})

        conf = _f(src.get("final_confidence"), _f(a_slot.get("confidence"), 0.5))
        points = _slist(src.get("uncertainty_points"))
        human_needed = _slist(src.get("human_needed"))
        dispute_reasons = dispute_map.get(sid, [])

        if not points and dispute_reasons:
            points.append(f"A/B存在分歧：{', '.join(dispute_reasons)}")
        if conf < 0.6 and not points:
            points.append("关键论断对应的定量证据不足或统计口径不完整")

        needs_review = bool(src.get("needs_review")) or conf < 0.6 or bool(points)
        level = str(src.get("uncertainty_level") or "").strip().lower()
        if level not in {"low", "medium", "high"}:
            level = "high" if (conf < 0.6 or needs_review or len(points) >= 2) else ("medium" if (conf < 0.8 or points) else "low")

        if not human_needed and (needs_review or level in {"medium", "high"}):
            human_needed = [
                "补充可复现的实验设置、样本规模与统计口径说明",
                "补充关键性能指标与对照基线结果",
            ]

        final_text = _fallback_text(slot, a_slot, b_slot)
        final_text = str(src.get("final_text") or final_text).strip()
        final_text = _with_highlight(final_text, points)

        refs = _slist(src.get("source_refs"))
        if not refs:
            refs = _slist(a_slot.get("source_refs")) or _slist(b_slot.get("source_refs"))

        why = str(src.get("why") or "基于证据完整性与表达稳健性进行裁决。")

        normalized.append(
            {
                "slot_id": sid,
                "final_text": final_text,
                "final_confidence": conf,
                "source_refs": refs,
                "why": why,
                "needs_review": needs_review,
                "uncertainty_level": level,
                "uncertainty_points": points,
                "human_needed": human_needed,
            }
        )

    return normalized


def _build_evidence(
    slot: Dict[str, Any],
    raw_docs: List[Document],
    prev_docs: List[Document],
    seed_terms: Dict[str, Any],
) -> Dict[str, Any]:
    terms = re.findall(
        r"[\u4e00-\u9fff]{2,}|[a-zA-Z][a-zA-Z0-9\-]{2,}",
        f"{slot.get('title', '')} {slot.get('description', '')} "
        f"{' '.join(map(str, (seed_terms.get('keywords') or [])[:6]))} "
        f"{' '.join(map(str, (seed_terms.get('domains') or [])[:3]))}".lower(),
    )
    terms = list(dict.fromkeys(terms))[:14]

    def score_docs(docs: List[Document], max_n: int, max_len: int, prefix: str) -> List[Dict[str, Any]]:
        scored = []
        for doc in docs:
            cnt = sum(1 for t in terms if t and t in (doc.page_content or "").lower())
            if cnt > 0:
                scored.append((cnt, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": f"{prefix}_{i}",
             "content": doc.page_content[:max_len],
             "metadata": dict(doc.metadata or {})}
            for i, (_, doc) in enumerate(scored[:max_n], 1)
        ]

    return {
        "slot_id": slot.get("slot_id", ""),
        "slot_title": slot.get("title", ""),
        "slot_description": str(slot.get("description", "")),
        "terms": terms,
        "project_evidence": score_docs(raw_docs, 4, 700, "proj"),
        "external_evidence": score_docs(prev_docs, 4, 500, "ext"),
    }


# ──────────────────────── Main node ────────────────────────

def analyze_slots_with_crosscheck(state: ReportState) -> Dict[str, Any]:
    """单轮双Agent交叉验证，直接输出最终结果，无需人工介入。"""

    raw_documents: List[Document] = state.get("raw_documents") or []
    template_slots: Dict[str, Any] = state.get("template_slots") or {}
    previous_work_docs: List[Document] = state.get("previous_work_docs") or []
    seed_terms: Dict[str, Any] = state.get("seed_terms") or {}

    if not raw_documents:
        return {"errors": {"analyze_slots_with_crosscheck": "missing raw_documents"}}
    if not template_slots:
        return {"errors": {"analyze_slots_with_crosscheck": "missing template_slots"}}

    slots = _slots(template_slots)
    if not slots:
        return {"errors": {"analyze_slots_with_crosscheck": "template_slots has no valid slots"}}

    run_id = state.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(state.get("intermediate_dir") or "artifacts/intermediate") / "node4" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[node4] start single-round crosscheck, slots={len(slots)}", flush=True)

    # ── 1. 构建所有槽位的证据包 ──
    slot_evidence: Dict[str, Any] = {
        slot["slot_id"]: _build_evidence(slot, raw_documents, previous_work_docs, seed_terms)
        for slot in slots
    }

    payload = {
        "slots": slots,
        "slot_evidence_map": slot_evidence,
        "constraints": {
            "prefer_project_paper": True,
            "external_is_supporting": True,
            "language": "zh",
            "style": "competition_report",
        },
    }

    # ── 2. 并发调用 Agent A 和 Agent B ──
    try:
        llm_a = _llm(state, temperature=0.2)
        llm_b = _llm(state, temperature=0.15)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_invoke, llm_a, AGENT_A_PROMPT, payload)
            fut_b = pool.submit(_invoke, llm_b, AGENT_B_PROMPT, payload)
            a_out = fut_a.result()
            b_out = fut_b.result()

        print(f"[node4] A/B done, A_slots={len(a_out.get('slots', []))}, B_slots={len(b_out.get('slots', []))}", flush=True)
    except Exception as exc:
        _write_json(out_dir / "error.json", {"stage": "agent_ab", "error": str(exc)})
        return {"run_id": run_id, "errors": {"analyze_slots_with_crosscheck": f"agent A/B failed: {exc}"}}

    # ── 3. 计算分歧列表 ──
    a_map = _slot_map(a_out.get("slots", []))
    b_map = _slot_map(b_out.get("slots", []))
    disputes: List[Dict[str, Any]] = []
    for slot in slots:
        sid = slot["slot_id"]
        a, b = a_map.get(sid, {}), b_map.get(sid, {})
        reasons = []
        a_conf = float(a.get("confidence", 0.0))
        b_conf = float(b.get("confidence", 0.0))
        if abs(a_conf - b_conf) >= 0.25:
            reasons.append("confidence_gap")
        a_refs = set(a.get("source_refs", []))
        b_refs = set(b.get("source_refs", []))
        if a_refs and b_refs and a_refs.isdisjoint(b_refs):
            reasons.append("evidence_ref_disjoint")
        if b.get("disagreements"):
            reasons.append("review_disagreement")
        if reasons:
            disputes.append({"slot_id": sid, "reasons": reasons,
                             "a_confidence": a_conf, "b_confidence": b_conf})

    print(f"[node4] disputes={len(disputes)}", flush=True)
    dispute_map = {str(d.get("slot_id") or ""): d.get("reasons", []) for d in disputes}

    # ── 4. Judge 裁决 ──
    try:
        llm_j = _llm(state, temperature=0.1)
        judge_payload = {**payload, "agent_a_output": a_out, "agent_b_output": b_out, "crosscheck_disputes": disputes}
        judge_out = _invoke(llm_j, JUDGE_PROMPT, judge_payload)
        print(f"[node4] judge done, judge_slots={len(judge_out.get('slots', []))}", flush=True)
    except Exception as exc:
        _write_json(out_dir / "error.json", {"stage": "judge", "error": str(exc)})
        # Judge失败时回退到Agent A结果
        print(f"[node4] judge failed ({exc}), falling back to Agent A output", flush=True)
        judge_out = {
            "slots": [
                {
                    "slot_id": s["slot_id"],
                    "final_text": _with_highlight(
                        _fallback_text(
                            s,
                            a_map.get(s["slot_id"], {}),
                            b_map.get(s["slot_id"], {}),
                        ),
                        ["Judge调用失败，当前内容仅基于Agent A/B草案自动回退生成"],
                    ),
                    "final_confidence": _f(a_map.get(s["slot_id"], {}).get("confidence"), 0.5),
                    "source_refs": a_map.get(s["slot_id"], {}).get("source_refs", []),
                    "why": "Judge调用失败，回退到Agent A草案",
                    "needs_review": True,
                    "uncertainty_level": "high",
                    "uncertainty_points": ["Judge裁决阶段失败，最终文本由回退策略生成"],
                    "human_needed": [
                        "人工确认本槽位关键结论是否与原论文一致",
                        "人工补充缺失的实验数据或证据引用",
                    ],
                }
                for s in slots
            ]
        }

    # ── 5. 整理最终输出 ──
    judge_slots = _normalize_judge_slots(
        slots=judge_out.get("slots", []),
        slot_defs=slots,
        a_map=a_map,
        b_map=b_map,
        dispute_map=dispute_map,
    )
    judge_out = {"slots": judge_slots}

    analysis_notes = {
        "filled_slots": {
            s["slot_id"]: s.get("final_text", "")
            for s in judge_slots if s.get("slot_id") and s.get("final_text")
        },
        "slot_confidence": {
            s["slot_id"]: _f(s.get("final_confidence"), 0.0)
            for s in judge_slots if s.get("slot_id")
        },
        "slot_sources": {
            s["slot_id"]: s.get("source_refs", [])
            for s in judge_slots if s.get("slot_id")
        },
        "review_flags": [s["slot_id"] for s in judge_slots if s.get("needs_review")],
        "missing_info_slots": [
            s["slot_id"] for s in judge_slots
            if _f(s.get("final_confidence"), 0.0) < 0.6
        ],
        "uncertainty_map": {
            s["slot_id"]: {
                "level": s.get("uncertainty_level", "low"),
                "points": s.get("uncertainty_points", []),
                "human_needed": s.get("human_needed", []),
            }
            for s in judge_slots if s.get("slot_id")
        },
    }

    print(f"[node4] filled_slots={len(analysis_notes['filled_slots'])}, "
          f"review_flags={len(analysis_notes['review_flags'])}", flush=True)

    # ── 6. 落盘中间结果 ──
    _write_json(out_dir / "slot_evidence_map.json", slot_evidence)
    _write_json(out_dir / "agent_a_output.json", a_out)
    _write_json(out_dir / "agent_b_output.json", b_out)
    _write_json(out_dir / "crosscheck_disputes.json", disputes)
    _write_json(out_dir / "agent_judge_output.json", judge_out)
    _write_json(out_dir / "analysis_notes.json", analysis_notes)

    return {
        "run_id": run_id,
        "slot_evidence_map": slot_evidence,
        "agent_a_output": a_out,
        "agent_b_output": b_out,
        "crosscheck_disputes": disputes,
        "agent_judge_output": judge_out,
        "analysis_notes": analysis_notes,
        # 清除多轮协商相关状态字段，避免 flow.py 路由误判
        "node4_current_round": 1,
        "node4_max_rounds": 1,
        "node4_discussion_rounds": [{"round": 1, "disputes_count": len(disputes)}],
        "node4_unresolved_slots": analysis_notes["review_flags"],
        "node4_feedback_mode": "none",
        "node4_feedback_pending": False,
        "node4_feedback_request_file": "",
        "node4_human_feedback": {},
    }
