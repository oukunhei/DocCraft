from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    from docx import Document  # type: ignore
    from docx.enum.text import WD_COLOR_INDEX  # type: ignore
except Exception:  # noqa: BLE001
    Document = None
    WD_COLOR_INDEX = None

try:
    from state import ReportState
except Exception:  # noqa: BLE001
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from state import ReportState


def _resolve_template_path(template_file: str | None) -> Path:
    project_root = Path(__file__).resolve().parents[1]

    if template_file:
        p = Path(template_file)
        if p.exists():
            return p
        rel = project_root / template_file
        if rel.exists():
            return rel

    for name in ("template.docx", "templete.docx"):
        p = project_root / name
        if p.exists():
            return p

    return project_root / (template_file or "template.docx")


def _collect_slot_defs(template_slots: Dict[str, Any]) -> List[Dict[str, Any]]:
    core = template_slots.get("core_slots") or []
    other = template_slots.get("other_technical_slots") or []
    return [s for s in core + other if s.get("slot_id")]


def _fallback_filled_slots(state: ReportState) -> Dict[str, str]:
    # 1) 优先使用节点4标准输出
    analysis_notes = state.get("analysis_notes") or {}
    filled = analysis_notes.get("filled_slots") or {}
    if filled:
        out = {str(k): str(v) for k, v in filled.items() if str(v).strip()}
        if out:
            out["__source__"] = "analysis_notes"
            return out

    # 2) 回退到仲裁Agent原始输出
    judge = state.get("agent_judge_output") or {}
    judge_slots = judge.get("slots") or []
    out: Dict[str, str] = {}
    for item in judge_slots:
        sid = str(item.get("slot_id") or "").strip()
        text = str(item.get("final_text") or "").strip()
        if sid and text:
            out[sid] = text
    if out:
        out["__source__"] = "agent_judge_output"
        return out

    # 3) 最后回退到Agent A草稿
    agent_a = state.get("agent_a_output") or {}
    a_slots = agent_a.get("slots") or []
    for item in a_slots:
        sid = str(item.get("slot_id") or "").strip()
        text = str(item.get("draft_text") or "").strip()
        if sid and text:
            out[sid] = text
    if out:
        out["__source__"] = "agent_a_output"
    return out


def _replace_placeholders(doc: Any, filled_slots: Dict[str, str]) -> int:
    replaced = 0
    patterns = []
    for slot_id in filled_slots:
        patterns.append("{{" + slot_id + "}}")
        patterns.append("<<" + slot_id + ">>")
        patterns.append("【" + slot_id + "】")

    def replace_text(text: str) -> str:
        nonlocal replaced
        out = text
        for slot_id, value in filled_slots.items():
            for p in ("{{" + slot_id + "}}", "<<" + slot_id + ">>", "【" + slot_id + "】"):
                if p in out:
                    out = out.replace(p, value)
                    replaced += 1
        return out

    for para in doc.paragraphs:
        if para.text:
            new_text = replace_text(para.text)
            if new_text != para.text:
                para.text = new_text

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    if para.text:
                        new_text = replace_text(para.text)
                        if new_text != para.text:
                            para.text = new_text

    return replaced


def _normalize_text(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _find_paragraph_indices_by_title(doc: Any, title: str) -> List[int]:
    target = _normalize_text(title)
    if not target:
        return []

    indices: List[int] = []
    for idx, para in enumerate(doc.paragraphs):
        text = (para.text or "").strip()
        if not text:
            continue
        norm = _normalize_text(text)
        if not norm:
            continue
        # 标题匹配：相等/包含都算命中
        if norm == target or target in norm or norm in target:
            indices.append(idx)
    return indices


def _insert_after_paragraph(doc: Any, paragraph_index: int, text: str) -> Any:
    # python-docx 没有直接的 insert_after，这里用 XML 层插入并返回新段落对象
    paragraph = doc.paragraphs[paragraph_index]
    from docx.oxml import OxmlElement  # type: ignore
    from docx.text.paragraph import Paragraph  # type: ignore

    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)  # type: ignore[attr-defined]

    new_para = Paragraph(new_p, paragraph._parent)
    new_para.text = text
    return new_para


def _set_paragraph_highlight(paragraph: Any) -> None:
    if WD_COLOR_INDEX is None:
        return
    if not paragraph.runs:
        run = paragraph.add_run(paragraph.text)
        paragraph.text = ""
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return
    for run in paragraph.runs:
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def _write_by_headings(
    doc: Any,
    slot_defs: List[Dict[str, Any]],
    filled_slots: Dict[str, str],
    confidence_map: Dict[str, Any],
    source_map: Dict[str, Any],
    review_flags: set[str],
    uncertainty_map: Dict[str, Any],
) -> Dict[str, Any]:
    inserted = 0
    unresolved: List[str] = []

    for slot in slot_defs:
        slot_id = str(slot.get("slot_id") or "").strip()
        title = str(slot.get("title") or slot_id).strip()
        content = str(filled_slots.get(slot_id, "")).strip()
        if not slot_id or not content:
            continue

        hit_indices = _find_paragraph_indices_by_title(doc, title)
        if not hit_indices:
            unresolved.append(slot_id)
            continue

        # 在首个命中标题后插入内容和元信息
        idx = hit_indices[0]
        p_content = _insert_after_paragraph(doc, idx, content)
        inserted += 1

        conf = confidence_map.get(slot_id)
        refs = source_map.get(slot_id, [])
        flag = "是" if slot_id in review_flags else "否"
        uncertainty = uncertainty_map.get(slot_id) or {}
        u_level = str(uncertainty.get("level") or "low")
        u_points = uncertainty.get("points") or []
        u_human = uncertainty.get("human_needed") or []
        first_point = str(u_points[0]) if u_points else "无"
        meta = (
            f"[AI草稿] 置信度: {conf} | 需人工复核: {flag} | 引用: {', '.join(refs) if refs else '无'}"
            f" | 不确定等级: {u_level}"
        )
        p_meta = _insert_after_paragraph(doc, idx + 1, meta)
        if u_points or u_human:
            need_line = (
                f"[不确定点] {first_point} | "
                f"[需人工补充] {'; '.join(str(x) for x in u_human) if u_human else '无'}"
            )
            p_need = _insert_after_paragraph(doc, idx + 2, need_line)
        else:
            p_need = None

        # 对低置信度/人工复核项高亮
        low_conf = False
        try:
            low_conf = float(conf or 0.0) < 0.6
        except Exception:  # noqa: BLE001
            low_conf = True
        if low_conf or slot_id in review_flags or u_level in {"medium", "high"}:
            _set_paragraph_highlight(p_content)
            _set_paragraph_highlight(p_meta)
            if p_need is not None:
                _set_paragraph_highlight(p_need)

    return {"inserted_by_heading": inserted, "unresolved_slots": unresolved}


def _build_review_checklist_docx(
    checklist_path: Path,
    run_id: str,
    slot_defs: List[Dict[str, Any]],
    filled_slots: Dict[str, str],
    confidence_map: Dict[str, Any],
    source_map: Dict[str, Any],
    review_flags: set[str],
    uncertainty_map: Dict[str, Any],
) -> None:
    doc = Document()
    doc.add_heading("报告复核清单", level=1)
    doc.add_paragraph(f"Run ID: {run_id}")

    if not review_flags:
        doc.add_paragraph("当前没有被标记为人工复核的槽位。")
        doc.save(str(checklist_path))
        return

    slot_map = {str(s.get("slot_id")): s for s in slot_defs}
    for slot_id in review_flags:
        slot = slot_map.get(slot_id, {})
        title = str(slot.get("title") or slot_id)
        content = str(filled_slots.get(slot_id, "")).strip()
        conf = confidence_map.get(slot_id)
        refs = source_map.get(slot_id, [])
        uncertainty = uncertainty_map.get(slot_id) or {}
        u_level = str(uncertainty.get("level") or "low")
        u_points = [str(x) for x in (uncertainty.get("points") or []) if str(x).strip()]
        u_human = [str(x) for x in (uncertainty.get("human_needed") or []) if str(x).strip()]

        doc.add_heading(f"{title} ({slot_id})", level=2)
        doc.add_paragraph(f"置信度: {conf}")
        doc.add_paragraph(f"引用: {', '.join(refs) if refs else '无'}")
        doc.add_paragraph(f"不确定等级: {u_level}")
        if u_points:
            doc.add_paragraph("具体不确定点:")
            for item in u_points:
                doc.add_paragraph(item, style="List Bullet")
        if u_human:
            doc.add_paragraph("需要人工补充:")
            for item in u_human:
                doc.add_paragraph(item, style="List Bullet")
        doc.add_paragraph("建议检查项:")
        doc.add_paragraph("1. 结论是否可由引用证据直接支持。")
        doc.add_paragraph("2. 是否存在夸大、推断过度或遗漏反例。")
        doc.add_paragraph("3. 表述是否符合参赛报告风格。")
        doc.add_paragraph("当前草稿内容:")
        p = doc.add_paragraph(content if content else "[空]")
        _set_paragraph_highlight(p)

    doc.save(str(checklist_path))


def _append_slots_as_section(
    doc: Any,
    section_title: str,
    slot_defs: List[Dict[str, Any]],
    filled_slots: Dict[str, str],
    confidence_map: Dict[str, Any],
    source_map: Dict[str, Any],
    review_flags: set[str],
    uncertainty_map: Dict[str, Any],
) -> int:
    appended = 0
    doc.add_page_break()
    doc.add_heading(section_title, level=1)

    for slot in slot_defs:
        slot_id = str(slot.get("slot_id"))
        title = str(slot.get("title") or slot_id)
        content = str(filled_slots.get(slot_id, "")).strip()
        if not slot_id or not content:
            continue

        appended += 1
        doc.add_heading(title, level=2)
        doc.add_paragraph(content)

        confidence = confidence_map.get(slot_id)
        refs = source_map.get(slot_id, [])
        flag = "是" if slot_id in review_flags else "否"
        uncertainty = uncertainty_map.get(slot_id) or {}
        u_level = str(uncertainty.get("level") or "low")
        u_points = [str(x) for x in (uncertainty.get("points") or []) if str(x).strip()]
        u_human = [str(x) for x in (uncertainty.get("human_needed") or []) if str(x).strip()]
        note = (
            f"置信度: {confidence} | 需人工复核: {flag} | 引用: {', '.join(refs) if refs else '无'}"
            f" | 不确定等级: {u_level}"
        )
        p_note = doc.add_paragraph(note)
        if u_points or u_human:
            p_detail = doc.add_paragraph(
                f"不确定点: {u_points[0] if u_points else '无'} | "
                f"需人工补充: {'; '.join(u_human) if u_human else '无'}"
            )
        else:
            p_detail = None

        low_conf = False
        try:
            low_conf = float(confidence or 0.0) < 0.6
        except Exception:  # noqa: BLE001
            low_conf = True
        if low_conf or slot_id in review_flags or u_level in {"medium", "high"}:
            _set_paragraph_highlight(p_note)
            if p_detail is not None:
                _set_paragraph_highlight(p_detail)

    return appended


def build_report_docx(state: ReportState) -> Dict[str, Any]:
    """节点5：基于模板创建新 docx 并填入节点4产出的槽位内容。"""

    if Document is None:
        return {
            "errors": {
                "build_report_docx": "missing dependency python-docx. Please install python-docx.",
            }
        }

    template_slots = state.get("template_slots") or {}
    analysis_notes = state.get("analysis_notes") or {}
    filled_slots = _fallback_filled_slots(state)

    if not template_slots:
        return {"errors": {"build_report_docx": "missing template_slots from node2"}}
    slot_source = str(filled_slots.pop("__source__", "none")) if filled_slots else "none"

    if not filled_slots:
        return {
            "errors": {
                "build_report_docx": (
                    "missing filled slots from node4 outputs "
                    "(analysis_notes/agent_judge_output/agent_a_output are empty)"
                )
            }
        }

    template_path = _resolve_template_path(state.get("template_file"))
    if not template_path.exists():
        return {
            "errors": {
                "build_report_docx": f"template_file not found: {template_path}",
            }
        }

    run_id = state.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(state.get("intermediate_dir") or "artifacts/intermediate").parent / "final"
    out_root.mkdir(parents=True, exist_ok=True)
    output_path = out_root / f"report_filled_{run_id}.docx"
    checklist_path = out_root / f"report_review_checklist_{run_id}.docx"

    # 不修改原模板：先复制再写。
    shutil.copy2(template_path, output_path)

    doc = Document(str(output_path))
    replaced_count = _replace_placeholders(doc, filled_slots)

    slot_defs = _collect_slot_defs(template_slots)
    confidence_map = analysis_notes.get("slot_confidence") or {}
    source_map = analysis_notes.get("slot_sources") or {}
    review_flags = set(analysis_notes.get("review_flags") or [])
    uncertainty_map = analysis_notes.get("uncertainty_map") or {}

    # 如果 analysis_notes 为空，尽量从仲裁输出补齐置信度与引用，降低节点耦合脆弱性。
    if not confidence_map or not source_map or not uncertainty_map:
        judge = state.get("agent_judge_output") or {}
        for item in judge.get("slots") or []:
            sid = str(item.get("slot_id") or "").strip()
            if not sid:
                continue
            if sid not in confidence_map:
                confidence_map[sid] = item.get("final_confidence")
            if sid not in source_map:
                source_map[sid] = item.get("source_refs") or []
            if sid not in uncertainty_map:
                uncertainty_map[sid] = {
                    "level": item.get("uncertainty_level") or "low",
                    "points": item.get("uncertainty_points") or [],
                    "human_needed": item.get("human_needed") or [],
                }
            if item.get("human_review_required") or item.get("needs_review"):
                review_flags.add(sid)

    # 第一优先：按章节标题定位写入
    heading_result = _write_by_headings(
        doc,
        slot_defs=slot_defs,
        filled_slots=filled_slots,
        confidence_map=confidence_map,
        source_map=source_map,
        review_flags=review_flags,
        uncertainty_map=uncertainty_map,
    )
    unresolved_ids = set(heading_result.get("unresolved_slots") or [])
    unresolved_slot_defs = [
        s
        for s in slot_defs
        if str(s.get("slot_id") or "") in unresolved_ids and str(filled_slots.get(str(s.get("slot_id") or ""), "")).strip()
    ]

    appended_fallback_count = 0

    # 如果模板里没有占位符，就在文档末尾追加“AI生成草稿”章节。
    if replaced_count == 0 and heading_result["inserted_by_heading"] == 0:
        appended_fallback_count = _append_slots_as_section(
            doc=doc,
            section_title="AI 自动填充草稿（请人工复核）",
            slot_defs=slot_defs,
            filled_slots=filled_slots,
            confidence_map=confidence_map,
            source_map=source_map,
            review_flags=review_flags,
            uncertainty_map=uncertainty_map,
        )
    elif unresolved_slot_defs:
        # 关键改进：部分命中模板标题时，未命中的槽位也必须保留，避免最终报告丢段。
        appended_fallback_count = _append_slots_as_section(
            doc=doc,
            section_title="AI 自动补充草稿（未命中模板标题）",
            slot_defs=unresolved_slot_defs,
            filled_slots=filled_slots,
            confidence_map=confidence_map,
            source_map=source_map,
            review_flags=review_flags,
            uncertainty_map=uncertainty_map,
        )

    doc.save(str(output_path))

    _build_review_checklist_docx(
        checklist_path=checklist_path,
        run_id=run_id,
        slot_defs=slot_defs,
        filled_slots=filled_slots,
        confidence_map=confidence_map,
        source_map=source_map,
        review_flags=review_flags,
        uncertainty_map=uncertainty_map,
    )

    summary = {
        "output_docx": str(output_path),
        "review_checklist_docx": str(checklist_path),
        "template_source": str(template_path),
        "slot_source": slot_source,
        "replaced_placeholders": replaced_count,
        "inserted_by_heading": heading_result["inserted_by_heading"],
        "appended_unresolved_slots": appended_fallback_count,
        "unresolved_slots": heading_result["unresolved_slots"],
        "filled_slots_count": len(filled_slots),
    }

    return {
        "run_id": run_id,
        "final_report_docx": str(output_path),
        "final_review_checklist_docx": str(checklist_path),
        "final_report": f"report generated at {output_path}",
        "report_build_summary": summary,
    }
