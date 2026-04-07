from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


try:
    # 优先导入项目内的 state.py
    from state import ReportState
except Exception:  # noqa: BLE001
    # 直接运行 nodes/preprocess_paper.py 时，先把项目根目录加入搜索路径
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from state import ReportState


def _detect_language(text: str) -> str:
    """非常简单的中英检测：看是否包含一定比例的中文字符.

    不依赖额外第三方库，够当前任务使用。如果以后需要更准，可以换成 langdetect。
    """

    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if not chinese_chars:
        return "en"
    ratio = len(chinese_chars) / max(len(text), 1)
    if ratio > 0.3:
        return "zh"
    return "mixed"


def _choose_loader(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".pdf"}:
        return PyPDFLoader(str(path))
    if suffix in {".docx"}:
        return Docx2txtLoader(str(path))
    if suffix in {".md", ".markdown"}:
        return UnstructuredMarkdownLoader(str(path))
    # 其他情况按纯文本处理
    return TextLoader(str(path), encoding="utf-8")


def _clean_text(text: str) -> str:
    # 合并多余空白
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _add_basic_metadata(chunks: List[Document], source_path: Path) -> List[Document]:
    for idx, doc in enumerate(chunks):
        metadata = dict(doc.metadata) if doc.metadata else {}
        metadata.setdefault("source", str(source_path.name))
        metadata["chunk_index"] = idx

        text = doc.page_content
        lang = _detect_language(text)
        metadata.setdefault("language", lang)

        lower = text.lower()
        if "参考文献" in text or "references" in lower or "bibliography" in lower:
            metadata.setdefault("section", "references")

        doc.metadata = metadata
    return chunks


def _json_default(value: object) -> str:
    return str(value)


def _persist_node1_outputs(
    intermediate_dir: str,
    run_id: str,
    doc_summary: Dict[str, object],
    raw_documents: List[Document],
) -> None:
    node1_dir = Path(intermediate_dir) / "node1" / run_id
    node1_dir.mkdir(parents=True, exist_ok=True)

    preview = []
    for idx, doc in enumerate(raw_documents[:5]):
        preview.append(
            {
                "index": idx,
                "metadata": dict(doc.metadata or {}),
                "content_preview": doc.page_content[:300],
            }
        )

    (node1_dir / "doc_summary.json").write_text(
        json.dumps(doc_summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (node1_dir / "raw_documents_preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def preprocess_project_paper(state: ReportState) -> Dict[str, object]:
    """节点1：读取并预处理原项目论文.

    输入：state["source_file"]
    输出：更新 raw_documents 与 doc_summary
    """

    source = state.get("source_file") or "project_doc.pdf"
    path = Path(source)
    run_id = state.get("run_id") or _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    intermediate_dir = str(state.get("intermediate_dir") or "artifacts/intermediate")

    print(f"[node1] start preprocess_project_paper, source={path}, run_id={run_id}", flush=True)

    if not path.exists():
        errors = {"preprocess_project_paper": f"source_file not found: {path}"}
        print(f"[node1] failed: {errors['preprocess_project_paper']}", flush=True)
        return {"errors": errors}

    try:
        loader = _choose_loader(path)
        documents: List[Document] = loader.load()
    except Exception as exc:  # noqa: BLE001
        errors = {"preprocess_project_paper": f"failed to load document: {exc}"}
        print(f"[node1] failed: {errors['preprocess_project_paper']}", flush=True)
        return {"errors": errors}

    for doc in documents:
        doc.page_content = _clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ".",
            "!",
            "?",
            " ",
        ],
    )
    chunks = splitter.split_documents(documents)
    chunks = _add_basic_metadata(chunks, path)

    pages = {c.metadata.get("page") for c in chunks if c.metadata.get("page") is not None}
    total_pages = len(pages) if pages else None

    if chunks:
        langs = [c.metadata.get("language", "") for c in chunks]
        zh = sum(1 for l in langs if l == "zh")
        en = sum(1 for l in langs if l == "en")
        if zh >= en and zh > 0:
            main_lang = "zh"
        elif en > 0:
            main_lang = "en"
        else:
            main_lang = "mixed"
    else:
        main_lang = ""

    sections = []
    for c in chunks:
        section = c.metadata.get("section")
        if section and section not in sections:
            sections.append(section)

    doc_summary = {
        "total_pages": total_pages,
        "num_chunks": len(chunks),
        "language": main_lang,
        "sections_found": sections,
        "preprocessed_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }

    _persist_node1_outputs(
        intermediate_dir=intermediate_dir,
        run_id=run_id,
        doc_summary=doc_summary,
        raw_documents=chunks,
    )

    node1_dir = Path(intermediate_dir) / "node1" / run_id
    print(
        (
            f"[node1] done: chunks={len(chunks)}, pages={total_pages}, language={main_lang}, "
            f"artifacts={node1_dir}"
        ),
        flush=True,
    )

    return {"run_id": run_id, "raw_documents": chunks, "doc_summary": doc_summary}

if __name__ == '__main__':
    # 方便调试，直接运行这个脚本会处理当前目录下的 project_doc.pdf
    state = ReportState(source_file="project_doc.pdf")
    result = preprocess_project_paper(state)
    if "errors" in result:
        print("Errors:", result["errors"])
    else:
        print("Document Summary:", result["doc_summary"])
        print(f"First chunk content:\n{result['raw_documents'][0].page_content[:500]}")