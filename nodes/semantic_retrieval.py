"""语义证据检索模块

基于向量嵌入 + FAISS 的语义搜索，替代原有的关键词匹配。
支持优雅降级：若 embedding API 不可用，自动回退到空结果（保留关键词匹配作为兜底）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:  # noqa: BLE001
    FAISS = None
    OpenAIEmbeddings = None

# 模块级缓存：run_id -> SemanticRetriever
_retriever_cache: Dict[str, "SemanticRetriever"] = {}


class SemanticRetriever:
    """基于 FAISS + OpenAI Embeddings 的语义检索器。"""

    def __init__(
        self,
        documents: List[Document],
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ):
        self.documents = documents
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.vectorstore: Optional[Any] = None
        self._build_index()

    def _build_index(self) -> None:
        if FAISS is None or OpenAIEmbeddings is None:
            return
        try:
            embeddings = OpenAIEmbeddings(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self.vectorstore = FAISS.from_documents(self.documents, embeddings)
        except Exception as exc:  # noqa: BLE001
            print(f"[semantic_retrieval] embedding index build failed: {exc}", flush=True)
            self.vectorstore = None

    def search(self, query: str, k: int = 6) -> List[Document]:
        if self.vectorstore is None:
            return []
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as exc:  # noqa: BLE001
            print(f"[semantic_retrieval] search failed: {exc}", flush=True)
            return []


def build_retriever(
    run_id: str,
    documents: List[Document],
    api_key: str,
    base_url: Optional[str] = None,
    model: str = "text-embedding-3-small",
) -> Optional[SemanticRetriever]:
    """为指定 run_id 构建语义检索器（带缓存）。"""
    if run_id in _retriever_cache:
        return _retriever_cache[run_id]
    try:
        retriever = SemanticRetriever(documents, api_key, base_url, model)
        _retriever_cache[run_id] = retriever
        return retriever
    except Exception as exc:  # noqa: BLE001
        print(f"[semantic_retrieval] build_retriever failed: {exc}", flush=True)
        return None


def search(run_id: str, query: str, k: int = 6) -> List[Document]:
    """基于 run_id 检索语义相关文档。"""
    retriever = _retriever_cache.get(run_id)
    if retriever is None:
        return []
    return retriever.search(query, k)


def hybrid_evidence(
    run_id: str,
    slot: Dict[str, Any],
    raw_docs: List[Document],
    seed_terms: Dict[str, Any],
    semantic_k: int = 6,
    keyword_k: int = 4,
    max_len: int = 700,
) -> List[Dict[str, Any]]:
    """混合检索：语义搜索 + 关键词匹配，去重后返回证据列表。

    返回格式与原有的 _build_evidence 中的 project_evidence 一致。
    """
    # 1. 语义检索
    semantic_docs = search(run_id, f"{slot.get('title', '')} {slot.get('description', '')}", k=semantic_k)
    semantic_ids = {id(d) for d in semantic_docs}

    # 2. 关键词匹配（保留原有逻辑作为补充）
    terms = re.findall(
        r"[\u4e00-\u9fff]{2,}|[a-zA-Z][a-zA-Z0-9\-]{2,}",
        f"{slot.get('title', '')} {slot.get('description', '')} "
        f"{' '.join(map(str, (seed_terms.get('keywords') or [])[:6]))} "
        f"{' '.join(map(str, (seed_terms.get('domains') or [])[:3]))}".lower(),
    )
    terms.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", str(slot.get("slot_id") or "").replace("_", " ").lower()))
    terms = list(dict.fromkeys(terms))[:14]

    scored = []
    for doc in raw_docs:
        if id(doc) in semantic_ids:
            continue  # 语义已召回的跳过
        cnt = sum(1 for t in terms if t and t in (doc.page_content or "").lower())
        if cnt > 0:
            scored.append((cnt, doc))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 3. 合并
    results: List[Dict[str, Any]] = []
    for i, doc in enumerate(semantic_docs, 1):
        results.append({
            "id": f"sem_{i}",
            "content": (doc.page_content or "")[:max_len],
            "metadata": {**dict(doc.metadata or {}), "source": "semantic"},
        })
    for i, (_, doc) in enumerate(scored[:keyword_k], 1):
        results.append({
            "id": f"kw_{i}",
            "content": (doc.page_content or "")[:max_len],
            "metadata": {**dict(doc.metadata or {}), "source": "keyword"},
        })

    # 4. 去重（按内容前100字符）
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for item in results:
        key = item["content"][:100]
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped[:semantic_k + keyword_k]


import re
