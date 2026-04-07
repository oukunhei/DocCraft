from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from state import ReportState
except Exception:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from state import ReportState


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

RELATED_WORK_RULES = [
    r"相关工作", r"文献综述", r"研究现状", r"已有工作",
    r"related\s+work", r"literature\s+review", r"prior\s+work",
]

EXTRACT_TERMS_PROMPT = """你是中国大学生计算机设计大赛（4C）报告的科研检索助手。
给你一篇论文/项目文档片段后，请提取用于外部检索的种子信息。

严格返回 JSON：
{
  "keywords": ["..."],
  "domains": ["..."],
  "problem_statement": "..."
}

要求：
1. keywords 返回 6-12 个高质量术语，优先包含：核心算法名称、任务名称、数据集名称、评估指标名称，尽量包含中英文关键术语。
2. domains 返回 1-3 个领域标签（如"计算机视觉""自然语言处理""物联网""推荐系统"等）。
3. problem_statement 用一句中文描述项目要解决的核心问题，格式：「[领域] + [具体任务] + [技术挑战]」。
4. 仅返回 JSON，不要额外文本。
"""

BUILD_QUERY_PROMPT = """你是中国大学生计算机设计大赛（4C）报告的技术检索 query 规划助手。
基于输入的 seeds 和模板槽位，生成两类搜索 query，专门服务于参赛报告写作：

1) paper_queries：学术论文检索，用于"相关工作综述"和"技术方案"章节
   - 目标：找到与项目方法相关的基线方法、对比方法、先驱工作，以体现创新性
   - 优先检索能帮助对比分析的综述/Survey 类论文

2) news_queries：行业动态/应用检索，用于"应用推广"和"项目背景意义"章节
   - 目标：找到能支撑项目应用价值和社会意义的产业数据、政策文件、落地案例

严格返回 JSON：
{
  "paper_queries": ["..."],
  "news_queries": ["..."]
}

要求：
1. 每类返回 3-5 条 query。
2. paper_queries 必须包含：① 核心方法+survey/review；② 主要竞品/基线方法；③ 核心任务+benchmark。
3. news_queries 必须包含：① 领域应用场景+落地案例；② 相关产业政策/市场规模数据；③ 实际用户需求/痛点。
4. 仅返回 JSON，不要额外文本。
"""


def _env_bool(name: str, default: bool) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on", "y"} if os.getenv(name) is not None else default


def _split_sites(raw: str) -> List[str]:
    sites = [s.strip() for s in re.split(r"[\s,]+", raw) if s.strip()]
    return list(dict.fromkeys(sites))


def _expand_site_queries(query: str, sites: List[str], split: bool) -> List[Tuple[str, str]]:
    if not sites:
        return [(query, "all")]
    if not split:
        return [(f"{query} {' OR '.join(f'site:{s}' for s in sites)}", "all")]
    return [(f"{query} site:{site}", site) for site in sites]


def _should_retry_http(code: int) -> bool:
    return code in {408, 425, 429, 500, 502, 503, 504}


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return _should_retry_http(exc.code)
    text = str(exc).lower()
    return any(sig in text for sig in ["unexpected_eof", "remote end closed", "timed out", "connection reset", "ssl"])


def _build_search_urls(query: str) -> List[str]:
    encoded = urllib.parse.quote(query, safe="")
    endpoint = os.getenv("JINA_SEARCH_URL")
    if endpoint:
        return [endpoint.format(query=encoded) if "{query}" in endpoint else endpoint]
    primary = os.getenv("SEARCH_ENGINE_URL") or "https://www.bing.com/search?q={query}"
    fallback = os.getenv("SEARCH_FALLBACK_ENGINE_URL") or "https://duckduckgo.com/html/?q={query}"
    return [f"https://r.jina.ai/{engine.format(query=encoded)}" for engine in (primary, fallback)]


def _jina_fetch_with_retry(search_urls: List[str], headers: Dict[str, str]) -> str:
    timeout = int(os.getenv("JINA_TIMEOUT_SECONDS", "45"))
    max_attempts = max(1, int(os.getenv("JINA_MAX_ATTEMPTS") or os.getenv("JINA_MAX_RETRIES") or "3"))
    base_delay = float(os.getenv("JINA_RETRY_BASE_DELAY_SECONDS") or "1.0")
    max_delay = float(os.getenv("JINA_RETRY_MAX_DELAY_SECONDS") or "8.0")
    last_err = None
    attempts = 0
    for url in search_urls:
        for attempt in range(1, max_attempts + 1):
            attempts += 1
            try:
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                last_err = RuntimeError(f"Jina HTTPError {e.code}: {e.read().decode('utf-8', errors='replace')}")
            except Exception as e:
                last_err = e
            if attempt >= max_attempts or not _is_transient_error(last_err):
                break
            sleep = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.8)
            time.sleep(sleep)
    raise RuntimeError(f"Jina search failed after {attempts} attempts: {last_err}")


def _normalize_jina_result(payload: Any) -> List[Dict[str, Any]]:
    def coerce(item):
        if not isinstance(item, dict):
            return {"title": "", "url": "", "snippet": "", "published_at": None}
        return {
            "title": (item.get("title") or item.get("name") or "").strip(),
            "url": (item.get("url") or item.get("link") or "").strip(),
            "snippet": (item.get("snippet") or item.get("description") or item.get("content") or "").strip(),
            "published_at": item.get("published_at") or item.get("date") or item.get("published"),
        }
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return [coerce(x) for x in payload["data"]]
        if "results" in payload and isinstance(payload["results"], list):
            return [coerce(x) for x in payload["results"]]
        if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("content"), str):
            markdown = payload["data"]["content"]
            return [{"title": m[0], "url": m[1], "snippet": "", "published_at": None}
                    for m in re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", markdown) if m[1] and not m[1].startswith("javascript:")]
    if isinstance(payload, str):
        return [{"title": m[0], "url": m[1], "snippet": "", "published_at": None}
                for m in re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", payload) if m[1] and not m[1].startswith("javascript:")]
    return []


def _llm_from_state(state: ReportState) -> ChatOpenAI:
    api_key = state.get("tool_api_key")
    base_url = state.get("tool_base_url")
    if not api_key or not base_url:
        raise ValueError("missing tool config in state: tool_api_key/tool_base_url")
    return ChatOpenAI(model=state.get("tool_model_name") or "step-3.5-flash", api_key=api_key, base_url=base_url, temperature=0.2)


def _call_json(llm: ChatOpenAI, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
    resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    text = text.strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fence = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.I)
        if fence:
            return json.loads(fence.group(1))
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end+1])
        raise ValueError(f"unable to parse json: {text[:300]}")


def _jina_search(query: str, category: str, top_k: int = 8) -> List[Dict[str, Any]]:
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("missing JINA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json",
               "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    urls = _build_search_urls(query)
    raw = _jina_fetch_with_retry(urls, headers)
    items = _normalize_jina_result(raw)
    items = [i for i in items if i.get("url") and "jina.ai" not in i.get("url", "")]
    normalized = []
    for idx, item in enumerate(items[:top_k]):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not url and not title:
            continue
        normalized.append({
            "title": title,
            "url": url,
            "snippet": (item.get("snippet") or "").strip(),
            "published_at": item.get("published_at"),
            "rank_raw": idx + 1,
            "source_type": category,
        })
    return normalized


def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_url, seen_title = set(), set()
    deduped = []
    for item in items:
        url = item.get("url", "").strip()
        norm_url = re.sub(r"[?&]utm_[^&=]+=[^&]*", "", url).rstrip("/")
        norm_title = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", (item.get("title") or "").lower())
        if (norm_url and norm_url in seen_url) or (norm_title and norm_title in seen_title):
            continue
        if norm_url:
            seen_url.add(norm_url)
        if norm_title:
            seen_title.add(norm_title)
        item["url"] = norm_url or url
        deduped.append(item)
    return deduped


def _score_item(item: Dict[str, Any], seeds: Dict[str, Any]) -> float:
    text = f"{item.get('title','')} {item.get('snippet','')}".lower()
    keywords = [str(k).lower() for k in seeds.get("keywords", [])]
    domains = [str(d).lower() for d in seeds.get("domains", [])]
    overlap = sum(1 for t in keywords+domains if t and t in text)
    source_bonus = 1.0 if item.get("source_type") == "paper" else 0.6
    freshness = 0.0
    if m := re.search(r"(20\d{2})", str(item.get("published_at") or "")):
        year = int(m.group(1))
        now_year = datetime.now(UTC).year
        if year >= now_year - 2:
            freshness = 0.5
        elif year >= now_year - 4:
            freshness = 0.2
    return overlap + source_bonus + freshness


def _to_documents(items: Iterable[Dict[str, Any]], query: str, provider: str = "jina") -> List[Document]:
    return [Document(page_content=item.get("snippet") or item.get("title") or "",
                     metadata={"source_type": item.get("source_type", "web"), "provider": provider,
                               "title": item.get("title", ""), "url": item.get("url", ""),
                               "published_at": item.get("published_at"), "query": query,
                               "rank_raw": item.get("rank_raw"), "score": item.get("score")})
            for item in items]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def search_related(state: ReportState) -> Dict[str, Any]:
    raw_docs = state.get("raw_documents") or []
    if not raw_docs:
        return {"errors": {"search_related": "missing raw_documents from node1"}}

    run_id = state.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    node3_dir = Path(state.get("intermediate_dir") or "artifacts/intermediate") / "node3" / run_id

    # 检测是否存在相关工作章节
    related_snippets = []
    for doc in raw_docs:
        text = doc.page_content.strip()
        if not text:
            continue
        if any(re.search(pat, text, re.I) for pat in RELATED_WORK_RULES):
            related_snippets.append(text[:1200])
    has_related = bool(related_snippets)
    print(f"[node3] start search_related, has_related_work={has_related}, docs={len(raw_docs)}", flush=True)

    try:
        llm = _llm_from_state(state)
        source = "\n\n".join(related_snippets) if has_related else "\n\n".join(doc.page_content for doc in raw_docs[:40])
        seeds = _call_json(llm, EXTRACT_TERMS_PROMPT, f"请从以下文本提取检索 seeds：\n\n{source[:8000]}")
        seeds = {"keywords": seeds.get("keywords", []), "domains": seeds.get("domains", []), "problem_statement": seeds.get("problem_statement", "")}
        queries = _call_json(llm, BUILD_QUERY_PROMPT, json.dumps({"seeds": seeds, "template_slots": state.get("template_slots") or {}}, ensure_ascii=False))
        paper_queries = [q.strip() for q in queries.get("paper_queries", [])[:int(os.getenv("SEARCH_MAX_QUERIES_PER_TYPE", "3"))] if q.strip()]
        news_queries = [q.strip() for q in queries.get("news_queries", [])[:int(os.getenv("SEARCH_MAX_QUERIES_PER_TYPE", "3"))] if q.strip()]
    except Exception as e:
        _write_json(node3_dir / "error.json", {"error": str(e)})
        return {"errors": {"search_related": f"llm stage failed: {e}"}, "run_id": run_id}

    print(f"[node3] generated queries: paper={len(paper_queries)}, news={len(news_queries)}", flush=True)

    # 配置
    split_site = _env_bool("SEARCH_SPLIT_SITE_QUERIES", True)
    site_fail_thresh = int(os.getenv("SEARCH_SITE_FAIL_THRESHOLD", "2"))
    paper_sites = _split_sites(os.getenv("SEARCH_PAPER_SITES", "arxiv.org semanticscholar.org ieeexplore.ieee.org"))
    news_sites = _split_sites(os.getenv("SEARCH_NEWS_SITES", "techcrunch.com theverge.com wired.com"))
    max_attempts = int(os.getenv("JINA_MAX_ATTEMPTS") or os.getenv("JINA_MAX_RETRIES") or "3")
    min_paper = int(os.getenv("SEARCH_MIN_PAPER_DOCS", "6"))
    min_news = int(os.getenv("SEARCH_MIN_NEWS_DOCS", "4"))

    raw_results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    site_fail_cnt: Dict[str, int] = {}
    jina_available = True  # 标记 Jina 是否可用，首次超时后停止后续搜索

    def do_search(category: str, queries: List[str], sites: List[str]):
        nonlocal raw_results, failures, site_fail_cnt, jina_available
        if not jina_available:
            return
        for q in queries:
            if not jina_available:
                break
            for site_q, site in _expand_site_queries(q, sites, split_site):
                if not jina_available:
                    break
                use_fallback = split_site and site != "all" and site_fail_cnt.get(site, 0) >= site_fail_thresh
                eff_q = q if use_fallback else site_q
                try:
                    raw_results.extend(_jina_search(eff_q, category=category))
                except Exception as e:
                    err_str = str(e).lower()
                    if "timed out" in err_str or "timeout" in err_str or "connection" in err_str:
                        # 网络不可达，停止后续所有 Jina 请求
                        jina_available = False
                        failures.append({"query": eff_q, "error": str(e), "site": site,
                                         "attempt_count": max_attempts, "fallback_used": use_fallback,
                                         "jina_disabled": True})
                        print(f"[node3] Jina unreachable, skipping remaining searches: {e}", flush=True)
                        break
                    if site != "all":
                        site_fail_cnt[site] = site_fail_cnt.get(site, 0) + 1
                    failures.append({"query": eff_q, "error": str(e), "site": site,
                                     "attempt_count": max_attempts, "fallback_used": use_fallback})

    do_search("paper", paper_queries, paper_sites)
    do_search("news", news_queries, news_sites)

    # 配额补全（仅在 Jina 可用时执行）
    def backfill(category: str, seeds: Dict, target: int):
        nonlocal raw_results, failures, jina_available
        if not jina_available:
            return
        current = sum(1 for x in raw_results if x.get("source_type") == category)
        if current >= target:
            return
        keywords = [str(k).strip() for k in seeds.get("keywords", []) if str(k).strip()]
        domains = [str(d).strip() for d in seeds.get("domains", []) if str(d).strip()]
        problem = str(seeds.get("problem_statement") or "").strip()
        if category == "paper":
            queries = [f"{term} method benchmark related work" for term in (keywords[:5] + domains[:3])]
            if problem:
                queries.append(f"{problem} literature review baseline comparison")
        else:
            queries = [f"{term} industry trend 2024 2025" for term in (domains[:4] + keywords[:4])]
            if problem:
                queries.append(f"{problem} application market adoption")
        seen = set()
        for q in queries:
            if not jina_available:
                break
            qn = q.lower().strip()
            if not qn or qn in seen:
                continue
            seen.add(qn)
            try:
                raw_results.extend(_jina_search(q, category=category))
            except Exception as e:
                err_str = str(e).lower()
                if "timed out" in err_str or "timeout" in err_str or "connection" in err_str:
                    jina_available = False
                    print(f"[node3] Jina unreachable in backfill, stopping: {e}", flush=True)
                    break
                failures.append({"query": q, "error": str(e), "site": "fallback",
                                 "attempt_count": max_attempts, "fallback_used": True})
            if len(raw_results) >= target:
                break

    backfill("paper", seeds, min_paper)
    backfill("news", seeds, min_news)

    if not jina_available and not raw_results:
        # Jina 完全不可达：优雅降级，不阻断流程，previous_work_docs 为空
        print("[node3] Jina unavailable, proceeding with empty external docs (node4 will use project paper only)", flush=True)
        _write_json(node3_dir / "seed_terms.json", seeds)
        _write_json(node3_dir / "queries.json", {"paper_queries": paper_queries, "news_queries": news_queries})
        _write_json(node3_dir / "search_failures.json", failures)
        return {
            "run_id": run_id,
            "paper_has_related_work": has_related,
            "seed_terms": seeds,
            "search_queries": {"paper_queries": paper_queries, "news_queries": news_queries},
            "search_debug": {"jina_unavailable": True, "search_failures": failures,
                             "raw_count": 0, "deduped_count": 0, "ranked_count": 0,
                             "quota_met": False},
            "previous_work_docs": [],
        }

    deduped = _dedupe(raw_results)
    paper_cnt = sum(1 for x in deduped if x.get("source_type") == "paper")
    news_cnt = sum(1 for x in deduped if x.get("source_type") == "news")
    quota_met = paper_cnt >= min_paper and news_cnt >= min_news

    for item in deduped:
        item["score"] = _score_item(item, seeds)
    ranked = sorted(deduped, key=lambda x: x.get("score", 0.0), reverse=True)[:int(os.getenv("SEARCH_TOP_N", "30"))]

    prev_docs = _to_documents(ranked, query="")

    _write_json(node3_dir / "seed_terms.json", seeds)
    _write_json(node3_dir / "queries.json", {"paper_queries": paper_queries, "news_queries": news_queries})
    _write_json(node3_dir / "raw_results.json", raw_results)
    _write_json(node3_dir / "deduped_results.json", deduped)
    _write_json(node3_dir / "ranked_results.json", ranked)

    debug = {
        "raw_count": len(raw_results), "deduped_count": len(deduped), "ranked_count": len(ranked),
        "search_failures": failures, "paper_count": paper_cnt, "news_count": news_cnt,
        "site_fail_counts": site_fail_cnt, "fallback_triggered": any(f.get("fallback_used") for f in failures),
        "quota_target": {"paper": min_paper, "news": min_news}, "quota_met": quota_met,
        "jina_available": jina_available,
    }
    print(f"[node3] done: raw={debug['raw_count']}, deduped={debug['deduped_count']}, ranked={debug['ranked_count']}", flush=True)

    return {
        "run_id": run_id,
        "paper_has_related_work": has_related,
        "seed_terms": seeds,
        "search_queries": {"paper_queries": paper_queries, "news_queries": news_queries},
        "search_debug": debug,
        "previous_work_docs": prev_docs,
    }