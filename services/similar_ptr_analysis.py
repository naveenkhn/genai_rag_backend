import os
import re

# Reuse existing helpers from ptr_analysis to avoid duplicating logic
from services.ptr_analysis import (
    aproach_record_link,
    _load_domain_terms_text,
    llm_plan_doc_queries,
    _dedupe_docs,
    _build_docs_context,
)

_PTR_SIMILAR_RE = re.compile(
    r"(?i)\b(?:find\s+similar\s+ptrs?|similar\s+ptrs?)\b[:\-]?\s*(.+)$"
)
def extract_similar_ptr_issue(query: str):
    q = (query or "").strip()
    m = _PTR_SIMILAR_RE.search(q)
    if not m:
        return None
    issue = (m.group(1) or "").strip()
    return issue if issue else None
def _build_ptrs_context(ptr_hits: list, max_ptrs: int, logger=None) -> str:
    """
    Build a compact context block from PTR index hits.
    Keep it small and evidence-oriented: title, key extracted fields, fix/workaround, and record_url.
    """
    total = len(ptr_hits or [])
    used = 0
    parts = []

    for h in (ptr_hits or []):
        if used >= max_ptrs:
            break

        # Try multiple likely field names (index schema may vary)
        record_id = h.get("record_id") or h.get("recordId") or h.get("id") or ""
        title = h.get("title") or h.get("subject") or ""
        record_url = h.get("record_url") or h.get("recordUrl") or ""
        # If record_url missing but record_id exists, use your known NotesLink format
        if (not record_url) and record_id:
            record_url = aproach_record_link(record_id)

        status = h.get("status") or ""
        severity = h.get("severity") or ""
        system = h.get("system") or ""
        problem_summary = h.get("problem_summary") or h.get("problemSummary") or ""
        rootcause = h.get("rootcause") or h.get("root_cause") or ""
        fix = h.get("fix_workaround") or h.get("fix") or h.get("workaround") or ""
        keywords = h.get("keywords") or []

        # Captions (if semantic captions enabled)
        captions = h.get("@search.captions") or h.get("captions")
        caption_text = ""
        if isinstance(captions, list) and captions:
            # Azure captions are often [{"text": "...", "highlights": "..."}]
            c0 = captions[0]
            if isinstance(c0, dict):
                caption_text = c0.get("text", "") or ""
            else:
                caption_text = str(c0)
        elif isinstance(captions, str):
            caption_text = captions

        # Trim long fields
        def _trim(s, n=700):
            if not s:
                return ""
            s = str(s)
            return s if len(s) <= n else s[:n] + "…"

        parts.append(
            "[Similar PTR]\n"
            f"Title: {_trim(title, 180)}\n"
            f"PTR Link: {record_url}\n"
            f"Record ID: {record_id}\n"
            f"System: {system}\n"
            f"Status: {status}\n"
            f"Severity: {severity}\n"
            f"Problem summary: {_trim(problem_summary)}\n"
            f"Root cause: {_trim(rootcause)}\n"
            f"Fix/Workaround: {_trim(fix)}\n"
            f"Keywords: {keywords}\n"
            f"Caption: {_trim(caption_text, 300)}"
        )
        used += 1

    if logger:
        logger.info(
            "[PTR_SIMILAR_CONTEXT] "
            f"input_ptrs={total}, max_ptrs={max_ptrs}, ptrs_in_context={used}, trimmed={max(0, total - used)}"
        )

    return "\n\n".join(parts)

def llm_answer_similar_ptrs(llm, issue_description: str, ptr_hits: list, docs_context: str):
    sys = (
        "You are an assistant helping engineers investigate an issue by finding similar Winaproach PTRs and relevant documentation.\n"
        "You will be given:\n"
        "- The user's issue description (primary)\n"
        "- A list of similar PTR records retrieved from the PTR index (secondary evidence)\n"
        "- Documentation context snippets retrieved from Confluence (supporting evidence)\n\n"
        "Rules:\n"
        "- Do NOT invent PTR IDs, titles, or links. Use only what is provided in the PTR hits.\n"
        "- Use 'PTR Link' from the PTR hits when referencing a PTR.\n"
        "- If you infer something, label it clearly as an inference and tie it to evidence.\n"
        "- Prefer to propose investigation steps and likely solution patterns grounded in retrieved PTRs/docs.\n\n"
        "Return MARKDOWN with sections:\n"
        "- Overview (1–2 lines)\n"
        "- Interpretation of the issue (based on the user description)\n"
        "- Top similar PTRs (ranked list with 1–2 line 'why similar' and any fix/workaround highlights)\n"
        "- Common patterns observed (root-cause/fix themes across PTRs) (only if evidence exists)\n"
        "- Documentation insights (expected behavior / procedures) (cite docs)\n"
        "- Suggested investigation plan (5–8 steps; cite docs where relevant)\n"
        "- Conclusion / Confidence (2–4 lines)\n\n"
        "Doc citations format: [Document: <title> | URL: <url>]\n"
    )

    domain_terms_text = _load_domain_terms_text()

    user = (
        "User issue description:\n"
        f"{issue_description}\n\n"
        "Domain terminology reference (glossary):\n"
        + (domain_terms_text[:12000] if domain_terms_text else "(none)") + "\n\n"
        "Similar PTRs (retrieved):\n"
        + (_build_ptrs_context(ptr_hits, max_ptrs=len(ptr_hits) or 0) if ptr_hits else "(none)") + "\n\n"
        "Documentation context (retrieved):\n"
        + (docs_context or "(none)")
    )

    resp = llm.invoke([
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ])
    return str(resp.content or "").strip()


def ptr_similar_flow(
    query,
    llm,
    embeddings,
    search_client_ptrs,
    retrieve_ptr_results_fn,
    index_ptrs_name,
    search_client_docs,
    retrieve_docs_results_fn,
    index_docs_name,
    logger,
):
    issue = extract_similar_ptr_issue(query)
    if not issue:
        return None

    logger.info(f"[PTR_SIMILAR] Searching similar PTRs for issue: {issue[:120]}")

    # 1) Retrieve similar PTRs from PTR index
    k_ptrs = int(os.getenv("K_TOP_PTRS", os.getenv("TOP_AISEARCH_PTR_KNN", "200")))
    max_ptrs_context = int(os.getenv("PTR_SIMILAR_MAX_PTRS_CONTEXT", "150"))

    try:
        issue_emb = embeddings.embed_query(issue)
    except Exception as e:
        logger.error(f"[PTR_SIMILAR] embedding failed: {e}")
        return {"answer": f"Failed to embed issue description: {e}", "sources": []}

    ptr_hits = []
    try:
        ptr_hits = retrieve_ptr_results_fn(
            search_client_ptrs, issue, issue_emb, k_ptrs, index_ptrs_name, logger
        ) or []
    except Exception as e:
        logger.warning(f"[PTR_SIMILAR] PTR retrieval failed: {e}")
        ptr_hits = []

    # Keep only top N for context; return full list in sources later
    ptr_hits_context = (ptr_hits or [])[:max_ptrs_context]

    # 2) Use existing llm_plan_doc_queries to get 3 doc queries (reuse logic; do NOT touch summarize ptr flow)
    # Build a pseudo-normalized record using the issue as timeline_text/title.
    pseudo_ptr_norm = {
        "record_id": None,
        "title": issue[:200],
        "rec_type": "PTR",
        "status": {},
        "severity": {},
        "system": {},
        "category": {},
        "component": {},
        "type": {},
        "keywords": [],
        "timeline_text": issue[:24000],
    }

    plan = llm_plan_doc_queries(llm=llm, ptr_norm=pseudo_ptr_norm, logger=logger, n_queries=3)
    doc_queries = (plan.get("queries", []) or [])

    if logger:
        logger.info("[PTR_SIMILAR_DOC_QUERIES] LLM decided the following documentation queries:")
        for i, q in enumerate(doc_queries, start=1):
            logger.info(f"  {i}. intent={q.get('intent')} | query=\"{q.get('query')}\"")

    # 3) Retrieve docs for those queries, merge + dedupe, build docs_context (reuse existing helpers)
    top_docs_per_q = int(os.getenv("TOP_AISEARCH_DOC_RESULTS", "200"))
    max_docs_total = int(os.getenv("PTR_MAX_DOCS_TOTAL", "100"))

    merged_docs = []
    for q in doc_queries:
        qtext = (q.get("query") or "").strip() if isinstance(q, dict) else str(q).strip()
        if not qtext:
            continue
        try:
            q_emb = embeddings.embed_query(qtext)
            docs_q, _ = retrieve_docs_results_fn(
                search_client_docs, qtext, q_emb, top_docs_per_q, index_docs_name, logger
            )
            if docs_q:
                merged_docs.extend(docs_q)
        except Exception as e:
            logger.warning(f"[PTR_SIMILAR_DOCS] retrieval failed for query='{qtext}': {e}")

    logger.info(f"[PTR_SIMILAR_DOCS] Docs before de-duplication: {len(merged_docs)}")
    merged_docs = _dedupe_docs(merged_docs)
    logger.info(f"[PTR_SIMILAR_DOCS] Docs after de-duplication: {len(merged_docs)}")

    docs_context = _build_docs_context(merged_docs, max_docs=max_docs_total, logger=logger)

    # 4) Final answer LLM call: issue + similar PTRs + docs
    answer_md = llm_answer_similar_ptrs(
        llm=llm,
        issue_description=issue,
        ptr_hits=ptr_hits_context,
        docs_context=docs_context,
    )

    # Build sources list: include PTR hits and doc hits (capped)
    sources = []

    # PTR sources
    for h in (ptr_hits or [])[:20]:
        rid = h.get("record_id") or h.get("recordId") or h.get("id")
        rurl = h.get("record_url") or h.get("recordUrl") or (aproach_record_link(rid) if rid else "")
        sources.append({
            "type": "ptr",
            "record_id": rid,
            "url": rurl,
            "title": h.get("title") or "",
            "score": h.get("score"),
            "reranker_score": h.get("reranker_score"),
        })

    # Doc sources
    for d in (merged_docs or [])[:max_docs_total]:
        sources.append({
            "type": "doc",
            "title": d.get("title", ""),
            "url": d.get("url", ""),
            "score": d.get("score", None),
            "reranker_score": d.get("reranker_score", None),
            "lastModified": d.get("lastModified", ""),
        })

    return {"answer": answer_md, "sources": sources}
