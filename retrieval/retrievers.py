import os
from azure.search.documents.models import VectorizedQuery
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_code_results(results_code, logger, index_name):
    docs = []
    context_parts_code = []
    kept_count = 0
    skipped_count = 0
    for r in results_code:
        score = r.get("@search.score", None)
        reranker_score = r.get("@search.reranker_score", None)
        highlights = r.get("@search.highlights", None)
        captions = r.get("@search.captions", None)
        repo = r.get("repo", "")
        file_path = r.get("filePath", "")
        symbol = r.get("symbol", "")
        code_type = r.get("type", "")
        start_line = r.get("startLine", "")
        end_line = r.get("endLine", "")
        signature = r.get("signature", "")
        opengrok_url = r.get("opengrokUrl", "")
        last_modified = r.get("lastModified", "")
        # Compose captions info for log
        if captions is None:
            captions_log = "None"
        elif isinstance(captions, list):
            captions_log = "[" + ", ".join(str(x) for x in captions) + "]"
        else:
            captions_log = str(captions)
        reranker_log = reranker_score
        score_log = score
        title = symbol
        # Keep if reranker_score is not None OR captions is non-empty/truthy
        keep = False
        if reranker_score is not None:
            keep = True
        elif captions is not None and ((isinstance(captions, list) and len(captions) > 0) or (not isinstance(captions, list) and captions)):
            keep = True
        log_line = f"Title: {title}, Score: {score_log}, Reranker: {reranker_log}, Captions: {captions_log}"
        filtered_doc = {
            "index": index_name,
            "repo": repo,
            "filePath": file_path,
            "symbol": symbol,
            "type": code_type,
            "startLine": start_line,
            "endLine": end_line,
            "signature": signature,
            "opengrokUrl": opengrok_url,
            "lastModified": last_modified,
            "score": score,
            "reranker_score": reranker_score,
            "highlights": highlights,
            "captions": captions,
        }
        if keep:
            if logger:
                logger.debug(f"[KEEP] {log_line}")
            docs.append(filtered_doc)
            context_parts_code.append(
                f"[Code]\nRepo: {repo}\nFile Path: {file_path}\nSymbol: {symbol}\nType: {code_type}\nLines: {start_line}-{end_line}\nSignature: {signature}\nOpenGrok URL: {opengrok_url}\nLast Modified: {last_modified}"
            )
            kept_count += 1
        else:
            if logger:
                logger.debug(f"[SKIP] {log_line}")
            skipped_count += 1
    # Sort docs and context_parts_code by score descending
    paired = list(zip(docs, context_parts_code))
    paired.sort(key=lambda x: x[0].get("score", 0), reverse=True)
    docs, context_parts_code = zip(*paired) if paired else ([], [])
    if logger:
        logger.debug(f"Code docs kept: {kept_count}, skipped: {skipped_count}")
    return list(docs), list(context_parts_code)

def process_docs_results(results_docs, logger, index_name):
    docs = []
    context_parts_docs = []
    kept_count = 0
    skipped_count = 0
    for r in results_docs:
        # Extract key values for structured logging
        score = r.get("@search.score", None)
        reranker_score = r.get("@search.reranker_score", None)
        highlights = r.get("@search.highlights", None)
        captions = r.get("@search.captions", None)
        title = r.get("title", "")
        url = r.get("url", "")
        parent_id = r.get("parentId", "")
        content = r.get("content", "")
        last_modified = r.get("lastModified", "")
        # Compose captions info for log
        if captions is None:
            captions_log = "None"
        elif isinstance(captions, list):
            captions_log = "[" + ", ".join(str(x) for x in captions) + "]"
        else:
            captions_log = str(captions)
        # Compose reranker for log
        reranker_log = reranker_score
        # Compose score for log
        score_log = score
        # Keep if reranker_score is not None OR captions is non-empty/truthy
        keep = False
        if reranker_score is not None:
            keep = True
        elif captions is not None and ((isinstance(captions, list) and len(captions) > 0) or (not isinstance(captions, list) and captions)):
            keep = True
        log_line = f"Title: {title}, Score: {score_log}, Reranker: {reranker_log}, Captions: {captions_log}"
        filtered_doc = {
            "index": index_name,
            "title": title,
            "url": url,
            "content": content,
            "lastModified": last_modified,
            "parentId": parent_id,
            "score": score,
            "reranker_score": reranker_score,
            "highlights": highlights,
            "captions": captions,
        }
        if keep:
            if logger:
                logger.debug(f"[KEEP] {log_line}")
            docs.append(filtered_doc)
            context_parts_docs.append(
                f"[Document]\nTitle: {title}\nURL: {url}\nScore: {score}\nLast Modified: {last_modified}\nParent ID: {parent_id}\nContent: {content}"
            )
            kept_count += 1
        else:
            if logger:
                logger.debug(f"[SKIP] {log_line}")
            skipped_count += 1
    # Sort docs and context_parts_docs by score descending
    paired = list(zip(docs, context_parts_docs))
    paired.sort(key=lambda x: x[0].get("score", 0), reverse=True)
    docs, context_parts_docs = zip(*paired) if paired else ([], [])
    if logger:
        logger.debug(f"Docs kept: {kept_count}, skipped: {skipped_count}")
    return list(docs), list(context_parts_docs)

def retrieve_docs_results(search_client_docs, query, query_embedding, K_DOCS, index_name, logger):
    vector_query = VectorizedQuery(
        vector=query_embedding,
        fields="embedding",
        kind="vector",
        k_nearest_neighbors=K_DOCS
    )
    results_docs = search_client_docs.search(
        search_text=query,
        vector_queries=[vector_query],
        include_total_count=True,
        top=100,
        select=["id", "title", "url", "parentId", "content", "lastModified"],
        query_type="semantic",
        query_language="en",
        semantic_configuration_name="default",
        query_caption="extractive",
        query_answer="extractive"
    )
    docs, context_parts_docs = process_docs_results_parallel(results_docs, logger, index_name)
    return docs, context_parts_docs


def retrieve_code_results(search_client_code, query, query_embedding, K_CODE, index_name, logger, repos=None):
    from dotenv import load_dotenv
    import os
    load_dotenv()
    K_TOP_CODE_PER_REPO = int(os.getenv("TOP_AISEARCH_CODE_RESULTS_PER_REPO", 5))

    vector_query = VectorizedQuery(
        vector=query_embedding,
        fields="embedding",
        kind="vector",
        k_nearest_neighbors=K_CODE
    )

    results_code = []

    # Case 1: multiple repos → fetch per repo
    if repos and len(repos) > 1:
        logger.debug(f"Fetching top {K_TOP_CODE_PER_REPO} results per repo for {len(repos)} repos.")
        for repo in repos:
            try:
                repo_filter = f"repo eq '{repo}'"
                repo_results = search_client_code.search(
                    search_text=query,
                    vector_queries=[vector_query],
                    include_total_count=True,
                    select=[
                        "repo", "filePath", "symbol", "type", "startLine", "endLine",
                        "signature", "opengrokUrl", "lastModified"
                    ],
                    query_type="semantic",
                    query_language="en",
                    semantic_configuration_name="default",
                    filter=repo_filter,
                )
                repo_results_list = list(repo_results)[:K_TOP_CODE_PER_REPO]
                results_code.extend(repo_results_list)
                logger.debug(f"Retrieved {len(repo_results_list)} results from repo '{repo}'.")
            except Exception as e:
                logger.exception(f"Error retrieving results for repo '{repo}': {e}")

    # Case 2: single repo or no filter → existing single-query behavior
    else:
        filter_expr = None
        if repos:
            repo_filters = " or ".join([f"repo eq '{r}'" for r in repos])
            filter_expr = f"({repo_filters})"

        results_code = search_client_code.search(
            search_text=query,
            vector_queries=[vector_query],
            include_total_count=True,
            select=[
                "repo", "filePath", "symbol", "type", "startLine", "endLine",
                "signature", "opengrokUrl", "lastModified"
            ],
            query_type="semantic",
            query_language="en",
            semantic_configuration_name="default",
            filter=filter_expr
        )

    # Process and return
    docs, context_parts_code = process_code_results_parallel(results_code, logger, index_name)
    return docs, context_parts_code


def retrieve_mixed_results(
    search_client_docs,
    search_client_code,
    query,
    query_embedding,
    K_DOCS,
    K_CODE,
    index_docs_name,
    index_code_name,
    logger,
    repos=None
):
    """
    Retrieve top documents from both docs and code indexes.
    If repos are provided, code retrieval is filtered accordingly.
    """
    docs_docs, context_parts_docs = retrieve_docs_results(
        search_client_docs, query, query_embedding, K_DOCS, index_docs_name, logger
    )

    docs_code, context_parts_code = retrieve_code_results(
        search_client_code, query, query_embedding, K_CODE, index_code_name, logger, repos
    )

    return docs_docs, docs_code, context_parts_docs, context_parts_code


def retrieve_mixed_results_threaded(
    search_client_docs,
    search_client_code,
    query,
    query_embedding,
    K_DOCS,
    K_CODE,
    index_docs_name,
    index_code_name,
    logger,
    repos=None
):
    """
    Retrieve top documents from both docs and code indexes concurrently using threading.
    """
    results = {}

    def docs_job():
        return retrieve_docs_results(
            search_client_docs, query, query_embedding, K_DOCS, index_docs_name, logger
        )

    def code_job():
        return retrieve_code_results(
            search_client_code, query, query_embedding, K_CODE, index_code_name, logger, repos
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(docs_job): "docs",
            executor.submit(code_job): "code"
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
                if logger:
                    logger.debug(f"[THREAD:{key}] Retrieved {len(results[key][0])} items.")
            except Exception as e:
                logger.exception(f"[THREAD-ERROR] {key} retrieval failed: {e}")
                results[key] = ([], [])

    docs_docs, context_parts_docs = results.get("docs", ([], []))
    docs_code, context_parts_code = results.get("code", ([], []))
    return docs_docs, docs_code, context_parts_docs, context_parts_code

def process_docs_results_parallel(results_docs, logger, index_name):
    """
    Parallel version of process_docs_results using ThreadPoolExecutor.
    Processes each document concurrently for faster filtering and formatting.
    """
    docs, context_parts_docs = [], []
    kept_count = skipped_count = 0

    def process_one_doc(r):
        score = r.get("@search.score", None)
        reranker_score = r.get("@search.reranker_score", None)
        highlights = r.get("@search.highlights", None)
        captions = r.get("@search.captions", None)
        doc_id = r.get("id", "")
        title = r.get("title", "")
        url = r.get("url", "")
        parent_id = r.get("parentId", "")
        content = r.get("content", "")
        last_modified = r.get("lastModified", "")

        keep = False
        if reranker_score is not None:
            keep = True
        elif captions and ((isinstance(captions, list) and len(captions) > 0) or (not isinstance(captions, list))):
            keep = True

        filtered_doc = {
            "index": index_name,
            "id": doc_id,
            "title": title,
            "url": url,
            "content": content,
            "lastModified": last_modified,
            "parentId": parent_id,
            "score": score,
            "reranker_score": reranker_score,
            "highlights": highlights,
            "captions": captions,
        }

        if keep:
            context = (
                f"[Document]\nTitle: {title}\nURL: {url}\nScore: {score}\n"
                f"Last Modified: {last_modified}\nParent ID: {parent_id}\nContent: {content}"
            )
            return ("keep", filtered_doc, context)
        else:
            return ("skip", None, None)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_one_doc, r) for r in results_docs]
        for f in as_completed(futures):
            status, doc, ctx = f.result()
            if status == "keep":
                docs.append(doc)
                context_parts_docs.append(ctx)
                kept_count += 1
            else:
                skipped_count += 1

    paired = list(zip(docs, context_parts_docs))
    paired.sort(key=lambda x: x[0].get("score", 0), reverse=True)
    docs, context_parts_docs = zip(*paired) if paired else ([], [])
    if logger:
        logger.debug(f"[THREAD:DOCS] kept={kept_count}, skipped={skipped_count}")
    return list(docs), list(context_parts_docs)


def process_code_results_parallel(results_code, logger, index_name):
    """
    Parallel version of process_code_results using ThreadPoolExecutor.
    Processes each code snippet concurrently for faster filtering and formatting.
    """
    docs, context_parts_code = [], []
    kept_count = skipped_count = 0

    def process_one_code(r):
        score = r.get("@search.score", None)
        reranker_score = r.get("@search.reranker_score", None)
        captions = r.get("@search.captions", None)
        repo = r.get("repo", "")
        file_path = r.get("filePath", "")
        symbol = r.get("symbol", "")
        code_type = r.get("type", "")
        start_line = r.get("startLine", "")
        end_line = r.get("endLine", "")
        signature = r.get("signature", "")
        opengrok_url = r.get("opengrokUrl", "")
        last_modified = r.get("lastModified", "")

        keep = False
        if reranker_score is not None:
            keep = True
        elif captions and ((isinstance(captions, list) and len(captions) > 0) or (not isinstance(captions, list))):
            keep = True

        filtered_doc = {
            "index": index_name,
            "repo": repo,
            "filePath": file_path,
            "symbol": symbol,
            "type": code_type,
            "startLine": start_line,
            "endLine": end_line,
            "signature": signature,
            "opengrokUrl": opengrok_url,
            "lastModified": last_modified,
            "score": score,
            "reranker_score": reranker_score,
            "captions": captions,
        }

        if keep:
            context = (
                f"[Code]\nRepo: {repo}\nFile Path: {file_path}\nSymbol: {symbol}\n"
                f"Type: {code_type}\nLines: {start_line}-{end_line}\nSignature: {signature}\n"
                f"OpenGrok URL: {opengrok_url}\nLast Modified: {last_modified}"
            )
            return ("keep", filtered_doc, context)
        else:
            return ("skip", None, None)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_one_code, r) for r in results_code]
        for f in as_completed(futures):
            status, doc, ctx = f.result()
            if status == "keep":
                docs.append(doc)
                context_parts_code.append(ctx)
                kept_count += 1
            else:
                skipped_count += 1

    paired = list(zip(docs, context_parts_code))
    paired.sort(key=lambda x: x[0].get("score", 0), reverse=True)
    docs, context_parts_code = zip(*paired) if paired else ([], [])
    if logger:
        logger.debug(f"[THREAD:CODE] kept={kept_count}, skipped={skipped_count}")
    return list(docs), list(context_parts_code)


def process_ptr_results_parallel(results_ptr, logger, index_name):
    """
    Process PTR search results similar to docs/code processors.
    We keep records if reranker_score exists OR captions exist (when semantic captions are enabled),
    otherwise keep everything (PTR index may not have captions/reranker depending on query).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    docs = []
    kept_count = skipped_count = 0

    def process_one(r):
        score = r.get("@search.score", None)
        reranker_score = r.get("@search.reranker_score", None)
        captions = r.get("@search.captions", None)
        highlights = r.get("@search.highlights", None)

        # Keep logic:
        # - If reranker/captions exist, use the same rule as docs/code.
        # - If neither exists (common for some configurations), keep by default.
        has_any_semantic_signal = (reranker_score is not None) or (captions is not None)
        keep = True
        if has_any_semantic_signal:
            keep = False
            if reranker_score is not None:
                keep = True
            elif captions is not None and (
                (isinstance(captions, list) and len(captions) > 0) or (not isinstance(captions, list) and captions)
            ):
                keep = True

        # Do not assume schema; keep the full record minus huge fields if any
        filtered = dict(r)
        filtered["index"] = index_name
        filtered["score"] = score
        filtered["reranker_score"] = reranker_score
        filtered["highlights"] = highlights
        filtered["captions"] = captions

        if keep:
            return ("keep", filtered)
        return ("skip", None)

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(process_one, r) for r in results_ptr]
        for f in as_completed(futures):
            status, doc = f.result()
            if status == "keep":
                docs.append(doc)
                kept_count += 1
            else:
                skipped_count += 1

    # Sort: prefer reranker_score then score
    docs.sort(key=lambda x: ((x.get("reranker_score") or 0), (x.get("score") or 0)), reverse=True)

    if logger:
        logger.debug(f"[THREAD:PTRS] kept={kept_count}, skipped={skipped_count}")

    return docs


def retrieve_ptr_results(search_client_ptrs, query, query_embedding, K_PTRS, index_name, logger):
    """
    Retrieve similar PTRs from a PTR index (vector + semantic hybrid).
    We deliberately do NOT use 'select' to avoid breaking if index schema differs.
    """
    top = int(os.getenv("TOP_AISEARCH_PTR_RESULTS", "50"))
    semantic_name = os.getenv("PTR_SEMANTIC_CONFIG", "semconfig")

    # PTR index vector field name (see create_ptr_index.py -> contentVector)
    ptr_vector_field = os.getenv("PTR_VECTOR_FIELD", "contentVector").strip() or "contentVector"

    vector_query = VectorizedQuery(
        vector=query_embedding,
        fields=ptr_vector_field,
        kind="vector",
        k_nearest_neighbors=K_PTRS
    )

    try:
        results_ptrs = search_client_ptrs.search(
            search_text=query,
            vector_queries=[vector_query],
            include_total_count=True,
            top=top,
            query_type="semantic",
            query_language="en",
            semantic_configuration_name=semantic_name,
            query_caption="extractive",
            query_answer="extractive",
        )
    except Exception as e:
        # If vector field is wrong/missing, fall back to semantic-only so the feature still works.
        msg = str(e)
        if ("Unknown field" in msg and "vector field" in msg) or ("UnknownField" in msg) or ("InvalidRequestParameter" in msg):
            if logger:
                logger.warning(
                    f"[PTR_SIMILAR] Vector field '{ptr_vector_field}' not valid for index '{index_name}'. Falling back to semantic-only search.")
            results_ptrs = search_client_ptrs.search(
                search_text=query,
                include_total_count=True,
                top=top,
                query_type="semantic",
                query_language="en",
                semantic_configuration_name=semantic_name,
                query_caption="extractive",
                query_answer="extractive",
            )
        else:
            raise

    docs_ptrs = process_ptr_results_parallel(results_ptrs, logger, index_name)
    return docs_ptrs