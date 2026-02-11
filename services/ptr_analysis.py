import os
import json
import re
import time
import urllib.request
import urllib.error
import yaml
from prompts.prompts import CLASSIFIER_MSG_DOCS

import urllib.request
import urllib.error
import ssl

# TEMPORARY WORKAROUND:
# MCP server presents an expired TLS certificate (issuer cert expired Jan 12 2026).
# This disables SSL verification for urllib until the MCP ingress cert is rotated.
ssl._create_default_https_context = ssl._create_unverified_context

# --- intent detection ---
_PTR_SUMMARY_RE = re.compile(
    r"(?i)\b(?:summarize|summary)\s+ptr\s*(?:#|:)?\s*(\d{7,8})\b"
    r"|\bptr\s*(?:#|:)?\s*(\d{7,8})\s*(?:summary|summarize)?\b"
)

_MCP_SESSION_ID = None
_MCP_DID_NOTIFY_INITIALIZED = False
_SESSION_INVALID_PAT = re.compile(r"no valid session id provided", re.IGNORECASE)

def aproach_record_link(record_id: str | int) -> str:
    rid = str(record_id).strip()
    return f"https://aproach.muc.amadeus.net/NotesLink/nl?RNID={rid}"

def mcp_initialize_session(mcp_url, headers):
    init_obj = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "clientInfo": {"name": "rag_backend", "version": "0.1"},
            "protocolVersion": "2024-11-05",
            "capabilities": {},
        },
    }
    req = urllib.request.Request(
        mcp_url,
        data=json.dumps(init_obj).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp.read()  # consume SSE body
        sid = resp.headers.get("mcp-session-id")
    if not sid:
        raise RuntimeError("MCP initialize did not return mcp-session-id header")
    return sid.strip()


def mcp_send_initialized(mcp_url: str, mcp_user: str, mcp_pass: str, mcp_email: str) -> None:
    """
    MCP protocol handshake step after initialize: notifications/initialized.
    Your working import flow does this; some servers reject tools/call until it is sent.
    """
    global _MCP_SESSION_ID, _MCP_DID_NOTIFY_INITIALIZED

    if _MCP_DID_NOTIFY_INITIALIZED:
        return
    if not _MCP_SESSION_ID:
        raise RuntimeError("mcp_send_initialized called without MCP session id")

    notify_obj = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }

    headers = _mcp_headers(mcp_user, mcp_pass, mcp_email)
    headers["mcp-session-id"] = _MCP_SESSION_ID

    req = urllib.request.Request(
        url=mcp_url,
        data=json.dumps(notify_obj).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        resp.read()

    _MCP_DID_NOTIFY_INITIALIZED = True


def extract_ptr_id(query: str):
    m = _PTR_SUMMARY_RE.search((query or "").strip())
    if not m:
        return None
    return m.group(1) or m.group(2)


def _parse_sse_last_json(raw_text: str):
    """Return the last JSON object from SSE lines like: data: {...}"""
    last_obj = None
    for line in (raw_text or "").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        try:
            last_obj = json.loads(payload)
        except Exception:
            continue
    return last_obj


def _mcp_headers(mcp_user, mcp_pass, mcp_email):
    return {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-aproach-username": mcp_user or "",
        "x-aproach-password": mcp_pass or "",
        "x-aproach-user-email": mcp_email or "",
    }


def mcp_tool_call(mcp_url: str, tool_name: str, arguments: dict, mcp_user: str, mcp_pass: str, mcp_email: str):
    global _MCP_SESSION_ID, _MCP_DID_NOTIFY_INITIALIZED

    if not mcp_url:
        raise RuntimeError("MCP_URL is not set")

    if not _MCP_SESSION_ID:
        headers_init = _mcp_headers(mcp_user, mcp_pass, mcp_email)
        _MCP_SESSION_ID = mcp_initialize_session(mcp_url, headers_init)
        _MCP_DID_NOTIFY_INITIALIZED = False
        mcp_send_initialized(mcp_url, mcp_user, mcp_pass, mcp_email)

    req_obj = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000) % 1000000000,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    headers = _mcp_headers(mcp_user, mcp_pass, mcp_email)
    headers["mcp-session-id"] = _MCP_SESSION_ID

    req = urllib.request.Request(
        url=mcp_url,
        data=json.dumps(req_obj).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    def _send_once() -> str:
        headers = _mcp_headers(mcp_user, mcp_pass, mcp_email)
        headers["mcp-session-id"] = _MCP_SESSION_ID
        req = urllib.request.Request(
            url=mcp_url,
            data=json.dumps(req_obj).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read().decode("utf-8", errors="replace")

    try:
        raw = _send_once()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")

        if e.code == 400 and _SESSION_INVALID_PAT.search(body or ""):
            headers_init = _mcp_headers(mcp_user, mcp_pass, mcp_email)
            _MCP_SESSION_ID = mcp_initialize_session(mcp_url, headers_init)
            _MCP_DID_NOTIFY_INITIALIZED = False
            mcp_send_initialized(mcp_url, mcp_user, mcp_pass, mcp_email)
            try:
                raw = _send_once()
            except urllib.error.HTTPError as e2:
                body2 = e2.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"MCP HTTP {e2.code} {e2.reason}: {body2[:2000]}")
        else:
            raise RuntimeError(f"MCP HTTP {e.code} {e.reason}: {body[:2000]}")

    obj = _parse_sse_last_json(raw)
    if not obj:
        raise RuntimeError("MCP: could not parse SSE JSON response")

    # JSON-RPC error shape: {"jsonrpc":"2.0","id":...,"error":{"code":...,"message":...,"data":...}}
    if isinstance(obj, dict) and obj.get("error") is not None:
        err = obj.get("error") or {}
        code = err.get("code", "")
        msg = (err.get("message", "") or "").strip()
        data = err.get("data", None)
        if data is not None and str(data).strip():
            raise RuntimeError(f"MCP RPC error {code}: {msg} | data={str(data)[:1000]}")
        try:
            err_preview = json.dumps(err, ensure_ascii=False)[:1200]
        except Exception:
            err_preview = str(err)[:1200]
        raise RuntimeError(f"MCP RPC error {code}: {msg} | err={err_preview}")

    result = obj.get("result")
    if result is None:
        preview = raw[:2000] if isinstance(raw, str) else str(raw)[:2000]
        raise RuntimeError(f"MCP: missing result. Keys={list(obj.keys())}. RawPreview={preview}")

    # MCP error shape typically: {"content":[...], "isError": true}
    if isinstance(result, dict) and result.get("isError"):
        raise RuntimeError(f"MCP tool error: {result.get('content')}")

    return result


def fetch_ptr_fft(ptr_id: str, mcp_url: str, mcp_user: str, mcp_pass: str, mcp_email: str):
    # MCP expects record_id to be a string
    return mcp_tool_call(
        mcp_url=mcp_url,
        tool_name="fetch_record_details",
        arguments={"record_id": str(ptr_id), "with_details": "fft"},
        mcp_user=mcp_user,
        mcp_pass=mcp_pass,
        mcp_email=mcp_email,
    )


def _dedupe_timeline(entries: list):
    seen = set()
    out = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        t = (e.get("text") or e.get("message") or "").strip()
        ts = (e.get("date") or e.get("timestamp") or e.get("created") or "").strip()
        key = (ts, t)
        if not t:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "date": ts or "",
            "author": (e.get("author") or e.get("user") or ""),
            "text": t,
        })
    return out

def _pick(d: dict, *keys: str):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur if isinstance(cur, dict) else None


def _dict_val(d):
    if not isinstance(d, dict):
        return None
    v = d.get("value")
    return str(v) if v is not None else None


def _dict_desc(d):
    if not isinstance(d, dict):
        return None
    return d.get("description")


def _group_code(d):
    if not isinstance(d, dict):
        return None
    return d.get("code")


def _group_name(d):
    if not isinstance(d, dict):
        return None
    return d.get("name")

def _user_obj(u):
    """
    Normalize MCP user object into the shape used by normalized_records:
    {login, short_name, location, user_type}
    """
    if not isinstance(u, dict):
        return None
    login = u.get("login")
    short_name = u.get("short_name") or u.get("shortName") or u.get("name")
    location = u.get("location")
    user_type = u.get("user_type") or u.get("userType") or u.get("type")

    out = {}
    if login is not None:
        out["login"] = str(login)
    if short_name is not None:
        out["short_name"] = str(short_name)
    if location is not None:
        out["location"] = str(location)
    if user_type is not None:
        out["user_type"] = str(user_type)

    return out or None

def _keywords_to_list(s):
    if not s:
        return []
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p]


def _try_parse_json_loose(text: str):
    """
    Try to parse JSON from MCP content text.
    Handles:
      - pure JSON string
      - JSON embedded in surrounding text (extract first {...} block)
    Returns dict or None.
    """
    if not text:
        return None
    t = text.strip()

    # 1) direct JSON
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) embedded JSON object
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        candidate = t[start : end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def normalize_ptr(ptr_id: str, mcp_result: dict, max_timeline: int = 50):
    """
    Normalize MCP FFT payload into the SAME schema you already use in normalized_records,
    like ptr_29672576.json.
    """
    # 1) Extract raw payload text from MCP result.content[].text
    raw_text_parts = []
    if isinstance(mcp_result, dict):
        content = mcp_result.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    raw_text_parts.append(c.get("text", ""))

    raw_text = "\n".join([t for t in raw_text_parts if t]).strip()

    # 2) Convert to JSON dict first
    payload = _try_parse_json_loose(raw_text) or {}

    # 3) Expected FFT shape: payload["records"][0]["common"], payload["records"][0]["ffts"]
    recs = payload.get("records") if isinstance(payload, dict) else None
    if isinstance(recs, list) and recs:
        r0 = recs[0] if isinstance(recs[0], dict) else {}
        common = r0.get("common") or {}
        ffts = r0.get("ffts") or []

        # value/description objects
        rec_typeid = _pick(common, "rec_typeid")
        statusid   = _pick(common, "statusid")
        severityid = _pick(common, "severityid")
        urgencyid  = _pick(common, "urgencyid")
        systemid   = _pick(common, "as_systemid")
        categoryid = _pick(common, "categoryid")
        componentid = _pick(common, "componentid")
        typeid     = _pick(common, "typeid")

        record_id_val = common.get("rnid") or ptr_id
        try:
            record_id_int = int(record_id_val)
        except Exception:
            record_id_int = int(ptr_id)

        title = (common.get("title") or "").strip()

        # groups
        assignee_group_obj = common.get("assignee_groupid") if isinstance(common, dict) else None
        responsible_group_obj = common.get("responsible_groupid") if isinstance(common, dict) else None

        # users (as your normalized_records)
        logged_user = common.get("logged_userid") if isinstance(common, dict) else None
        assignee_user = common.get("assignee_userid") if isinstance(common, dict) else None
        responsible_user = common.get("responsible_userid") if isinstance(common, dict) else None
        modified_user = common.get("modified_userid") if isinstance(common, dict) else None

        # timeline from ffts[] : keep ALL entries (cap to max_timeline), dedupe
        timeline = []
        if isinstance(ffts, list):
            for e in ffts:
                if not isinstance(e, dict):
                    continue
                txt = e.get("fft")
                if not txt:
                    continue
                ts = e.get("update_date") or ""
                g = _group_code(e.get("update_groupid")) or ""
                uobj = e.get("update_userid")
                user_login = (uobj or {}).get("login") if isinstance(uobj, dict) else None
                timeline.append({"ts": ts, "text": txt, "group": g, "user": user_login})

        timeline.sort(key=lambda x: (not x.get("ts"), x.get("ts") or ""))

        seen = set()
        deduped = []
        for t in timeline:
            key = ((t.get("ts") or "").strip(), (t.get("text") or "").strip())
            if not key[1]:
                continue
            if key in seen:
                continue
            seen.add(key)
            deduped.append(t)
        timeline = deduped

        if max_timeline and len(timeline) > max_timeline:
            timeline = timeline[:max_timeline]

        timeline_text = "\n\n".join(
            [f"[{t.get('ts') or 'unknown'}] {(t.get('group') or 'UNKNOWN')}: {t.get('text')}".strip()
             for t in timeline]
        ).strip()

        return {
            "source": "winaproach",
            "record_id": record_id_int,
            "record_url": aproach_record_link(record_id_int),
            "rec_type": (rec_typeid or {}).get("type_name") or "PTR",
            "title": title,
            "status":   {"code": _dict_val(statusid) or "",   "label": _dict_desc(statusid) or ""},
            "severity": {"code": _dict_val(severityid) or "", "label": _dict_desc(severityid) or ""},
            "urgency":  {"code": _dict_val(urgencyid) or "",  "label": _dict_desc(urgencyid) or ""},
            "system":   {"code": _dict_val(systemid) or "",   "label": _dict_desc(systemid) or ""},
            "category": {"code": _dict_val(categoryid) or "", "label": _dict_desc(categoryid) or ""},
            "component":{"code": _dict_val(componentid) or "", "label": _dict_desc(componentid) or ""},
            "type":     {"code": _dict_val(typeid) or "",     "label": _dict_desc(typeid) or ""},
            "assignee_group": {
                "code": _group_code(assignee_group_obj) or "",
                "name": _group_name(assignee_group_obj) or ""
            },
            "responsible_group": {
                "code": _group_code(responsible_group_obj) or "",
                "name": _group_name(responsible_group_obj) or ""
            },
            "keywords": _keywords_to_list(common.get("keywords")),
            "dates": {
                "logged": common.get("logged_date"),
                "modified": common.get("modified_date"),
                "closed": common.get("closed_date"),
            },
            "users": {
                "logged": _user_obj(logged_user) or {},
                "assignee": _user_obj(assignee_user) or {},
                "responsible": _user_obj(responsible_user) or {},
                "modified": _user_obj(modified_user) or {},
            },
            "timeline": timeline,
            "timeline_text": timeline_text,
        }

    # fallback: still a clean JSON object, same schema
    try:
        rid_int = int(ptr_id)
    except Exception:
        rid_int = 0

    title = ""
    if isinstance(payload, dict):
        title = (payload.get("title") or payload.get("subject") or "").strip()

    return {
        "source": "winaproach",
        "record_id": rid_int,
        "record_url": aproach_record_link(rid_int or ptr_id),
        "rec_type": "PTR",
        "title": title,
        "status": {"code": "", "label": ""},
        "severity": {"code": "", "label": ""},
        "urgency": {"code": "", "label": ""},
        "system": {"code": "", "label": ""},
        "category": {"code": "", "label": ""},
        "component": {"code": "", "label": ""},
        "type": {"code": "", "label": ""},
        "assignee_group": {"code": "", "name": ""},
        "responsible_group": {"code": "", "name": ""},
        "keywords": [],
        "dates": {"logged": None, "modified": None, "closed": None},
        "users": {"logged": {}, "assignee": {}, "responsible": {}, "modified": {}},
        "timeline": [],
        "timeline_text": "",
    }

# =========================
# Confluence doc query planner (PTR enrichment)
# =========================

PTR_DOC_QUERY_PLANNER_SYSTEM = (
    "You are a senior Seat Management (SIT) investigator creating Confluence search queries.\n"
    "Goal: generate a VERY SMALL set of HIGH-QUALITY documentation queries that help explain a PTR investigation\n"
    "WITHOUT guessing or assuming facts.\n\n"
    "You will be given:\n"
    "- A normalized PTR JSON record (facts, may be incomplete)\n"
    "- A summary of what the documentation corpus contains\n"
    "- A domain terminology reference (glossary)\n\n"
    "Return AT MOST 3 queries. Each query must fill one investigation gap:\n"
    "1) Expected behavior (what should happen)\n"
    "2) Known issue / root-cause patterns (documented failure modes)\n"
    "3) Fix / workaround / validation steps (procedures)\n\n"
    "Hard rules:\n"
    "- Do NOT restate PTR content; do NOT include PTR ids, usernames, group names, or timestamps.\n"
    "- Do NOT speculate about causes.\n"
    "- Queries must be documentation-oriented and investigation-useful.\n"
    "- Prefer concrete technical terms from the PTR timeline (error signatures, component/connector ids, commands, host/service names).\n"
    "- Keep each query concise (<= 12 words when possible).\n\n"
    "Output STRICT JSON only (no markdown, no commentary):\n"
    "{\n"
    '  "queries": [\n'
    '    {"intent":"expected_behavior","query":"..."},\n'
    '    {"intent":"known_issue","query":"..."},\n'
    '    {"intent":"fix_validation","query":"..."}\n'
    "  ],\n"
    "}\n"
)

def _load_domain_terms_text(logger=None) -> str:
    try:
        with open("data/domain_terms.yaml", "r") as f:
            obj = yaml.safe_load(f)
        if isinstance(obj, (dict, list)):
            return yaml.dump(obj, sort_keys=False, allow_unicode=True)
    except Exception as e:
        if logger:
            logger.warning(f"Could not load domain_terms.yaml for PTR doc query planning: {e}")
    return ""

def _parse_json_strict(raw: str):
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty LLM output")
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)

def llm_plan_doc_queries(llm, ptr_norm: dict, logger=None, n_queries: int = 3) -> dict:
    """
    Produce up to 3 high-signal Confluence queries for PTR enrichment using:
    - CLASSIFIER_MSG_DOCS (corpus awareness)
    - domain_terms.yaml (vocabulary grounding)
    """
    domain_terms_text = _load_domain_terms_text(logger)

    # Only high-signal inputs (avoid user/group/admin noise)
    planner_input = {
        "record_id": ptr_norm.get("record_id"),
        "title": ptr_norm.get("title"),
        "rec_type": ptr_norm.get("rec_type") or "PTR",
        "status": ptr_norm.get("status"),
        "severity": ptr_norm.get("severity"),
        "system": ptr_norm.get("system"),
        "category": ptr_norm.get("category"),
        "component": ptr_norm.get("component"),
        "type": ptr_norm.get("type"),
        "keywords": ptr_norm.get("keywords", []),
        "timeline_text": (ptr_norm.get("timeline_text") or "")[:24000],  # cap
    }

    user_prompt = (
        "### Documentation corpus summary (what you can retrieve)\n"
        f"{CLASSIFIER_MSG_DOCS}\n\n"
        "### Domain terminology reference (glossary)\n"
        f"{domain_terms_text}\n\n"
        "### Normalized PTR (high-signal fields)\n"
        f"{json.dumps(planner_input, ensure_ascii=False, indent=2)}\n\n"
        "Task:\n"
        "- Return EXACTLY 3 queries unless insufficient signal (then return 2 or 1).\n"
        "- Focus on: expected behavior, known failure modes, fix/workaround/validation procedures.\n"
        "- Avoid: assignment/user/group/date/administrative noise.\n"
    )

    messages = [
        {"role": "system", "content": PTR_DOC_QUERY_PLANNER_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    resp = llm.invoke(messages)
    raw = (resp.content or "").strip()

    data = _parse_json_strict(raw)

    out = {"queries": []}

    if isinstance(data, dict):
        qlist = data.get("queries", []) or []
        if isinstance(qlist, list):
            for q in qlist:
                if not isinstance(q, dict):
                    continue
                intent = (q.get("intent") or "").strip()
                query = (q.get("query") or "").strip()
                if not query:
                    continue
                out["queries"].append({"intent": intent or "unknown", "query": query})

    out["queries"] = out["queries"][: max(1, int(n_queries or 3))]
    return out


def _dedupe_docs(docs: list) -> list:
    """
    Deduplicate docs conservatively.

    IMPORTANT: Confluence pages are chunked into multiple index documents that share the same URL.
    Therefore, URL is NOT a safe dedupe key.

    Strategy:
    - Prefer dedupe by 'id' when present (each chunk has a unique id).
    - If id is missing, fallback to (url + content hash) which only removes exact duplicates.
    - Keep the highest (reranker_score, score) for any duplicate key.
    """
    import hashlib

    best = {}
    for d in (docs or []):
        if not isinstance(d, dict):
            continue

        doc_id = (d.get("id") or "").strip()
        url = (d.get("url") or "").strip()
        content = d.get("content") or ""

        if doc_id:
            key = ("id", doc_id)
        else:
            # Exact-duplicate fallback only
            h = hashlib.sha1(str(content).encode("utf-8", errors="ignore")).hexdigest()
            key = ("hash", url, h)

        prev = best.get(key)
        if prev is None:
            best[key] = d
            continue

        prev_key = (prev.get("reranker_score") or 0, prev.get("score") or 0)
        cur_key = (d.get("reranker_score") or 0, d.get("score") or 0)
        if cur_key > prev_key:
            best[key] = d

    out = list(best.values())
    out.sort(key=lambda x: ((x.get("reranker_score") or 0), (x.get("score") or 0)), reverse=True)
    return out


def _build_docs_context(doc_results: list, max_docs: int, logger=None):
    parts = []
    total_docs = len(doc_results or [])
    kept = 0

    for d in doc_results or []:
        if kept >= max_docs:
            break

        title = d.get("title", "")
        url = d.get("url", "")
        content = d.get("content", "")
        snippet = content

        if isinstance(snippet, str) and len(snippet) > 1200:
            snippet = snippet[:1200] + "…"

        parts.append(
            f"[Document]\nTitle: {title}\nURL: {url}\nContent: {snippet}"
        )
        kept += 1

    if logger:
        logger.info(
            "[PTR_DOCS_CONTEXT] "
            f"input_docs={total_docs}, "
            f"max_docs={max_docs}, "
            f"docs_in_context={kept}, "
            f"trimmed={max(0, total_docs - kept)}"
        )

    return "\n\n".join(parts)


def llm_summarize_ptr(llm, ptr_norm: dict, docs_context: str):
    sys = (
        "You are an assistant summarizing Winaproach PTR investigations for the SIT domain.\n"
        "Use the PTR record data as the primary source; use documentation snippets to enrich/validate the summary and suggest validation steps.\n"
        "If docs contradict PTR, mention the uncertainty explicitly.\n\n"
        "IMPORTANT:\n"
        "- If the PTR timeline or documentation mentions log locations, trace links, monitoring dashboards, or diagnostic commands, highlight them clearly.\n"
        "- If a fix/workaround references a Pull Request, commit, Gerrit review, Bitbucket link, or similar change artifact, call it out explicitly.\n"
        "- If multiple fixes or partial fixes exist, mention all with context.\n"
        "- If documentation describes expected behavior and the PTR describes behavior that differs, add an explicit 'Deviation from expected behavior' subsection.\n"
        "  In that subsection: state expected behavior (from docs), observed behavior (from PTR), and why they differ (only if supported by PTR/docs). Always cite docs.\n"
        "- If the PTR does NOT explicitly contain a fix/workaround, do NOT invent one.\n"
        "  Instead add a section 'Investigation approach (doc-driven)' with 5–8 concrete steps derived from the retrieved documentation.\n"
        "  Each step MUST include at least one doc citation. If no doc supports a step, do not include it.\n\n"
        "Return a clean, structured MARKDOWN answer with sections:\n"
        "- Overview (1–2 lines)\n"
        "- Problem summary\n"
        "- Impact\n"
        "- Root cause (if known)\n"
        "- Deviation from expected behavior (only if docs define expected behavior)\n"
        "- Issue reproduction steps (if mentioned)\n"
        "- Links confirming the issue (ONLY direct links already present in the PTR record, such as logs, traces, screenshots, IR/Jira references)\n"
        "- Fix / Workaround (include PR / change links if available)\n"
        "- Investigation approach (doc-driven) (only if fix/workaround is missing)\n"
        "- Validation / How to confirm resolution (observable checks AFTER a fix/workaround, such as logs, traces, commands, or expected outputs — ONLY if explicitly mentioned)\n"
        "- Include a short 'Conclusion / Confidence' section at the end ONLY if the summary relies on inferred documentation, lacks an explicit fix, or highlights deviation from expected behavior.\n"
        "- References (PTR/IR/Jira/URLs if present)\n\n"
        "Always include doc citations as: [Document: <title> | URL: <url>] when you use doc info.\n"
        "For PTR references, use ONLY the provided 'record_url' field from the PTR JSON.\n"
        "Win@proach is internal; do NOT guess/invent PTR URLs or treat random strings as PTR links.\n"
        "Do not invent titles, URLs, PR numbers, or links.\n"
    )

    domain_terms_text = _load_domain_terms_text()

    user = (
        "PTR (normalized):\n" + json.dumps(ptr_norm, ensure_ascii=False) + "\n\n"
        "Domain terminology reference (glossary):\n"
        + (domain_terms_text[:12000] if domain_terms_text else "(none)") + "\n\n"
        "Documentation context (retrieved):\n" + (docs_context or "(none)")
    )

    resp = llm.invoke([
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ])
    return str(resp.content or "").strip()


def ptr_flow(
    query,
    llm,
    embeddings,
    search_client_docs,
    retrieve_docs_results_fn,
    index_docs_name,
    logger,
):
    ptr_id = extract_ptr_id(query)
    if not ptr_id:
        return None

    logger.info(f"[PTR_SUMMARY] Fetching PTR {ptr_id} via MCP")

    # 1) Fetch
    mcp_res = fetch_ptr_fft(
        ptr_id,
        os.getenv("MCP_URL"),
        os.getenv("MCP_USER"),
        os.getenv("MCP_PASS"),
        os.getenv("MCP_EMAIL"),
    )

    # 2) Normalize (YOU ALREADY FIXED THIS 👍)
    ptr_norm = normalize_ptr(ptr_id, mcp_res)

    # 3) Ask LLM for 3 high-signal doc queries
    plan = llm_plan_doc_queries(
        llm=llm,
        ptr_norm=ptr_norm,
        logger=logger,
        n_queries=3,
    )
    doc_queries = plan.get("queries", []) or []

    if logger:
        logger.info("[PTR_DOC_QUERIES] LLM decided the following documentation queries:")
        for i, q in enumerate(doc_queries, start=1):
            logger.info(
                f"  {i}. intent={q.get('intent')} | query=\"{q.get('query')}\""
        )

    # 4) Azure Search: retrieve docs for each query, then merge + dedupe by URL
    top_docs_per_q = int(os.getenv("TOP_AISEARCH_DOC_RESULTS", "200"))
    max_docs_total = int(os.getenv("PTR_MAX_DOCS_TOTAL", "150"))

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
            if logger:
                logger.warning(f"[PTR_DOCS] retrieval failed for query='{qtext}': {e}")

    logger.info(f"[PTR_DOCS] Docs before de-duplication: {len(merged_docs)}")

    merged_docs = _dedupe_docs(merged_docs)

    logger.info(f"[PTR_DOCS] Docs after de-duplication: {len(merged_docs)}")

    docs_context = _build_docs_context(
        merged_docs,
        max_docs=max_docs_total,
        logger=logger
    )

    # 5) Final LLM summary grounded in PTR + docs
    answer_md = llm_summarize_ptr(llm, ptr_norm, docs_context)

    sources = []
    for d in (merged_docs or [])[:max_docs_total]:
        sources.append(
            {
                "title": d.get("title", ""),
                "url": d.get("url", ""),
                "score": d.get("score", None),
                "reranker_score": d.get("reranker_score", None),
                "lastModified": d.get("lastModified", ""),
            }
        )

    return {"answer": answer_md, "sources": sources}
