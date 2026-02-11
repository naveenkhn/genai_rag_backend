import numpy as np
from services.repo_info import REPO_INFO
from typing import Tuple, List
from prompts.prompts import CLASSIFIER_MSG_DOCS, CLASSIFIER_MSG_CODE

CLASSIFIER_INSTRUCTIONS = (
    "You are a Seat Management (SIT) domain-aware classifier and query rewriter.\n"
    "You have access to both documentation (Confluence-based specifications, processes, and rules) "
    "and source code (repositories containing Seat Management logic). Your task is to identify "
    "what type of query the user is asking, rewrite it clearly, and infer relevant repositories.\n\n"

    "### Background Context\n"
    "Below are summaries describing the two types of knowledge sources you can rely on.\n\n"
    "#### Documentation (Confluence Knowledge Base):\n"
    f"{CLASSIFIER_MSG_DOCS}\n\n"
    "#### Code Repositories:\n"
    f"{CLASSIFIER_MSG_CODE}\n\n"

    "### Task 1 – Classification\n"
    "Classify the query into exactly one of the following categories:\n"
    "- 'docs_question': queries about specifications, functional flows, configurations, or business rules.\n"
    "- 'mixed': queries that require both functional and technical understanding — for example, where the user asks how a documented rule, process, or configuration is realized in code.\n\n"

    "Classification guidance:\n"
    "- **docs_question** → when the query focuses on *what*, *why*, or *functional aspects*, such as specifications, flows, rules, or message purposes. These questions usually seek explanations of system behavior without referring to specific code or implementation.\n"
    "- **mixed** → when the query combines both functional and technical intent — e.g., questions containing both 'what' or 'why' and 'how', or referencing behavioral/implementation terms like 'logic', 'algorithm', 'mechanism', 'workflow', or 'implementation'. These require both documentation and code understanding.\n"
    "- Always classify as **mixed** if the query mentions any of: 'algorithm', 'logic', 'mechanism', 'workflow', 'implementation', 'processing', or 'how it works'.\n"
    "- Always classify as **mixed** if the query includes or references code snippets, file paths, repo names, stack traces, or error messages used for investigation.\n"
    "- Prefer 'mixed' if any technical or behavioral term appears or if uncertain.\n"

    "### Task 2 – Query Rewriting\n"
    "Rewrite the query so that it becomes fully self-contained and contextually complete for retrieval.\n\n"

    "Coreference resolution rule:\n"
    "- When the current query uses pronouns or deictic phrases such as 'it', 'them', 'this flow', or 'that', "
    "resolve these references using only **one** past query — specifically, the *most recent* one that is semantically relevant.\n"
    "- You may also use the **last assistant answer** strictly for resolving pronouns or references that depend on items enumerated or explained in that answer.\n"
    "- Example: if the last answer listed database tables or COM/OTF parameters, and the user asks 'where is this updated?', interpret 'this' using the last answer.\n"
    "- Do NOT use the last answer to introduce new meaning, combine unrelated topics, or expand the scope of the user's intent.\n"
    "- Do not merge multiple topics from different past turns unless the user explicitly asks to compare or combine them.\n\n"

    "No cross-topic merging:\n"
    "- If multiple past queries seem related, pick the most recent one and ignore all others.\n"
    "- Never conjoin or blend subjects from two or more separate queries.\n\n"

    "Return-only-one-topic rewrite:\n"
    "- The rewritten query must reference a single, concrete subject or flow.\n"
    "- If ambiguity remains, prefer the immediately previous applicable turn as the reference.\n\n"

    "Other rewriting guidelines:\n"
    "- Focus on semantic continuity — ensure the rewritten query conveys the *complete meaning* without depending on prior context.\n"
    "- Do **not** expand abbreviations or domain terms unless absolutely necessary for clarity.\n"
    "- Preserve domain-specific terminology (SMPREQ, SRBRCQ, seatmap, rebuild, etc.) exactly as used in the corpus.\n"
    "- Keep the rewritten query concise, factual, and directly answerable by retrieval.\n\n"

    "Examples:\n"
    "  Q1: explain SRBRCQ implementation\n"
    "  Q2: explain characteristic template\n"
    "  Q3: which edifact message controls this flow?\n"
    "  → Rewritten Q3: which EDIFACT message controls characteristic template?\n\n"

    "### Task 3 – Repository Inference\n"
    "Based on your understanding of the rewritten query and repository purposes (provided in code section), "
    "return the most relevant repository or repositories as an array (e.g., ['stm_main']).\n\n"
    "Repository selection rules:\n"
    "- By default, return exactly **one** repository.\n"
    "- If the query explicitly mentions database structures, tables, SQL, or schema, include 'dba' alongside the main repo (e.g., ['stm_main', 'dba']).\n"
    "- If the query references a specific EDIFACT message (e.g., SRBRCQ, ISELGQ, SMPREQ) that appears in multiple repositories within the provided repo information, "
    "return **all** those repositories that handle the message (e.g., ['stm_main', 'ste']).\n"
    "- Prefer the primary functional repository (STM, STE, FMT, DFR, etc.) that owns the logic or flow described.\n"
    "- Avoid returning unrelated or auxiliary repositories unless the query clearly spans multiple systems.\n\n"
    "Return only a valid JSON object with the following fields:\n"
    "{\n"
    "  'classification': 'docs_question' | 'mixed',\n"
    "  'rewritten_query': '<rewritten query>',\n"
    "  'repos': ['repo1'] or ['repo1', 'dba']\n"
    "}\n\n"
    "Strictly output JSON — no markdown, comments, or explanations."
)


from typing import Tuple, List

def classify_and_rewrite_query(query: str, past_context: dict, llm, logger, embeddings=None) -> Tuple[str, str, List[str]]:
    """
    Classifies the query into 'code_question', 'docs_question', or 'mixed' and rewrites the query if needed.
    Returns (classification, rewritten_query, repos).
    past_context: dict with keys 'past_queries' (list of up to 3 previous user queries) and 'last_answer' (assistant's last answer or None)
    """
    # The system prompt already embeds both classifier messages and repo summaries via CLASSIFIER_INSTRUCTIONS
    system_prompt = CLASSIFIER_INSTRUCTIONS
    # Load domain terms context for better domain-aware classification
    import yaml
    try:
        with open("data/domain_terms.yaml", "r") as f:
            domain_terms = yaml.safe_load(f)
        if isinstance(domain_terms, (dict, list)):
            domain_context = yaml.dump(domain_terms, sort_keys=False, allow_unicode=True)
            system_prompt += "\n\n### Domain Context (for classification reference)\n" + domain_context
    except Exception as e:
        logger.warning(f"Could not load domain_terms.yaml for classifier: {e}")

    # Extract past_queries and last_answer from past_context
    past_queries = past_context.get("past_queries", [])
    last_answer = past_context.get("last_answer", None)

    user_msg_lines = []
    user_msg_lines.append(f"Current query: {query}")
    user_msg_lines.append(f"Past queries: {past_queries}")
    if last_answer:
        user_msg_lines.append(f"Last assistant answer: {last_answer}")
    user_msg = "\n".join(user_msg_lines)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    logger.debug(f"LLM classify_and_rewrite request: {messages}")
    try:
        response = llm.invoke(messages)
        raw = (response.content or "").strip()
        logger.info(f"[CLASSIFIER RAW OUTPUT] {raw}")
        import json, re, ast
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
        except Exception:
            try:
                data = ast.literal_eval(cleaned)
            except Exception:
                logger.error(f"[CLASSIFIER PARSE FAILED] Could not parse response: {cleaned}")
                return "docs_question", query, ["stm_main"]

        classification = str(data.get("classification", "docs_question")).lower()
        rewritten_query = data.get("rewritten_query", query) or query
        repos = data.get("repos", [])

        # Disabled forced mixed classification — rely entirely on LLM classification.
        # --- Detect code-related investigation context ---
        # code_signals = [
        #     "def ", "class ", "void ", "int ", "std::", "traceback", "exception",
        #     "error:", "file:", "repo", "symbol", "line", "stacktrace",
        #     "```", ".cpp", ".h", ".py", ".log", "opengrok", "function", "method"
        # ]
        # combined_text = f"{query.lower()} {rewritten_query.lower()}"
        # if any(sig in combined_text for sig in code_signals):
        #     classification = "mixed"

        # Fallbacks / guards
        if classification not in {"docs_question", "mixed"}:
            classification = "docs_question"
        if not isinstance(repos, list):
            repos = []
        # --- Repo confidence logic temporarily disabled ---
        # repo_confidence = []
        # if embeddings is not None:
        #     repo_confidence = compute_repo_confidence(rewritten_query, embeddings, logger)
        #     logger.debug(f"[REPO_CONFIDENCE_SCORES] {repo_confidence}")
        # # If LLM did not provide repos, fill with top 2-3 from repo_confidence
        # if not repos:
        #     if repo_confidence:
        #         repos = [r for r, _ in repo_confidence[:3]]
        #     else:
        #         repos = ["stm_main"]
        # # Always log repo confidence scores for debugging
        # if repo_confidence:
        #     logger.info(f"[REPO_CONFIDENCE_RANKED] {repo_confidence}")
        return classification, rewritten_query, repos
    except Exception:
        logger.exception("LLM invocation failed; defaulting to docs_question")
        return "docs_question", query, ["stm_main"]


# --- Repo confidence scoring using embeddings ---
def compute_repo_confidence(query: str, embeddings, logger):
    """
    Compute semantic similarity between the user query and each repo description in REPO_INFO.
    Returns a sorted list of (repo_name, score) tuples.
    """
    try:
        query_vec = np.array(embeddings.embed_query(query))
        repo_scores = {}

        for repo, info in REPO_INFO.items():
            ref_text = " ".join([
                info.get("purpose", ""),
                " ".join(info.get("core_responsibilities", [])),
                " ".join(info.get("supported_messages", []))
            ])
            repo_vec = np.array(embeddings.embed_query(ref_text))
            # Cosine similarity
            sim = float(np.dot(query_vec, repo_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(repo_vec)))
            repo_scores[repo] = sim

        sorted_repos = sorted(repo_scores.items(), key=lambda x: x[1], reverse=True)

        if logger:
            logger.debug("[REPO_CONFIDENCE] " + ", ".join([f"{r}:{s:.2f}" for r, s in sorted_repos]))

        return sorted_repos
    except Exception as e:
        if logger:
            logger.exception(f"compute_repo_confidence failed: {e}")
        return []
