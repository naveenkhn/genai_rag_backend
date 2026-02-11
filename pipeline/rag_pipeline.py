import os
import json
from datetime import datetime
from dotenv import load_dotenv
from azure.search.documents.models import VectorizedQuery
import yaml

from core.logging_config import logger
from core.utils import is_chitchat
from services.classifier import classify_and_rewrite_query
from services.ptr_analysis import ptr_flow
from services.similar_ptr_analysis import ptr_similar_flow
from retrieval.retrievers import retrieve_code_results, retrieve_docs_results, retrieve_mixed_results, retrieve_ptr_results
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from prompts.prompts import (
    CLASSIFIER_MSG_DOCS,
    CLASSIFIER_MSG_CODE,
    SYSTEM_MSG_DOCS,
    SYSTEM_MSG_CODE,
    SYSTEM_MSG_MIXED,
)
from retrieval.retrievers import retrieve_mixed_results_threaded

# Load environment variables
load_dotenv()

# Config values
K_TOP_DOCS = int(os.getenv("TOP_AISEARCH_DOC_RESULTS"))
K_TOP_CODE = int(os.getenv("TOP_AISEARCH_CODE_RESULTS"))

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
INDEX_DOCS = os.getenv("INDEX_DOCS")
INDEX_CODE = os.getenv("INDEX_CODE")
INDEX_PTRS = os.getenv("INDEX_PTRS")  # e.g., ptrs-index-v1

credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))

search_client_docs = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_DOCS,
    credential=credential
)
search_client_code = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_CODE,
    credential=credential
)

search_client_ptrs = None
if INDEX_PTRS:
    search_client_ptrs = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_PTRS,
        credential=credential
    )

embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

def run_rag_pipeline(query, history):
    """
    Executes the RAG pipeline:
      1. Classify the query.
      2. Embeds the query.
      3. Retrieves top docs from relevant indexes.
      4. Builds context string and messages.
      5. Calls the LLM.
      6. Returns dict with answer and sources.
    """
    # Step 0: handle chit-chat queries quickly
    if is_chitchat(query):
        answer = llm.invoke([{"role": "user", "content": query}])
        return {"answer": str(answer.content), "sources": []}
    
    # PTR summarize mode (live fetch via MCP + docs enrichment)
    ptr_result = ptr_flow(
        query=query,
        llm=llm,
        embeddings=embeddings,
        search_client_docs=search_client_docs,
        retrieve_docs_results_fn=retrieve_docs_results,
        index_docs_name=INDEX_DOCS,
        logger=logger,
    )
    if ptr_result:
        return ptr_result

    # PTR similar issues mode (search PTR index + docs enrichment)
    if search_client_ptrs is not None:
        ptr_similar_result = ptr_similar_flow(
            query=query,
            llm=llm,
            embeddings=embeddings,
            search_client_ptrs=search_client_ptrs,
            retrieve_ptr_results_fn=retrieve_ptr_results,
            index_ptrs_name=INDEX_PTRS,
            search_client_docs=search_client_docs,
            retrieve_docs_results_fn=retrieve_docs_results,
            index_docs_name=INDEX_DOCS,
            logger=logger,
        )
        if ptr_similar_result:
            return ptr_similar_result


    # Collect last 3 past user queries
    past_user_queries = [msg['content'] for msg in history if msg['role'] == 'user'][-3:]
    # Collect last assistant answer (if available)
    last_answer = None
    for msg in reversed(history):
        if msg.get('role') == 'assistant':
            last_answer = msg.get('content')
            break
    past_context = {
        "past_queries": past_user_queries,
        "last_answer": last_answer
    }

    # Step 1: classify and rewrite query
    classification, rewritten_query, detected_repos = classify_and_rewrite_query(
        query, past_context, llm=llm, logger=logger, embeddings=embeddings
    )
    logger.debug(f"Detected repos: {detected_repos}")
    logger.debug(f"Classifier decision: {classification}")
    logger.debug(f"Rewritten query: {rewritten_query}")

    # Step 2: embed rewritten query
    query_embedding = embeddings.embed_query(rewritten_query)

    docs = []
    context_parts_docs = []
    code_docs = []
    context_parts_code = []

    # Step 3: retrieve top docs from relevant indexes
    if classification == "docs_question":
        docs, context_parts_docs = retrieve_docs_results(
            search_client_docs, rewritten_query, query_embedding,
            K_TOP_DOCS, INDEX_DOCS, logger
        )
    else:
        # Handle both 'code_question' and 'mixed' as mixed retrieval
        # if classification == "code_question":
        #     code_docs, context_parts_code = retrieve_code_results(search_client_code, rewritten_query, query_embedding, K_TOP_CODE, INDEX_CODE, logger, detected_repos)
        docs_docs, code_docs, context_parts_docs, context_parts_code = retrieve_mixed_results_threaded(
            search_client_docs, search_client_code, rewritten_query, query_embedding,
            K_TOP_DOCS, K_TOP_CODE, INDEX_DOCS, INDEX_CODE, logger, detected_repos
        )
        docs.extend(docs_docs)
        docs.extend(code_docs)
        # DEBUG: removed JSON dump of retrieved code results

    # Step 4: build prompt (system message depends on classification)
    if classification == "docs_question":
        system_msg = SYSTEM_MSG_DOCS
    # elif classification == "code_question":
    #     system_msg = SYSTEM_MSG_CODE
    else:
        system_msg = SYSTEM_MSG_MIXED

    # Build repository context from repo_info for detected repos
    repo_context_block = ""
    if detected_repos:
        try:
            from services.repo_info import REPO_INFO
            detailed_summaries = []
            for repo in detected_repos[:3]:  # limit to 3 repos max
                info = REPO_INFO.get(repo)
                if info:
                    summary_text = (
                        f"\n[Repository: {repo}]\n"
                        f"Purpose:\n{info.get('purpose', '')}\n\n"
                        f"Main Components:\n" + "\n".join(f"• {c}" for c in info.get('main_components', [])) + "\n\n"
                        f"Supported Messages:\n" + (
                            "\n".join(f"• {k}: {v}" for k, v in info.get('supported_messages', {}).items())
                            if info.get('supported_messages') else "• None\n"
                        ) + "\n\n"
                        f"Core Responsibilities:\n" + "\n".join(f"• {r}" for r in info.get('core_responsibilities', [])) + "\n\n"
                        f"Interacting Repositories: {', '.join(info.get('interacting_repos', [])) or 'None'}\n"
                    )
                    detailed_summaries.append(summary_text)
            if detailed_summaries:
                repo_context_block = (
                    "\n\n[Repository Context]\nThe following repositories were identified as relevant for this query:\n"
                    + "\n".join(detailed_summaries)
                )
        except Exception as e:
            logger.exception(f"Failed to build repository context: {e}")

    # === Load domain terms and workflows ===
    def load_yaml_file(file_path):
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load {file_path}: {e}")
            return {}

    domain_terms = load_yaml_file("data/domain_terms.yaml")
    workflows = load_yaml_file("data/workflows.yaml")

    # Include every line from YAMLs
    domain_context = ""
    if isinstance(domain_terms, (dict, list)):
        try:
            domain_context = yaml.dump(domain_terms, sort_keys=False, allow_unicode=True)
        except Exception as e:
            logger.warning(f"Failed to dump domain_terms.yaml: {e}")

    workflow_context = ""
    if isinstance(workflows, (dict, list)):
        try:
            workflow_context = yaml.dump(workflows, sort_keys=False, allow_unicode=True)
        except Exception as e:
            logger.warning(f"Failed to dump workflows.yaml: {e}")

    extended_context_block = ""
    if domain_context:
        extended_context_block += "\n\n### Domain Terms Context\n" + domain_context
    if workflow_context:
        extended_context_block += "\n\n### Workflow Context\n" + workflow_context

    context = ""
    if context_parts_docs:
        context += "### Specifications \n" + "\n\n".join(context_parts_docs) + "\n\n"
    if context_parts_code:
        context += "### Code \n" + "\n\n".join(context_parts_code)

    # Compose final system message with repo context and extended context
    full_system_msg = system_msg + repo_context_block + extended_context_block

    messages = [{"role": "system", "content": full_system_msg}]
    if history:
        messages.extend(history[-3:])
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    # Log the full messages being passed to the LLM in a readable format
    logger.debug("=== Messages passed to LLM (formatted) ===")
    for i, msg in enumerate(messages):
        logger.debug(f"Message {i} - role: {msg['role']}, content: {msg['content'][:100]}...")
    logger.debug("=== End of messages ===")

    # Step 5: context preview logging
    log_context_preview(context, context_parts_docs, context_parts_code, rewritten_query)
    answer = llm.invoke(messages)

    # Log the raw answer content returned from the LLM with clear separators
    logger.debug("=== Raw answer content from LLM ===")
    logger.debug(f"LLM answer (truncated): {str(answer.content)[:200]}...")
    logger.debug("=== End of LLM answer ===")

    return {
        "answer": str(answer.content)
    }

def log_context_preview(context, context_parts_docs, context_parts_code, rewritten_query=None):
    if rewritten_query:
        logger.debug(f"[Rewritten Query] {rewritten_query}")
    logger.debug("=== Context Preview ===")
    logger.debug(f"Total context length: {len(context)} characters")
    logger.debug(f"Total document parts: {len(context_parts_docs)}")
    logger.debug(f"Total code parts: {len(context_parts_code)}")


if __name__ == "__main__":
    test_query = "what is saleable configuration ?"
    test_history = [{"role": "user", "content": "Hello"}]

    logger.debug("=== Running RAG pipeline test ===")
    result = run_rag_pipeline(test_query, test_history)
    logger.debug(f"Answer: {result['answer']}")
