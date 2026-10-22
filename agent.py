"""Agentic aviation maintenance assistant built with LangGraph and Ollama qwen2.5:3b.

Features:
  - Input abbreviation normalization via `utils.normalizer.normalize_text`
  - Tool binding: `search_records` and `get_record` from `search_engine`
  - Strict Refusal Threshold Logic (refuses queries with top match score < 0.35)
  - Explicit record ID citations (e.g. `[Record #100062]`)
  - Execution trace recording tool calls and intermediate decisions
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from search_engine import detect_recurring_defects, get_record, search_records
from utils.normalizer import normalize_text


REFUSAL_THRESHOLD = 0.35
REFUSAL_MESSAGE = "Insufficient evidence in historical logbooks to provide a confident fix."


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------
@tool
def search_records_tool(query: str) -> str:
    """Search aviation maintenance records using hybrid BM25 and FAISS search."""
    results = search_records(query, top_k=5)
    return json.dumps(results, indent=2)


@tool
def get_record_tool(record_id: str) -> str:
    """Retrieve full details of a specific aviation maintenance log record by IDENT ID."""
    rec = get_record(record_id)
    if not rec:
        return f"Record {record_id} not found."
    return json.dumps(rec, indent=2)


TOOLS = [search_records_tool, get_record_tool]


# ---------------------------------------------------------------------------
# LLM Initialization with Cloud Gemini & Local Ollama Support
# ---------------------------------------------------------------------------
DEFAULT_GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-3.8-flash",
]


class GeminiLLM:
    """Lightweight LangChain-compatible wrapper for Google Gemini API with automatic model fallback."""

    def __init__(self, api_key: str | list[str], model: str | None = None) -> None:
        if isinstance(api_key, str):
            self.api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        else:
            self.api_keys = list(api_key)

        primary_model = (model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")).strip()
        self.model = primary_model

        # Build candidate fallback models list starting with primary model
        configured_fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "")
        if configured_fallbacks:
            fallback_list = [m.strip() for m in configured_fallbacks.split(",") if m.strip()]
        else:
            fallback_list = DEFAULT_GEMINI_MODELS

        seen = set()
        self.candidate_models: list[str] = []
        for m in [primary_model] + fallback_list:
            if m not in seen:
                seen.add(m)
                self.candidate_models.append(m)

    def invoke(self, messages: list[Any]) -> AIMessage:
        prompt_lines = []
        for msg in messages:
            msg_type = getattr(msg, "type", "user")
            content = getattr(msg, "content", "")
            if msg_type == "system":
                prompt_lines.append(f"System Instructions:\n{content}\n")
            elif msg_type == "human":
                prompt_lines.append(f"User Problem:\n{content}\n")
            elif msg_type == "tool":
                prompt_lines.append(f"Retrieved Historical Records:\n{content}\n")
            elif content:
                prompt_lines.append(f"{content}\n")

        prompt_text = "\n".join(prompt_lines)
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 3072},
        }).encode("utf-8")

        last_error = None
        for key_idx, current_api_key in enumerate(self.api_keys):
            for model_idx, target_model in enumerate(self.candidate_models):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent"
                try:
                    req = urllib.request.Request(
                        url,
                        data=payload,
                        headers={"Content-Type": "application/json", "x-goog-api-key": current_api_key},
                    )
                    with urllib.request.urlopen(req, timeout=40) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text_parts = [p.get("text", "") for p in parts if "text" in p]
                            text = "\n".join(text_parts).strip()
                            if text:
                                if target_model != self.model:
                                    print(f"Fallback active: Successfully generated response using model '{target_model}'.")
                                return AIMessage(content=text)
                except urllib.error.HTTPError as http_err:
                    last_error = http_err
                    status = http_err.code
                    print(
                        f"Warning: Model '{target_model}' returned HTTP {status} "
                        f"(Rate limit / quota exhausted or model unavailable). "
                        f"Cascading to next fallback model..."
                    )
                    continue
                except Exception as exc:
                    last_error = exc
                    print(f"Warning: Model '{target_model}' failed ({exc}). Trying next fallback model...")
                    continue

        print(f"All Gemini models exhausted or failed ({last_error}); attempting local Ollama fallback...")
        try:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
            chat_model = ChatOllama(model=model_name, base_url=base_url, temperature=0.1)
            return chat_model.invoke(messages)
        except Exception as o_exc:
            print(f"Ollama fallback error: {o_exc}")

        return AIMessage(
            content="API rate limits exhausted across all available Gemini models and local Ollama is offline. Please retry in a few moments."
        )


def get_llm():
    """Initialize primary LLM (Google Gemini cloud if configured, else local ChatOllama)."""
    google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        try:
            model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
            print(f"Initializing Google Gemini API ({model_name}) with multi-model fallback...")
            return GeminiLLM(api_key=google_api_key, model=model_name)
        except Exception as exc:
            print(f"Warning: Gemini init failed ({exc}), falling back to Ollama...")

    # Primary local model: qwen2.5:3b via Ollama
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    chat_model = ChatOllama(model=model_name, base_url=base_url, temperature=0.1)
    return chat_model.bind_tools(TOOLS)


LLM = get_llm()


# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    raw_query: str
    normalized_query: str
    search_results: list[dict[str, Any]]
    recurring_defect_info: dict[str, Any]
    refused: bool
    answer: str
    cited_records: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------
def normalize_input_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw user input using aviation abbreviation dictionary."""
    raw = state.get("raw_query", "")
    norm = normalize_text(raw)
    trace = list(state.get("tool_trace", []))
    trace.append({
        "step": "normalize_input",
        "tool": "utils.normalizer.normalize_text",
        "input": raw,
        "output": norm,
    })
    return {"normalized_query": norm, "tool_trace": trace}


def retrieve_and_evaluate_node(state: AgentState) -> dict[str, Any]:
    """Execute hybrid search, detect recurring defects, and enforce refusal logic."""
    query = state.get("normalized_query") or state.get("raw_query", "")
    results = search_records(query, top_k=5)

    trace = list(state.get("tool_trace", []))
    top_score = results[0]["score"] if results else 0.0

    # Call detect_recurring_defects on the search results before generating the answer
    recurring_info = detect_recurring_defects(results)

    trace.append({
        "step": "retrieve_records",
        "tool": "search_records",
        "input": {"query": query},
        "retrieved_count": len(results),
        "top_score": top_score,
        "threshold": REFUSAL_THRESHOLD,
        "recurring_defect_info": recurring_info,
    })

    # Strict Refusal Threshold Logic:
    # If no results or highest match score is below 0.35, refuse immediately.
    if not results or top_score < REFUSAL_THRESHOLD:
        trace.append({
            "step": "refusal_evaluation",
            "decision": "REFUSED",
            "reason": f"Top match score {top_score:.4f} is below threshold {REFUSAL_THRESHOLD}",
        })
        return {
            "search_results": results,
            "recurring_defect_info": {"is_recurring": False},
            "refused": True,
            "answer": REFUSAL_MESSAGE,
            "cited_records": [],
            "tool_trace": trace,
        }

    trace.append({
        "step": "refusal_evaluation",
        "decision": "ACCEPTED",
        "top_score": top_score,
    })
    return {
        "search_results": results,
        "recurring_defect_info": recurring_info,
        "refused": False,
        "tool_trace": trace,
    }


def should_refuse(state: AgentState) -> str:
    """Conditional router based on refusal status."""
    return "refused" if state.get("refused", False) else "generate"


def generate_node(state: AgentState) -> dict[str, Any]:
    """Generate structured response citing retrieved logbook records."""
    raw_query = state.get("raw_query", "")
    norm_query = state.get("normalized_query", "")
    results = state.get("search_results", [])
    trace = list(state.get("tool_trace", []))

    recurring_info = state.get("recurring_defect_info", {})
    recurrence_context = ""
    if recurring_info.get("is_recurring"):
        comp = recurring_info.get("component", "UNKNOWN")
        cnt = recurring_info.get("count", 0)
        symptoms_str = "; ".join(recurring_info.get("symptoms", [])[:3])
        recurrence_context = (
            f"\n\nFLEET RECURRENCE CONTEXT:\n"
            f"- Subsystem [{comp}] has {cnt} documented occurrences in historical logbooks.\n"
            f"- Documented failure symptoms: {symptoms_str}.\n"
            f"- Factor this recurring defect pattern into your diagnostic breakdown."
        )

    system_prompt = (
        "You are an expert aviation maintenance technician (A&P) and MRO diagnostic specialist.\n"
        "Your task: diagnose the discrepancy and provide verified corrective actions based STRICTLY on the retrieved historical logbooks.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. DO NOT include generic conversational filler (such as 'Based on the retrieved historical maintenance records...'). Jump directly into the technical diagnosis.\n"
        "2. Explain the SPECIFIC MECHANICAL DEFECTS and symptoms in detail—do not just recite numbers or IDs. Detail how the fault manifested (e.g., ground run pressure spikes, relief valve bypass, cylinder compression leaks, transducer drift) and why.\n"
        "3. Detail the exact historical repairs performed (e.g., adjusting oil pressure relief valve, torquing fittings, servicing oil cooler, replacing gaskets).\n"
        "4. Keep the response focused, structured, and informative (concise yet packed with technical troubleshooting substance).\n"
        "5. For EVERY finding, symptom, and repair action, explicitly cite the verified logbook record using '[Record #<ID>]' (e.g., [Record #100062]).\n\n"
        "FORMATTING STRUCTURE:\n"
        "### 1. Problem & Malfunction Breakdown\n"
        "Explain the specific operating conditions, failure symptoms, and affected assemblies cited in [Record #XXXXXX].\n\n"
        "### 2. Historical Corrective Actions & Root Causes\n"
        "Detail the physical repairs and root cause discoveries made in past logs [Record #XXXXXX].\n\n"
        "### 3. Recommended Flight-Line Troubleshooting Steps\n"
        "Provide prioritized, step-by-step diagnostic actions for the maintenance technician."
    )

    call_id = f"call_{uuid.uuid4().hex[:8]}"

    # Tool invocation message chain
    ai_tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_records_tool",
            "args": {"query": norm_query},
            "id": call_id,
            "type": "tool_call",
        }],
    )

    # Tool output containing the retrieved records
    tool_output_payload = [
        {
            "ident": r["ident"],
            "problem": r["problem_normalized"],
            "action": r["action_normalized"],
            "confidence_score": r["score"],
        }
        for r in results
    ]

    tool_msg = ToolMessage(
        content=json.dumps(tool_output_payload, indent=2),
        tool_call_id=call_id,
        name="search_records_tool",
    )

    user_query_payload = f"Reported Discrepancy: {raw_query}{recurrence_context}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query_payload),
        ai_tool_call,
        tool_msg,
    ]

    response = LLM.invoke(messages)

    # Check if LLM requested an additional tool call (e.g. get_record_tool)
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            t_name = tc.get("name", "")
            t_args = tc.get("args", {})
            t_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")

            if "get_record" in t_name:
                rec_id = t_args.get("record_id", "")
                rec_data = get_record(rec_id)
                trace.append({
                    "step": "get_record_tool",
                    "tool": "get_record",
                    "input": {"record_id": rec_id},
                    "found": rec_data is not None,
                })
                sub_tool_msg = ToolMessage(
                    content=json.dumps(rec_data, indent=2) if rec_data else f"Record {rec_id} not found.",
                    tool_call_id=t_id,
                    name=t_name,
                )
                messages.extend([response, sub_tool_msg])
                response = LLM.invoke(messages)
                break

    answer_text = response.content if hasattr(response, "content") else str(response)

    # Extract all cited record IDs from answer text (e.g. [Record #100062] or Record #100062)
    cited_ids_raw = re.findall(r"(?:Record\s*#?|\[Record\s*#?)(\d+)", answer_text, re.IGNORECASE)
    cited_ids = set(int(cid) for cid in cited_ids_raw)

    # Match cited IDs against retrieved records; fallback to top search results if none parsed
    cited_records = [r for r in results if r["ident"] in cited_ids]
    if not cited_records and results:
        cited_records = results[:2]

    trace.append({
        "step": "generate_answer",
        "model": "qwen2.5:3b",
        "cited_ids": [r["ident"] for r in cited_records],
    })

    return {
        "answer": answer_text,
        "cited_records": cited_records,
        "tool_trace": trace,
    }


# ---------------------------------------------------------------------------
# Build the LangGraph Workflow
# ---------------------------------------------------------------------------
def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("normalize", normalize_input_node)
    workflow.add_node("retrieve", retrieve_and_evaluate_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("normalize")
    workflow.add_edge("normalize", "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        should_refuse,
        {
            "refused": END,
            "generate": "generate",
        },
    )

    workflow.add_edge("generate", END)
    return workflow.compile()


AGENT_GRAPH = build_agent_graph()


def run_agent(query: str) -> dict[str, Any]:
    """Execute the full agentic pipeline for a query.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
          - answer: str
          - cited_records: list[dict]
          - tool_trace: list[dict]
    """
    initial_state: AgentState = {
        "raw_query": query,
        "normalized_query": "",
        "search_results": [],
        "recurring_defect_info": {"is_recurring": False},
        "refused": False,
        "answer": "",
        "cited_records": [],
        "tool_trace": [],
    }

    final_state = AGENT_GRAPH.invoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "cited_records": final_state.get("cited_records", []),
        "tool_trace": final_state.get("tool_trace", []),
        "recurring_defect_info": final_state.get("recurring_defect_info", {"is_recurring": False}),
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING AGENTIC BACKEND WITH VALID QUERY")
    print("=" * 60)
    valid_query = "ENG OIL PRESS LOW"
    res1 = run_agent(valid_query)
    print(f"\nQuery: '{valid_query}'")
    print(f"\nAnswer:\n{res1['answer']}")
    print(f"\nCited Record IDs: {[r['ident'] for r in res1['cited_records']]}")
    print(f"Tool Trace Steps: {[t['step'] for t in res1['tool_trace']]}")

    print("\n" + "=" * 60)
    print("TESTING REFUSAL THRESHOLD WITH IRRELEVANT QUERY")
    print("=" * 60)
    irrelevant_query = "how to bake a chocolate cake"
    res2 = run_agent(irrelevant_query)
    print(f"\nQuery: '{irrelevant_query}'")
    print(f"Answer: {res2['answer']}")
    print(f"Cited Record IDs: {[r['ident'] for r in res2['cited_records']]}")
    print(f"Tool Trace Steps: {[t['step'] for t in res2['tool_trace']]}")
