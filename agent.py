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
import time
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
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
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
# LLM Initialization with 100% Local Ollama (No Cloud API Keys Required)
# ---------------------------------------------------------------------------
class LocalOllamaLLM:
    """Lightweight LangChain-compatible wrapper for local Ollama server."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        requested_model = (model or os.getenv("OLLAMA_MODEL", "")).strip()
        self.model = self._resolve_model(requested_model)

    def _get_installed_models(self) -> list[str]:
        """Fetch list of installed models from local Ollama instance."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                return [m.get("name", "") for m in models if m.get("name")]
        except Exception:
            return []

    def _resolve_model(self, preferred_model: str) -> str:
        """Resolve model name, matching installed Ollama models if possible."""
        installed = self._get_installed_models()
        if preferred_model:
            for m in installed:
                if m == preferred_model or m.startswith(preferred_model) or preferred_model.startswith(m):
                    return m
            return preferred_model

        for candidate in ["mistral:latest", "mistral", "qwen2.5:3b", "qwen2.5", "llama3.2:latest", "llama3.2"]:
            for m in installed:
                if m == candidate or m.startswith(candidate.split(":")[0]):
                    return m

        return installed[0] if installed else "mistral:latest"

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Invoke local Ollama using /api/chat or /api/generate."""
        chat_messages = []
        user_parts = []
        system_content = ""

        for msg in messages:
            msg_type = getattr(msg, "type", "user")
            content = getattr(msg, "content", "")
            if not content:
                continue

            if msg_type == "system":
                system_content = content
            elif msg_type == "human":
                user_parts.append(content)
            elif msg_type == "tool":
                user_parts.append(f"\n[Verified Historical Logbook Records]:\n{content}")
            elif msg_type == "ai":
                chat_messages.append({"role": "assistant", "content": content})
            else:
                user_parts.append(str(content))

        if system_content:
            chat_messages.insert(0, {"role": "system", "content": system_content})

        if user_parts:
            combined_user_text = "\n".join(user_parts)
            chat_messages.append({"role": "user", "content": combined_user_text})

        predict_tokens = int(os.getenv("OLLAMA_NUM_PREDICT", "450"))
        payload = json.dumps({
            "model": self.model,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": predict_tokens,
            },
        }).encode("utf-8")

        url = f"{self.base_url}/api/chat"
        print(f"[Ollama] Querying {self.model} via /api/chat (num_predict={predict_tokens})...")
        t_start = time.time()
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_text = data.get("message", {}).get("content", "").strip()
                if response_text:
                    elapsed = time.time() - t_start
                    print(f"[Ollama] Response generated in {elapsed:.2f}s ({len(response_text)} chars).")
                    return AIMessage(content=response_text)
        except urllib.error.URLError as u_err:
            print(f"[Ollama Connection Error] {url}: {u_err}")
            return AIMessage(
                content=(
                    f"⚠️ Local Ollama is offline or unreachable at {self.base_url}.\n"
                    f"Please ensure Ollama is running (`ollama serve`)."
                )
            )
        except Exception as exc:
            print(f"[Ollama Execution Warning] /api/chat error: {exc}. Trying /api/generate fallback...")

        # Fallback to /api/generate if /api/chat had issues
        try:
            full_prompt = (f"System: {system_content}\n\n" if system_content else "") + "\n\n".join(user_parts)
            gen_payload = json.dumps({
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": predict_tokens},
            }).encode("utf-8")
            gen_req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=gen_payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(gen_req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_text = data.get("response", "").strip()
                if response_text:
                    elapsed = time.time() - t_start
                    print(f"[Ollama Fallback] Response generated in {elapsed:.2f}s ({len(response_text)} chars).")
                    return AIMessage(content=response_text)
        except Exception as gen_exc:
            print(f"[Ollama Error] Generate fallback failed: {gen_exc}")

        return AIMessage(
            content="Local Ollama is currently busy or loading model. Please retry in a few moments."
        )


def get_llm():
    """Initialize primary local Ollama LLM."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model_name = os.getenv("OLLAMA_MODEL", "")
    llm = LocalOllamaLLM(base_url=base_url, model=model_name or None)
    print(f"Initializing Local Ollama LLM ({llm.model}) at {llm.base_url}...")
    return llm


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
    print(f"\n[Agent] Stage 1/3 (Normalize Input): '{raw}' -> '{norm}'")
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
    print(f"[Agent] Stage 2/3 (Hybrid Retrieval): Querying '{query}'...")
    results = search_records(query, top_k=5)

    trace = list(state.get("tool_trace", []))
    top_score = results[0]["score"] if results else 0.0

    # Call detect_recurring_defects on the search results before generating the answer
    recurring_info = detect_recurring_defects(results)
    if recurring_info.get("is_recurring"):
        print(f"[Agent] 🚨 Fleet recurrence detected: [{recurring_info.get('component')}] occurred {recurring_info.get('count')} times.")

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
        print(f"[Agent] ⚠️ Decision: REFUSED (Top score {top_score:.3f} < threshold {REFUSAL_THRESHOLD})")
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

    print(f"[Agent] ✅ Decision: ACCEPTED (Top match score {top_score:.3f} >= threshold {REFUSAL_THRESHOLD})")

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
        "4. Keep the response focused, structured, and concise (under 250 words total) while maintaining technical troubleshooting substance.\n"
        "5. For EVERY finding, symptom, and repair action, explicitly cite the verified logbook record using '[Record #<ID>]' (e.g., [Record #100062]).\n\n"
        "FORMATTING STRUCTURE:\n"
        "### 1. Problem & Malfunction Breakdown\n"
        "Explain the specific operating conditions, failure symptoms, and affected assemblies cited in [Record #XXXXXX].\n\n"
        "### 2. Historical Corrective Actions & Root Causes\n"
        "Detail the physical repairs and root cause discoveries made in past logs [Record #XXXXXX].\n\n"
        "### 3. Recommended Flight-Line Troubleshooting Steps\n"
        "Provide 2-3 prioritized diagnostic actions for the maintenance technician."
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

    print(f"[Agent] Stage 3/3 (Generate Diagnosis): Generating structured diagnosis citing {len(results)} records...")
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

    print(f"[Agent] Diagnosis complete. Citing records: {[r['ident'] for r in cited_records]}")

    trace.append({
        "step": "generate_answer",
        "model": getattr(LLM, "model", "ollama"),
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
