"""Streamlit UI for Aviation Maintenance RAG Copilot.

Provides:
  - Dual-column layout:
      Left Column : Chat Window with fault input, responses, citations & refusal alerts.
      Right Column: Inspector Tabs (Cited Records, Agent Tool Trace, Query & Entity Inspector).
  - FastAPI client targeting http://127.0.0.1:8000/chat
  - Fallback direct execution mode if backend is offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import streamlit as st

from search_engine import extract_domain_terms
from utils.normalizer import normalize_text

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
CHAT_ENDPOINT = f"{FASTAPI_URL}/chat"
CLIENT_TIMEOUT = float(os.getenv("COPILOT_CLIENT_TIMEOUT", "180.0"))

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aviation Maintenance Copilot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished Aviation Cockpit / Tech Aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .record-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-term {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-score {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .refusal-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .recurrence-alert-card {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-left: 6px solid #DC2626;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(220, 38, 38, 0.1);
    }
    .recurrence-alert-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #991B1B;
        margin-bottom: 4px;
    }
    .recurrence-alert-body {
        font-size: 0.92rem;
        color: #7F1D1D;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_recurrence_card(rec_info: dict[str, Any]) -> None:
    """Display a prominent red warning card if a recurring defect pattern is detected."""
    if rec_info and rec_info.get("is_recurring"):
        name = rec_info.get("component", "UNKNOWN")
        x = rec_info.get("count", 0)
        summary = rec_info.get("summary", "")
        symptoms = rec_info.get("symptoms", [])
        actions = rec_info.get("actions", [])

        symptoms_html = ""
        if symptoms:
            items = "".join(f"<li>{s}</li>" for s in symptoms)
            symptoms_html = f"<div style='margin-top: 6px;'><strong>Documented Symptoms:</strong><ul style='margin: 4px 0 6px 18px; padding: 0;'>{items}</ul></div>"

        actions_html = ""
        if actions:
            items = "".join(f"<li>{a}</li>" for a in actions)
            actions_html = f"<div style='margin-top: 6px;'><strong>Historical Fleet Fixes:</strong><ul style='margin: 4px 0 6px 18px; padding: 0;'>{items}</ul></div>"

        st.markdown(
            f"""
            <div class='recurrence-alert-card'>
                <div class='recurrence-alert-title'>
                    🚨 FLEET RECURRENCE ALERT: [{name}] has {x} recurring write-ups in historical logbooks
                </div>
                <div class='recurrence-alert-body'>
                    {f"<div style='margin: 6px 0; color: #fca5a5;'>{summary}</div>" if summary else ""}
                    {symptoms_html}
                    {actions_html}
                    <div style='margin-top: 8px; font-size: 0.85rem; color: #fcd34d;'>
                        💡 <em>Root-Cause Advisory: Repeated write-ups indicate a systemic issue rather than an isolated component failure. Investigate underlying valves, lines, and sensors.</em>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



# ---------------------------------------------------------------------------
# Backend API Helper
# ---------------------------------------------------------------------------
def check_backend_health() -> bool:
    """Check if FastAPI server is responsive."""
    try:
        resp = httpx.get(FASTAPI_URL, timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False


def query_backend_api(prompt: str) -> dict[str, Any]:
    """Send POST request to FastAPI /chat endpoint with local fallback."""
    print(f"\n[UI] Submitting query to backend at {CHAT_ENDPOINT} (timeout={CLIENT_TIMEOUT}s): '{prompt}'")
    try:
        response = httpx.post(CHAT_ENDPOINT, json={"query": prompt}, timeout=CLIENT_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"[UI] Backend response received (200 OK). Citations: {len(data.get('cited_records', []))}")
            return data
        print(f"[UI Error] Backend returned HTTP {response.status_code}: {response.text}")
        return {
            "answer": f"Backend Error ({response.status_code}): {response.text}",
            "cited_records": [],
            "tool_trace": [{"step": "error", "detail": response.text}],
            "recurring_defect_info": {"is_recurring": False},
        }
    except (httpx.TimeoutException, TimeoutError) as t_err:
        print(f"[UI Timeout] Request to {CHAT_ENDPOINT} timed out after {CLIENT_TIMEOUT}s: {t_err}")
        return {
            "answer": (
                f"⏱️ Diagnostic request timed out after {CLIENT_TIMEOUT:.0f}s while awaiting Ollama generation. "
                "Local inference on CPU may take longer for deep reasoning. "
                "Please retry or ensure Ollama is not busy with another task."
            ),
            "cited_records": [],
            "tool_trace": [{"step": "timeout", "detail": f"Timed out after {CLIENT_TIMEOUT}s"}],
            "recurring_defect_info": {"is_recurring": False},
        }
    except httpx.ConnectError:
        print(f"[UI Warning] Could not connect to FastAPI at {CHAT_ENDPOINT}. Running local agent fallback...")
        try:
            from agent import run_agent

            res = run_agent(prompt)
            print(f"[UI Fallback] Local agent executed successfully. Citations: {len(res.get('cited_records', []))}")
            return res
        except Exception as exc:
            print(f"[UI Error] Local agent fallback failed: {exc}")
            return {
                "answer": f"Could not connect to FastAPI at {CHAT_ENDPOINT} and local fallback failed: {exc}",
                "cited_records": [],
                "tool_trace": [{"step": "connection_error", "detail": str(exc)}],
                "recurring_defect_info": {"is_recurring": False},
            }
    except Exception as exc:
        print(f"[UI Error] Unexpected client error: {exc}")
        return {
            "answer": f"Unexpected client error: {exc}",
            "cited_records": [],
            "tool_trace": [{"step": "client_error", "detail": str(exc)}],
            "recurring_defect_info": {"is_recurring": False},
        }


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "inspector_data" not in st.session_state:
    st.session_state.inspector_data = {
        "raw_query": "ENG OIL PRESS LOW",
        "normalized_query": "engine OIL pressure LOW",
        "domain_terms": ["engine", "pressure"],
        "cited_records": [],
        "tool_trace": [],
        "refused": False,
        "recurring_defect_info": {"is_recurring": False},
    }


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("✈️ Copilot Controls")

    # Backend Status Indicator
    is_online = check_backend_health()
    if is_online:
        st.success("🟢 FastAPI Backend Online (`:8000`)")
    else:
        st.warning("🟠 Backend Offline — Fallback Direct Mode Active")
        st.caption("To run full API server:\n`uvicorn app:app --reload`")

    st.markdown("---")
    st.subheader("💡 Sample Discrepancy Queries")

    sample_queries = [
        "ENG 1 CYLINDER PLUG FOULED",  # Recurring Defect Test
        "ENG OIL PRESS LOW",
        "ACFT HYD SYS A LEAK REPL PUMP",
        "CYL HEAD TEMP HIGH INOP",
        "MAGNETOS TIMING OUT OF SPEC",
        "TCAS TRAFFIC COLLISION RESOLUTION ADVISORY COMPUTATION ERROR",  # refusal benchmark
    ]

    for sq in sample_queries:
        label = f"🍰 {sq}" if "cake" in sq else f"🔧 {sq}"
        if st.button(label, key=f"btn_{sq}", use_container_width=True):
            st.session_state.pending_query = sq

    st.markdown("---")
    st.markdown(
        """
        **System Architecture**:
        - **Retrieval**: BM25 + FAISS (`nomic-embed-text`)
        - **Fusion**: Reciprocal Rank Fusion (RRF)
        - **Reasoning**: LangGraph + `qwen2.5:3b`
        - **Safety**: Strict Refusal Threshold (< 0.35)
        """
    )
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.inspector_data = {
            "raw_query": "",
            "normalized_query": "",
            "domain_terms": [],
            "cited_records": [],
            "tool_trace": [],
            "refused": False,
        }
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div class='main-header'>✈️ Aviation Maintenance RAG Copilot</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Agentic logbook troubleshooting system with citation tracking & refusal enforcement</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dual-Column Layout
# ---------------------------------------------------------------------------
col_chat, col_inspector = st.columns([1.1, 0.9], gap="large")

# Handle pending query from sidebar buttons
user_query = None
if "pending_query" in st.session_state and st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

# ===========================================================================
# Left Column: Chat Window & Input Form
# ===========================================================================
with col_chat:
    st.subheader("💬 Logbook Diagnostic Assistant")

    # High-Visibility Direct Input Box (Always Visible at Top of Chat)
    with st.form("diagnose_form", clear_on_submit=True):
        st.markdown("**Enter Aircraft Fault / Discrepancy Description:**")
        col_in, col_btn = st.columns([3.8, 1.2])
        with col_in:
            form_query = st.text_input(
                "Enter Discrepancy:",
                placeholder="e.g., HYD SYS A LEAK, CYL HEAD TEMP HIGH, ENG VIBRATION...",
                label_visibility="collapsed",
            )
        with col_btn:
            form_submitted = st.form_submit_button("🔍 Diagnose", use_container_width=True)

    st.markdown("---")

    # Determine active query from sidebar button, form submission, or chat_input
    active_prompt = None
    if user_query:
        active_prompt = user_query
    elif form_submitted and form_query and form_query.strip():
        active_prompt = form_query.strip()

    chat_container = st.container()

    # Display past chat history
    with chat_container:
        if not st.session_state.messages and not active_prompt:
            st.info("💡 Type a discrepancy in the box above or pick a sample query from the sidebar to begin troubleshooting.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("recurring_defect_info", {}).get("is_recurring"):
                    render_recurrence_card(msg["recurring_defect_info"])
                if msg.get("refused", False):
                    st.warning("⚠️ Refused: Insufficient evidence in historical logbooks.")
                st.markdown(msg["content"])
                if msg.get("cited_records"):
                    citations = [f"`[Record #{r['ident']}]`" for r in msg["cited_records"]]
                    st.caption("Citations: " + " ".join(citations))

    # Optional bottom chat input
    bottom_input = st.chat_input("Or type here and press Enter...")
    if not active_prompt and bottom_input:
        active_prompt = bottom_input

    if active_prompt:
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Call backend API
        with st.chat_message("assistant"):
            with st.spinner("Analyzing historical logbooks, searching BM25 + FAISS..."):
                response_data = query_backend_api(active_prompt)

            answer = response_data.get("answer", "")
            cited = response_data.get("cited_records", [])
            trace = response_data.get("tool_trace", [])
            recurring_info = response_data.get("recurring_defect_info", {"is_recurring": False})

            # Check if query was refused
            is_refused = (
                "Insufficient evidence in historical logbooks" in answer
                or any(t.get("decision") == "REFUSED" for t in trace)
            )

            # Prominent Red Warning Card if Recurring Defect Detected
            if recurring_info.get("is_recurring", False):
                render_recurrence_card(recurring_info)

            if is_refused:
                st.warning("⚠️ Refused: Insufficient evidence in historical logbooks.")

            st.markdown(answer)

            if cited:
                citations = [f"`[Record #{r['ident']}]`" for r in cited]
                st.caption("Citations: " + " ".join(citations))

            # Save assistant message to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "cited_records": cited,
                "refused": is_refused,
                "tool_trace": trace,
                "recurring_defect_info": recurring_info,
            })

            # Update Inspector Data
            norm_q = normalize_text(active_prompt)
            dom_terms = extract_domain_terms(active_prompt)
            st.session_state.inspector_data = {
                "raw_query": active_prompt,
                "normalized_query": norm_q,
                "domain_terms": dom_terms,
                "cited_records": cited,
                "tool_trace": trace,
                "refused": is_refused,
                "recurring_defect_info": recurring_info,
            }


# ===========================================================================
# Right Column: Inspector Tabs
# ===========================================================================
with col_inspector:
    st.subheader("🔍 Inspection & Verification Panel")

    insp = st.session_state.inspector_data
    tab_records, tab_trace, tab_entities = st.tabs([
        "📑 Cited Records",
        "⚙️ Agent Tool Trace",
        "🔤 Query & Entity Inspector",
    ])

    # -----------------------------------------------------------------------
    # Tab 1: Cited Records
    # -----------------------------------------------------------------------
    with tab_records:
        records = insp.get("cited_records", [])
        if not records:
            if insp.get("refused", False):
                st.info("ℹ️ No records cited. The query was rejected by the Refusal Threshold Logic (< 0.35 match score).")
            else:
                st.info("No records to inspect yet. Submit a query to view retrieved maintenance evidence.")
        else:
            st.markdown(f"**Showing {len(records)} Historical Logbook Records Cited:**")

            for rank, rec in enumerate(records, start=1):
                rec_id = rec.get("ident", "N/A")
                score = rec.get("score", 0.0)
                faiss_sim = rec.get("faiss_similarity", 0.0)
                rrf = rec.get("rrf_score", 0.0)
                terms = rec.get("domain_terms", [])

                with st.expander(f"📌 Record #{rec_id} — Match Score: {score:.3f}", expanded=(rank == 1)):
                    # Badges / Metadata
                    st.markdown(
                        f"""
                        <span class='badge badge-score'>Score: {score:.3f}</span>
                        <span class='badge'>FAISS Cosine: {faiss_sim:.3f}</span>
                        <span class='badge'>RRF: {rrf:.5f}</span>
                        <span class='badge'>BM25 Rank: {rec.get('bm25_rank', 'N/A')}</span>
                        <span class='badge'>FAISS Rank: {rec.get('faiss_rank', 'N/A')}</span>
                        """,
                        unsafe_allow_html=True,
                    )

                    if terms:
                        term_html = "".join([f"<span class='badge badge-term'>{t}</span>" for t in terms])
                        st.markdown(f"<div style='margin-top:6px;'>Domain Terms: {term_html}</div>", unsafe_allow_html=True)

                    st.markdown("---")

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown("**Raw Problem Description:**")
                        st.code(rec.get("problem", "N/A"), language="text")
                        st.markdown("**Normalized Problem:**")
                        st.caption(rec.get("problem_normalized", "N/A"))

                    with col_r2:
                        st.markdown("**Raw Corrective Action:**")
                        st.code(rec.get("action", "N/A"), language="text")
                        st.markdown("**Normalized Action:**")
                        st.caption(rec.get("action_normalized", "N/A"))

    # -----------------------------------------------------------------------
    # Tab 2: Agent Tool Trace
    # -----------------------------------------------------------------------
    with tab_trace:
        trace = insp.get("tool_trace", [])
        if not trace:
            st.info("Execution trace is empty. Run a query to view LangGraph intermediate decisions.")
        else:
            st.markdown("**LangGraph Execution Pipeline Steps:**")

            for idx, step in enumerate(trace, start=1):
                step_name = step.get("step", f"Step {idx}")
                status_icon = "🟢" if step.get("decision") != "REFUSED" else "🔴"

                with st.expander(f"{status_icon} Step {idx}: `{step_name}`", expanded=True):
                    st.json(step)

            st.markdown("---")
            with st.expander("📄 Complete Raw Trace Payload"):
                st.json(trace)

    # -----------------------------------------------------------------------
    # Tab 3: Query & Entity Inspector
    # -----------------------------------------------------------------------
    with tab_entities:
        raw_q = insp.get("raw_query", "")
        norm_q = insp.get("normalized_query", "")
        terms = insp.get("domain_terms", [])

        st.markdown("#### 1. Abbreviation & Grammar Normalization")
        st.caption("Expansion performed dynamically via `utils/normalizer.py` using `Aviation_Abbreviation_Dataset.csv` and `Aviation_grammar_Dataset.csv`:")

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.markdown("**Raw Shorthand Input:**")
            st.code(raw_q if raw_q else "No input", language="text")
        with col_q2:
            st.markdown("**Standardized & Lemmatized Output:**")
            st.code(norm_q if norm_q else "No input", language="text")

        st.markdown("---")
        st.markdown("#### 2. Technical Domain Term Extraction")
        st.caption("Extracted via `search_engine.extract_domain_terms()` using `Aviation_TermBanks_Dataset.csv`:")

        if terms:
            st.success(f"Identified {len(terms)} key technical component(s):")
            chip_html = "".join([f"<span class='badge badge-term' style='font-size:0.85rem; padding:6px 12px;'>{t}</span>" for t in terms])
            st.markdown(chip_html, unsafe_allow_html=True)
        else:
            st.warning("No technical domain terms detected in query.")

        st.markdown("---")
        st.markdown("#### 3. Match Confidence & Threshold Analysis")
        threshold = 0.35
        # Extract top score from trace if available
        top_score = 0.0
        for t in trace:
            if "top_score" in t:
                top_score = t["top_score"]
                break

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Refusal Threshold", f"{threshold:.2f}")
        col_m2.metric("Query Top Score", f"{top_score:.3f}")
        col_m3.metric("Decision", "REFUSED" if insp.get("refused", False) else "ACCEPTED")

        st.markdown("---")
        st.markdown("#### 4. Fleet Recurrence & Defect Analysis")
        rec_data = insp.get("recurring_defect_info", {})
        if rec_data.get("is_recurring"):
            st.error(
                f"🚨 **FLEET RECURRENCE ALERT:** Component `{rec_data.get('component')}` "
                f"has failed **{rec_data.get('count')}** times in historical logbooks (Records: `{rec_data.get('records')}`).\n\n"
                f"**Recommendation:** Consider root-cause analysis rather than isolated component replacement."
            )
        else:
            st.success("✅ **Normal Pattern:** No high-frequency recurring defect pattern detected (component threshold < 3).")
