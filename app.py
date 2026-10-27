"""FastAPI backend service for Aviation Maintenance Copilot.

Provides:
  - POST /chat : Takes JSON {"query": str} and returns:
      {
        "answer": str,
        "cited_records": list[dict],
        "tool_trace": list[dict]
      }
  - GET / : Service health status and endpoint overview.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import run_agent

PUBLIC_INDEX = PROJECT_ROOT / "public" / "index.html"

app = FastAPI(
  title="Aviation Maintenance Copilot API",
  description="Agentic RAG assistant for commercial aviation maintenance logbooks.",
  version="1.0.0",
)

# Enable CORS for potential Streamlit or web UI integration
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


class ChatRequest(BaseModel):
  query: str = Field(
    ...,
    description="Maintenance problem description, fault message, or query.",
    examples=["ENG OIL PRESS LOW", "HYD SYS A LEAK"],
  )


class ChatResponse(BaseModel):
  answer: str = Field(..., description="Generated answer or refusal response.")
  cited_records: list[dict[str, Any]] = Field(
    default_factory=list,
    description="Historical logbook records cited or used in the answer.",
  )
  tool_trace: list[dict[str, Any]] = Field(
    default_factory=list,
    description="Execution trace of tool calls and intermediate agent steps.",
  )
  recurring_defect_info: dict[str, Any] = Field(
    default_factory=lambda: {"is_recurring": False},
    description="Fleet recurrence detection metadata and severity.",
  )


@app.get("/")
def root():
  """Serve the Web Cockpit UI directly or fallback to service info."""
  if PUBLIC_INDEX.exists():
    return FileResponse(PUBLIC_INDEX)
  return {
    "status": "online",
    "service": "Aviation Maintenance Copilot API",
    "endpoints": {
      "chat": "POST /chat",
      "docs": "GET /docs",
    },
  }


@app.get("/api")
@app.get("/api/")
@app.get("/api/index.py")
@app.get("/api/index")
@app.get("/health")
def health() -> dict[str, str]:
  """Health check endpoint."""
  return {"status": "online"}


@app.post("/chat", response_model=ChatResponse)
@app.post("/chat/", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
@app.post("/api/chat/", response_model=ChatResponse)
@app.post("/api/index.py", response_model=ChatResponse)
@app.post("/api/index", response_model=ChatResponse)
@app.post("/api", response_model=ChatResponse)
@app.post("/api/", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
  """Process a maintenance query through the agentic RAG pipeline."""
  raw_query = request.query.strip()
  if not raw_query:
    raise HTTPException(status_code=400, detail="Query string cannot be empty.")

  start_t = time.time()
  print(f"\n[FastAPI /chat] Incoming maintenance query: '{raw_query}'")
  try:
    result = run_agent(raw_query)
    elapsed = time.time() - start_t
    cited = result.get("cited_records", [])
    print(f"[FastAPI /chat] Pipeline succeeded in {elapsed:.2f}s | Citations: {[r.get('ident') for r in cited]}")
    return ChatResponse(
      answer=result.get("answer", ""),
      cited_records=cited,
      tool_trace=result.get("tool_trace", []),
      recurring_defect_info=result.get(
        "recurring_defect_info", {"is_recurring": False}
      ),
    )
  except Exception as exc:
    elapsed = time.time() - start_t
    print(f"[FastAPI /chat Error] Pipeline failed after {elapsed:.2f}s: {exc}")
    traceback.print_exc()
    raise HTTPException(
      status_code=500,
      detail=f"Error executing agent pipeline: {exc}",
    ) from exc


if __name__ == "__main__":
  import uvicorn

  host = "127.0.0.1"
  port = 8000
  print("\n" + "=" * 65)
  print("  AVIATION MAINTENANCE COPILOT - FASTAPI BACKEND")
  print("=" * 65)
  print(f"• Local Server URL : http://{host}:{port}")
  print(f"• Interactive Docs : http://{host}:{port}/docs")
  print(f"• Chat Endpoint    : POST http://{host}:{port}/chat")
  print("=" * 65 + "\n")

  uvicorn.run(app, host=host, port=port)
