"""Hybrid search engine for aviation maintenance records.

Combines:
  1. BM25 (sparse keyword search via `rank_bm25`)
  2. FAISS (dense semantic vector search via Ollama `nomic-embed-text`)
  3. Reciprocal Rank Fusion (RRF) for merged retrieval
  4. Domain Term Bank (`Aviation_TermBanks_Dataset.csv`) for technical entity extraction

Provides:
  - `search_records(query: str, top_k: int = 5)`: unified hybrid search.
  - `extract_domain_terms(text: str) -> list[str]`: extracts key technical components.
  - `get_record(record_id: str | int)`: retrieves full record details.
"""

from __future__ import annotations

import csv
import json
import re
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    import faiss
except ImportError:
    faiss = None
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from utils.normalizer import normalize, normalize_text

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "maintnet_aviation_normalized.csv"
FAISS_INDEX_PATH = ROOT_DIR / "data" / "faiss_index.bin"
EMBEDDINGS_CACHE_PATH = ROOT_DIR / "data" / "embeddings_nomic.npy"
TERMBANKS_PATH = ROOT_DIR / "data" / "Aviation_TermBanks_Dataset.csv"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Domain Term Bank Loader & Technical Entity Extractor
# ---------------------------------------------------------------------------
def load_domain_terms(csv_path: Path | str | None = None) -> list[str]:
    """Load technical aviation terms from Aviation_TermBanks_Dataset.csv."""
    path = Path(csv_path) if csv_path else TERMBANKS_PATH
    if not path.exists():
        alt_path = Path.cwd() / "data" / "Aviation_TermBanks_Dataset.csv"
        if alt_path.exists():
            path = alt_path

    terms_set: set[str] = set()
    if path.exists():
        with open(path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = (row.get("Word") or "").strip().lower()
                if word:
                    # Clean internal whitespace
                    word = re.sub(r"\s+", " ", word)
                    terms_set.add(word)

    # Return terms sorted by length descending so longer phrases match first
    return sorted(terms_set, key=len, reverse=True)


DOMAIN_TERMS: list[str] = load_domain_terms()


def _compile_term_regex(terms: list[str]) -> tuple[re.Pattern, dict[str, str]]:
    """Compile a named-group regex allowing optional plural suffixes for terms."""
    group_map: dict[str, str] = {}
    patterns: list[str] = []
    for idx, term in enumerate(terms):
        grp_name = f"term_{idx}"
        group_map[grp_name] = term
        escaped = re.escape(term)
        if not term.endswith("s"):
            patterns.append(rf"(?P<{grp_name}>\b{escaped}(?:s|es)?\b)")
        else:
            patterns.append(rf"(?P<{grp_name}>\b{escaped}\b)")

    regex = re.compile("|".join(patterns), re.IGNORECASE) if patterns else re.compile(r"$^")
    return regex, group_map


_TERM_REGEX, _GROUP_MAP = _compile_term_regex(DOMAIN_TERMS)


def extract_domain_terms(text: str) -> list[str]:
    """Identify and extract key technical aviation domain components from text.

    Uses `Aviation_TermBanks_Dataset.csv` terms and matches against both
    the raw text and its abbreviation-normalized form.

    Parameters
    ----------
    text : str
        Input query, problem description, or maintenance log.

    Returns
    -------
    list[str]
        List of identified technical domain terms (canonical lowercase, deduplicated).
    """
    if not isinstance(text, str) or not text.strip():
        return []

    norm_text = normalize_text(text)
    # Search both raw and normalized representations so abbreviations (e.g. 'hyd')
    # and full words (e.g. 'hydraulic') are both captured
    search_target = f"{text} {norm_text}" if text.lower() != norm_text.lower() else text

    extracted: list[str] = []
    seen: set[str] = set()
    for match in _TERM_REGEX.finditer(search_target):
        canonical = _GROUP_MAP.get(match.lastgroup, match.group().lower())
        if canonical not in seen:
            seen.add(canonical)
            extracted.append(canonical)

    return extracted


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\b\w+\b", text.lower())


def get_ollama_embeddings(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Request embeddings for a list of texts from local Ollama server."""
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    return embeddings


# ---------------------------------------------------------------------------
# Hybrid Search Engine
# ---------------------------------------------------------------------------
class HybridSearchEngine:
    """Hybrid search engine combining BM25, FAISS dense vectors, and Domain Terms."""

    def __init__(self, data_path: Path = DATA_PATH, force_rebuild: bool = False) -> None:
        self.data_path = data_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        self._initialize(force_rebuild=force_rebuild)

    def _initialize(self, force_rebuild: bool = False) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Normalized dataset not found at {self.data_path}")

        print(f"Loading dataset from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records.")

        # 1. Build BM25 index on combined normalized text
        print("Building BM25 index on COMBINED_NORMALIZED...")
        corpus = self.df["COMBINED_NORMALIZED"].fillna("").tolist()
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index ready.")

        # 2. Build or load FAISS dense vector index
        self._setup_faiss_index(corpus, force_rebuild=force_rebuild)

    def _setup_faiss_index(self, corpus: list[str], force_rebuild: bool = False) -> None:
        """Load or build FAISS index using cached embeddings when available."""
        if faiss is None:
            print("FAISS module not installed; continuing in pure BM25 search mode.")
            self.faiss_index = None
            return

        if not force_rebuild and FAISS_INDEX_PATH.exists() and EMBEDDINGS_CACHE_PATH.exists():
            print(f"Loading cached FAISS index from {FAISS_INDEX_PATH}...")
            self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
            print(f"FAISS index loaded ({self.faiss_index.ntotal} vectors).")
            return

        embeddings: np.ndarray
        if not force_rebuild and EMBEDDINGS_CACHE_PATH.exists():
            print(f"Loading cached embeddings from {EMBEDDINGS_CACHE_PATH}...")
            embeddings = np.load(str(EMBEDDINGS_CACHE_PATH))
        else:
            print(f"Computing embeddings via Ollama '{EMBED_MODEL}' in batches of {BATCH_SIZE}...")
            all_embeddings = []
            total = len(corpus)
            for i in range(0, total, BATCH_SIZE):
                batch = corpus[i : i + BATCH_SIZE]
                batch_emb = get_ollama_embeddings(batch)
                all_embeddings.append(batch_emb)
                if (i + BATCH_SIZE) % 1000 < BATCH_SIZE or (i + BATCH_SIZE) >= total:
                    print(f"  Embedded {min(i + BATCH_SIZE, total)} / {total} records...")

            embeddings = np.vstack(all_embeddings).astype(np.float32)
            print(f"Saving embeddings cache to {EMBEDDINGS_CACHE_PATH}...")
            np.save(str(EMBEDDINGS_CACHE_PATH), embeddings)

        # Normalize vectors for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        print(f"Building FAISS IndexFlatIP (dim={dimension}, count={len(embeddings)})...")
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

        print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
        faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH))
        print("FAISS index built and saved.")

    def rebuild_index(self) -> None:
        """Force re-embedding of normalized dataset and rebuild of FAISS index."""
        print("\nRebuilding FAISS vector index and embeddings from scratch...")
        corpus = self.df["COMBINED_NORMALIZED"].fillna("").tolist()
        self._setup_faiss_index(corpus, force_rebuild=True)

    def search_bm25(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """Search using BM25, returning list of (doc_index, score)."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not initialized.")
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def search_faiss(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """Search using FAISS dense vectors, returning list of (doc_index, similarity)."""
        if faiss is None or self.faiss_index is None:
            return []
        try:
            query_emb = get_ollama_embeddings([query])
            faiss.normalize_L2(query_emb)
            distances, indices = self.faiss_index.search(query_emb, top_k)
            return [(int(indices[0][i]), float(distances[0][i])) for i in range(len(indices[0])) if indices[0][i] >= 0]
        except Exception:
            return []

    def search_records(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        candidate_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Execute hybrid search using Reciprocal Rank Fusion (RRF).

        Parameters
        ----------
        query : str
            User query (raw or abbreviated).
        top_k : int, default 5
            Number of top records to return.
        rrf_k : int, default 60
            RRF smoothing constant.
        candidate_k : int, default 50
            Number of candidates to retrieve from each retriever.

        Returns
        -------
        list[dict[str, Any]]
            Ranked list of record dictionaries with scores, text fields, and domain terms.
        """
        # Expand abbreviations for semantic and keyword matching
        normalized_query = normalize(query)
        bm25_search_query = f"{query} {normalized_query}" if query.lower() != normalized_query.lower() else query

        bm25_results = self.search_bm25(bm25_search_query, top_k=candidate_k)
        faiss_results = self.search_faiss(normalized_query, top_k=candidate_k)

        # Compute Reciprocal Rank Fusion (RRF) scores
        rrf_scores: dict[int, float] = {}
        bm25_rank_map: dict[int, int] = {}
        faiss_rank_map: dict[int, int] = {}
        faiss_sim_map: dict[int, float] = {}

        for rank, (idx, _) in enumerate(bm25_results, start=1):
            bm25_rank_map[idx] = rank
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))

        for rank, (idx, sim) in enumerate(faiss_results, start=1):
            faiss_rank_map[idx] = rank
            faiss_sim_map[idx] = sim
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))

        # Sort candidate documents by descending RRF score
        # Evaluate candidate pool
        sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:candidate_k]

        max_possible_rrf = 2.0 / (rrf_k + 1)  # Maximum possible RRF if ranked #1 in both

        # Filter stop words to evaluate true technical token overlap between query and candidates
        _stop_words = {
            "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
            "is", "are", "was", "were", "with", "by", "from", "up", "out", "off",
            "into", "over", "after", "no", "not", "be", "been", "it", "this", "that"
        }
        q_tokens = set(t for t in tokenize(f"{query} {normalized_query}") if t not in _stop_words and len(t) > 2)

        results: list[dict[str, Any]] = []
        for idx, rrf_score in sorted_candidates:
            row = self.df.iloc[idx]
            faiss_sim = max(float(faiss_sim_map.get(idx, 0.0)), 0.0)
            norm_rrf = min(float(rrf_score) / max_possible_rrf, 1.0)
            combined_str = str(row["COMBINED_NORMALIZED"])

            # Compute content token overlap between query and document to prevent incidental 1-word BM25 hits
            doc_tokens = set(tokenize(combined_str))
            overlap = q_tokens & doc_tokens
            overlap_ratio = len(overlap) / max(len(q_tokens), 1)
            effective_bm25 = (idx in bm25_rank_map) and (overlap_ratio >= 0.25 or len(overlap) >= 2)

            # Composite normalized match score in [0, 1]
            if effective_bm25 and (faiss_sim >= 0.52 or not faiss_results):
                effective_sim = faiss_sim if faiss_results else 0.75
                match_score = round(norm_rrf * (0.4 + 0.6 * effective_sim), 4)
            elif faiss_sim >= 0.60 and (overlap_ratio >= 0.20 or len(overlap) >= 2):
                match_score = round(norm_rrf * faiss_sim, 4)
            elif effective_bm25 and faiss_sim >= 0.40:
                match_score = round(norm_rrf * (0.15 + 0.25 * faiss_sim), 4)
            else:
                # Incidental single-word match or out-of-domain discrepancy
                match_score = round(norm_rrf * (faiss_sim ** 3) * max(overlap_ratio, 0.2), 4)

            record_domain_terms = extract_domain_terms(combined_str)

            results.append({
                "ident": int(row["IDENT"]),
                "score": match_score,
                "similarity_score": match_score,
                "normalized_rrf": round(norm_rrf, 4),
                "rrf_score": round(float(rrf_score), 6),
                "bm25_rank": bm25_rank_map.get(idx),
                "faiss_rank": faiss_rank_map.get(idx),
                "faiss_similarity": round(faiss_sim, 4),
                "domain_terms": record_domain_terms,
                "problem": str(row["PROBLEM"]),
                "problem_normalized": str(row["PROBLEM_NORMALIZED"]),
                "action": str(row["ACTION"]),
                "action_normalized": str(row["ACTION_NORMALIZED"]),
                "combined_text": combined_str,
            })

        # Rank candidates by composite confidence score, secondary by RRF score
        results.sort(key=lambda r: (r["score"], r["rrf_score"]), reverse=True)
        return results[:top_k]

    def get_record(self, record_id: str | int) -> dict[str, Any] | None:
        """Retrieve a specific record by its IDENT ID."""
        try:
            target_id = int(str(record_id).strip().lstrip("#"))
        except (ValueError, TypeError):
            return None
        matches = self.df[self.df["IDENT"] == target_id]
        if matches.empty:
            return None
        row = matches.iloc[0]
        combined_str = str(row["COMBINED_NORMALIZED"])
        return {
            "ident": int(row["IDENT"]),
            "domain_terms": extract_domain_terms(combined_str),
            "problem": str(row["PROBLEM"]),
            "problem_normalized": str(row["PROBLEM_NORMALIZED"]),
            "action": str(row["ACTION"]),
            "action_normalized": str(row["ACTION_NORMALIZED"]),
            "combined_text": combined_str,
        }


# Global singleton instance for easy import across modules
_ENGINE: HybridSearchEngine | None = None


def get_engine(force_rebuild: bool = False) -> HybridSearchEngine:
    """Return a shared singleton instance of HybridSearchEngine."""
    global _ENGINE
    if _ENGINE is None or force_rebuild:
        _ENGINE = HybridSearchEngine(force_rebuild=force_rebuild)
    return _ENGINE


def search_records(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Convenience function matching the required interface."""
    engine = get_engine()
    return engine.search_records(query=query, top_k=top_k)


def get_record(record_id: str | int) -> dict[str, Any] | None:
    """Convenience function to get a single record by ID."""
    engine = get_engine()
    return engine.get_record(record_id=record_id)


GENERIC_TERMS = {
    "pressure", "system", "check", "inspection", "operation", "remove", "replace",
    "installed", "condition", "unit", "level", "service", "high", "low", "assembly",
    "line", "leak", "fail", "failure",
}


def _resolve_component_label(comp: str, all_text: str) -> str:
    """Enhance single-word or generic terms into specific aviation subsystems."""
    lower_text = all_text.lower()
    lower_comp = comp.lower()

    if lower_comp == "pressure":
        if "oil" in lower_text or "eng" in lower_text:
            return "ENGINE OIL PRESSURE SYSTEM"
        elif "hyd" in lower_text or "hydraulic" in lower_text:
            return "HYDRAULIC PRESSURE SYSTEM"
        elif "fuel" in lower_text:
            return "FUEL PRESSURE SYSTEM"
        elif "manifold" in lower_text:
            return "MANIFOLD PRESSURE SYSTEM"
        return "PRESSURE SENSOR & MONITORING SYSTEM"

    if lower_comp in ("cylinder", "cyl"):
        return "CYLINDER POWER SECTION"

    if lower_comp in ("engine", "eng"):
        if "oil" in lower_text:
            return "ENGINE OIL SYSTEM"
        return "POWERPLANT / ENGINE SYSTEM"

    if lower_comp == "magneto":
        return "MAGNETO IGNITION SYSTEM"

    if lower_comp == "valve":
        if "relief" in lower_text:
            return "PRESSURE RELIEF VALVE"
        return "VALVE / FLOW REGULATOR"

    if lower_comp == "pump":
        if "hyd" in lower_text or "hydraulic" in lower_text:
            return "HYDRAULIC PUMP ASSEMBLY"
        elif "fuel" in lower_text:
            return "FUEL PUMP ASSEMBLY"
        return "FLUID PUMP ASSEMBLY"

    if lower_comp == "gasket":
        return "GASKET / SEAL INTEGRITY"

    return f"{comp.upper()} ASSEMBLY"


def detect_recurring_defects(retrieved_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect recurring component or fault patterns across retrieved records.

    Extracts domain components from each record using extract_domain_terms,
    counts component co-occurrences across the retrieved records, and
    flags recurring patterns if any component appears >= 3 times.

    Parameters
    ----------
    retrieved_records : list[dict]
        List of historical logbook records returned from retrieval.

    Returns
    -------
    dict[str, Any]
        If any component/fault pattern appears >= 3 times:
            {
                "is_recurring": True,
                "component": name,
                "count": freq,
                "severity": "HIGH",
                "records": [record_ids],
                "symptoms": ["Symptom 1", "Symptom 2", ...],
                "actions": ["Action 1", "Action 2", ...],
                "summary": "Detailed technical explanation..."
            }
        Otherwise:
            {"is_recurring": False}
    """
    if not retrieved_records:
        return {"is_recurring": False}

    component_records: dict[str, list[int | str]] = {}

    for record in retrieved_records:
        rec_id = record.get("ident")
        # Extract domain terms from text representations
        text = record.get("combined_text") or f"{record.get('problem', '')} {record.get('action', '')}"
        extracted_terms = extract_domain_terms(text)
        if not extracted_terms and record.get("domain_terms"):
            extracted_terms = record.get("domain_terms", [])

        # Count per-record occurrence to avoid duplicate counting within a single logbook entry
        for term in set(extracted_terms):
            canonical = term.strip()
            if not canonical:
                continue
            if canonical not in component_records:
                component_records[canonical] = []
            if rec_id is not None:
                component_records[canonical].append(rec_id)

    # Filter for components or fault patterns appearing >= 3 times
    recurring_candidates = [
        (comp, rec_ids) for comp, rec_ids in component_records.items() if len(rec_ids) >= 3
    ]

    if recurring_candidates:
        # Prioritize domain-specific hardware terms over generic measurement terms
        def _rank_candidate(item):
            comp_name, rec_ids = item
            is_generic = comp_name.lower() in GENERIC_TERMS
            return (0 if is_generic else 1, len(rec_ids), len(comp_name))

        recurring_candidates.sort(key=_rank_candidate, reverse=True)
        top_comp, matched_records = recurring_candidates[0]

        # Aggregate all matching text for contextual component labeling
        matched_text_list = []
        symptoms = []
        actions = []

        for r in retrieved_records:
            if r.get("ident") in matched_records:
                prob = (r.get("problem_normalized") or r.get("problem") or "").strip()
                act = (r.get("action_normalized") or r.get("action") or "").strip()
                matched_text_list.append(f"{prob} {act}")

                if prob and prob not in symptoms:
                    clean_prob = prob[0].upper() + prob[1:] if len(prob) > 1 else prob
                    symptoms.append(clean_prob)
                if act and act not in actions:
                    clean_act = act[0].upper() + act[1:] if len(act) > 1 else act
                    actions.append(clean_act)

        all_text = " ".join(matched_text_list)
        resolved_label = _resolve_component_label(top_comp, all_text)

        # Generate a concise, informative engineering summary
        symptom_snippet = "; ".join(symptoms[:2]) if symptoms else "recurring operational discrepancies"
        action_snippet = "; ".join(actions[:2]) if actions else "component adjustments and servicing"
        summary = (
            f"Chronic fleet pattern detected across {len(matched_records)} records involving {resolved_label.title()}. "
            f"Logged discrepancies include: {symptom_snippet}. "
            f"Historical corrective actions: {action_snippet}."
        )

        return {
            "is_recurring": True,
            "component": resolved_label,
            "count": len(matched_records),
            "severity": "HIGH",
            "records": matched_records,
            "symptoms": symptoms[:4],
            "actions": actions[:4],
            "summary": summary,
        }

    return {"is_recurring": False}



if __name__ == "__main__":
    rebuild_flag = "--rebuild" in sys.argv

    print("\n" + "=" * 65)
    print("  AVIATION MAINTENANCE SEARCH ENGINE & TECHNICAL ENTITY EXTRACTION")
    print("=" * 65)

    if rebuild_flag:
        print("[!] Rebuild flag detected: Re-embedding dataset and rebuilding FAISS index...")
        engine = get_engine(force_rebuild=True)
    else:
        engine = get_engine()

    test_query = "ENG OIL PRESS LOW"
    print(f"\nTest Query        : '{test_query}'")
    print(f"Normalized Query  : '{normalize(test_query)}'")
    query_terms = extract_domain_terms(test_query)
    print(f"Query Domain Terms: {query_terms}\n")

    top_records = search_records(test_query, top_k=3)

    print("-" * 65)
    print(f"TOP {len(top_records)} HYBRID SEARCH RESULTS (BM25 + FAISS via RRF):")
    print("-" * 65)
    for rank, rec in enumerate(top_records, start=1):
        print(f"\n[Rank {rank}] Record ID: {rec['ident']}")
        print(f"  Match Score     : {rec['score']} (Cosine Sim: {rec['faiss_similarity']}, RRF: {rec['rrf_score']})")
        print(f"  Domain Terms    : {rec['domain_terms']}")
        print(f"  Raw Problem     : {rec['problem']}")
        print(f"  Norm Problem    : {rec['problem_normalized']}")
        print(f"  Raw Action      : {rec['action']}")
        print(f"  Norm Action     : {rec['action_normalized']}")
