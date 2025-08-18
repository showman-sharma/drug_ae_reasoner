# drug_ae_reasoner/data/cadec_loader.py
"""
Helpers to read/query the CADEC KG.

Key features:
- Fast in-proc cache for the CADEC graph
- Robust drug lookup:
  1) Tries RxNorm CUI intersection (preferred)
  2) Falls back to normalized label equality / substring
- Utilities to fetch CADEC (drug -> AE) edges for matched drug nodes
"""

from __future__ import annotations

import os
import re
import pickle
import logging
from typing import Any, List, Tuple, Set, Dict

import networkx as nx
import numpy as np

from ..utils.encoding import encode_text

# Pull defaults from project config, but keep functions parametric so callers can override.
from ..config import CADEC_KG_PATH as _DEFAULT_CADEC_KG
from ..config import RX_PATH as _DEFAULT_RXN_RRF

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Graph cache
# ──────────────────────────────────────────────────────────────────────────────
_CADEC_KG_CACHE: Dict[str, nx.MultiDiGraph] = {}
# Cache for precomputed drug label embeddings per KG path
_DRUG_EMB_CACHE: Dict[str, List[Tuple[str, str, Set[str], np.ndarray]]] = {}


def _load_cadec_graph(kg_path: str | None = None) -> nx.MultiDiGraph:
    """
    Load CADEC KG (pickled networkx MultiDiGraph) once and cache it.
    """
    kg_path = kg_path or _DEFAULT_CADEC_KG
    if kg_path not in _CADEC_KG_CACHE:
        if not os.path.exists(kg_path):
            raise FileNotFoundError(f"CADEC KG not found at: {kg_path}")
        with open(kg_path, "rb") as f:
            G = pickle.load(f)
        if not isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
            raise TypeError("CADEC KG must be a (Multi)DiGraph")
        # Coerce to MultiDiGraph for consistent edge API
        if isinstance(G, nx.DiGraph):
            G = nx.MultiDiGraph(G)
        _CADEC_KG_CACHE[kg_path] = G
        logger.info("[CADEC] Loaded graph: %s (nodes=%d, edges=%d)", kg_path, G.number_of_nodes(), G.number_of_edges())
    return _CADEC_KG_CACHE[kg_path]


# ──────────────────────────────────────────────────────────────────────────────
# Normalization / RxNorm utilities
# ──────────────────────────────────────────────────────────────────────────────
_NORM_RX = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    """
    Lowercase, strip punctuation/whitespace, collapse to single spaces.
    """
    return _NORM_RX.sub(" ", (s or "").lower()).strip()


def _rxnorm_cuis_for(query: str, rrf_path: str | None = None) -> Set[str]:
    rrf_path = rrf_path or _DEFAULT_RXN_RRF

    # If a directory was given, try to resolve the file inside it
    if rrf_path and os.path.isdir(rrf_path):
        for cand in ("RXNCONSO.RRF", "rxnconso.rrf"):
            p = os.path.join(rrf_path, cand)
            if os.path.exists(p):
                rrf_path = p
                break

    cuis: Set[str] = set()
    if not rrf_path or not os.path.exists(rrf_path) or os.path.isdir(rrf_path):
        logger.warning("[RxNorm] RXNCONSO.RRF not found at: %s (CUI matching will be skipped)", rrf_path)
        return cuis

    q = _norm(query).replace(" ", "")
    with open(rrf_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 17:
                continue
            cui, lang, suppress, name = parts[0], parts[1], parts[16], parts[14]
            if lang != "ENG" or suppress == "Y":
                continue
            nm = _norm(name).replace(" ", "")
            if q in nm or nm in q:
                cuis.add(cui)
    return cuis



# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def get_cadec_drug_nodes(
    drug_label: str,
    kg_path: str | None = None,
    rxn_rrf_path: str | None = None,
    mel_top_k: int = 5,
    mel_threshold: float = 0.7,
) -> List[Tuple[str, str, Set[str]]]:
    """
    Find CADEC *drug* nodes matching the input query.

    Matching policy:
      1) Prefer RxNorm CUI intersection:
         - Map input text → set of RxNorm CUIs via RXNCONSO.RRF (loose normalization)
         - Return all CADEC drug nodes whose "cuis" intersect that set
      2) Fallback to normalized label equality / substring

    Returns:
      List of (node_id, label_lower, cuis_set)
    """
    G = _load_cadec_graph(kg_path)
    q_cuis = _rxnorm_cuis_for(drug_label, rxn_rrf_path)
    hits: List[Tuple[str, str, Set[str]]] = []

    # Pass 1: CUI intersection (if we found any CUIs for the query)
    if q_cuis:
        for node, data in G.nodes(data=True):
            if data.get("type") != "drug":
                continue
            node_cuis = set(data.get("cuis", []))
            if node_cuis & q_cuis:
                hits.append((node, (data.get("label") or "").lower(), node_cuis))
        if hits:
            logger.debug("[CADEC] %d drug node(s) matched by RxNorm CUI for query '%s'", len(hits), drug_label)
            return hits

    # Pass 2: label normalization
    q_norm = _norm(drug_label)
    for node, data in G.nodes(data=True):
        if data.get("type") != "drug":
            continue
        lbl = (data.get("label") or "")
        lbl_norm = _norm(lbl)
        if lbl_norm == q_norm or q_norm in lbl_norm:
            hits.append((node, lbl.lower(), set(data.get("cuis", []))))
    if hits:
        return hits

    # Pass 3: MEL embedding similarity search (SapBERT)
    if kg_path not in _DRUG_EMB_CACHE:
        emb_list: List[Tuple[str, str, Set[str], np.ndarray]] = []
        for node, data in G.nodes(data=True):
            if data.get("type") != "drug":
                continue
            lbl = (data.get("label") or "")
            emb_list.append((node, lbl.lower(), set(data.get("cuis", [])), encode_text(lbl)))
        _DRUG_EMB_CACHE[kg_path] = emb_list

    q_vec = encode_text(drug_label)
    sims = [float(np.dot(q_vec, emb)) for _, _, _, emb in _DRUG_EMB_CACHE[kg_path]]
    ordered = sorted(zip(_DRUG_EMB_CACHE[kg_path], sims), key=lambda x: x[1], reverse=True)
    for (node, lbl, cuis, _), sim in ordered[:mel_top_k]:
        if sim >= mel_threshold:
            hits.append((node, lbl, cuis))

    if not hits:
        logger.warning(
            "[CADEC] No drug nodes found for label='%s' (CUI+label+embedding).",
            drug_label,
        )
    return hits


def get_cadec_ae_pairs(
    drug_nodes: List[Tuple[str, str, Set[str]]],
    kg_path: str | None = None,
) -> List[Tuple[str, str, str]]:
    """
    For each input CADEC drug node, list outgoing edges to CADEC AE nodes.

    Args:
      drug_nodes: output of get_cadec_drug_nodes()
      kg_path: optional path override

    Returns:
      List of (drug_label_lower, ae_label_lower, cui_str)
    """
    if not drug_nodes:
        return []

    G = _load_cadec_graph(kg_path)
    pairs: List[Tuple[str, str, str]] = []

    # We assume edges are directed Drug -> AdverseEffect in the CADEC KG.
    for node_id, drug_label, cuis in drug_nodes:
        cui_str = ", ".join(sorted(cuis)) if cuis else ""
        # Compatible with MultiDiGraph; out_edges() yields (src, dst, data)
        for _, ae_node, data in G.out_edges(node_id, data=True):
            ndata = G.nodes.get(ae_node, {})
            if ndata.get("type") == "adverse_effect":
                ae_label = (ndata.get("label") or "").lower()
                pairs.append((drug_label, ae_label, cui_str))

    return pairs
