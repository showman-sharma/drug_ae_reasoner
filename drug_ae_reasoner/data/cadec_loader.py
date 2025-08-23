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
from collections import defaultdict
from typing import Any, List, Tuple, Set, Dict

import networkx as nx
import numpy as np

from ..utils.encoding import encode_text

# Pull defaults from project config, but keep functions parametric so callers can override.
from ..config import (
    CADEC_KG_PATH as _DEFAULT_CADEC_KG,
    RX_PATH as _DEFAULT_RXN_RRF,
    ENABLE_EMBEDDING_SEARCH as _USE_EMB_DEFAULT,
    REQUIRE_EMBEDDING_CONFIRMATION as _CONFIRM_EMBED_DEFAULT,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Graph cache
# ──────────────────────────────────────────────────────────────────────────────
_CADEC_KG_CACHE: Dict[str, nx.MultiDiGraph] = {}
# Cache for precomputed drug label embeddings per KG path
_DRUG_EMB_CACHE: Dict[str, List[Tuple[str, str, Set[str], np.ndarray]]] = {}
# Cache for RxNorm data keyed by directory
_RXN_CACHE: Dict[str, Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]] = {}


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


def _load_rxnorm_maps(rrf_path: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Load RxNorm maps: CUI->names, brand->generic, generic->brand."""
    dir_path = rrf_path
    if os.path.isfile(rrf_path):
        dir_path = os.path.dirname(rrf_path)
    if dir_path in _RXN_CACHE:
        return _RXN_CACHE[dir_path]

    cui_to_names: Dict[str, Set[str]] = defaultdict(set)
    conso = os.path.join(dir_path, "RXNCONSO.RRF")
    if os.path.exists(conso):
        with open(conso, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                parts = ln.rstrip("\n").split("|")
                if len(parts) < 17:
                    continue
                cui, lang, supp, name = parts[0], parts[1], parts[16], parts[14]
                if lang == "ENG" and supp != "Y":
                    cui_to_names[cui].add(name.lower())

    brand_to_generic: Dict[str, Set[str]] = defaultdict(set)
    generic_to_brand: Dict[str, Set[str]] = defaultdict(set)
    rel = os.path.join(dir_path, "RXNREL.RRF")
    if os.path.exists(rel):
        with open(rel, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                parts = ln.rstrip("\n").split("|")
                if len(parts) < 8:
                    continue
                cui1, cui2, rela = parts[0], parts[4], parts[7]
                if rela == "tradename_of":
                    brand_to_generic[cui1].add(cui2)
                    generic_to_brand[cui2].add(cui1)
                elif rela == "has_tradename":
                    brand_to_generic[cui2].add(cui1)
                    generic_to_brand[cui1].add(cui2)

    _RXN_CACHE[dir_path] = (cui_to_names, brand_to_generic, generic_to_brand)
    return _RXN_CACHE[dir_path]


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

    # Expand brand CUIs to generic CUIs
    _, brand_to_generic, _ = _load_rxnorm_maps(rrf_path)
    expanded = set(cuis)
    for c in list(cuis):
        expanded.update(brand_to_generic.get(c, set()))
    return expanded


def _merge_rxnorm_synonyms(G: nx.MultiDiGraph, rrf_path: str | None = None) -> bool:
    """Merge RxNorm brand/generic names into drug node 'synonyms'.

    Returns True if any node was updated.
    """
    rrf_path = rrf_path or _DEFAULT_RXN_RRF
    if not rrf_path:
        return False
    cui_to_names, _, generic_to_brand = _load_rxnorm_maps(rrf_path)
    if not cui_to_names:
        return False
    updated = False
    for _, data in G.nodes(data=True):
        if data.get("type") != "drug":
            continue
        node_cuis = set(data.get("cuis", []))
        if not node_cuis:
            continue
        syns = set(data.get("synonyms", []))
        for cui in node_cuis:
            syns.update(cui_to_names.get(cui, []))
            for b in generic_to_brand.get(cui, []):
                syns.update(cui_to_names.get(b, []))
        if syns and set(data.get("synonyms", [])) != syns:
            data["synonyms"] = sorted(syns)
            updated = True
    return updated



# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def get_cadec_drug_nodes(
    drug_label: str,
    kg_path: str | None = None,
    rxn_rrf_path: str | None = None,
    mel_top_k: int = 5,
    mel_threshold: float = 0.7,
    use_embedding: bool = _USE_EMB_DEFAULT,
    mel_require_confirmation: bool = _CONFIRM_EMBED_DEFAULT,
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
    # Ensure RxNorm synonyms are merged for label matching
    if _merge_rxnorm_synonyms(G, rxn_rrf_path):
        _DRUG_EMB_CACHE.pop(kg_path, None)

    q_cuis = _rxnorm_cuis_for(drug_label, rxn_rrf_path)
    q_norm = _norm(drug_label)
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

    # Pass 2: label normalization (including RxNorm synonyms)
    for node, data in G.nodes(data=True):
        if data.get("type") != "drug":
            continue
        cuis = set(data.get("cuis", []))
        names = [data.get("label") or ""] + list(data.get("synonyms", []))
        seen = set()
        for nm in names:
            nm_l = nm.lower()
            if nm_l in seen:
                continue
            seen.add(nm_l)
            nm_norm = _norm(nm)
            if nm_norm == q_norm or q_norm in nm_norm:
                hits.append((node, nm_l, cuis))
                break
    if hits:
        return hits

    # Pass 3: MEL embedding similarity search (SapBERT)
    if use_embedding and mel_top_k > 0:
        if kg_path not in _DRUG_EMB_CACHE:
            emb_list: List[Tuple[str, str, Set[str], np.ndarray]] = []
            for node, data in G.nodes(data=True):
                if data.get("type") != "drug":
                    continue
                cuis = set(data.get("cuis", []))
                names = [data.get("label") or ""] + list(data.get("synonyms", []))
                seen = set()
                for nm in names:
                    nm_l = nm.lower()
                    if nm_l in seen:
                        continue
                    seen.add(nm_l)
                    emb_list.append((node, nm_l, cuis, encode_text(nm)))
            _DRUG_EMB_CACHE[kg_path] = emb_list

        q_vec = encode_text(drug_label)
        sims = [float(np.dot(q_vec, emb)) for _, _, _, emb in _DRUG_EMB_CACHE[kg_path]]
        ordered = sorted(zip(_DRUG_EMB_CACHE[kg_path], sims), key=lambda x: x[1], reverse=True)
        for (node, lbl, cuis, _), sim in ordered[:mel_top_k]:
            if sim >= mel_threshold:
                if mel_require_confirmation:
                    lbl_norm = _norm(lbl)
                    if q_cuis & cuis or lbl_norm == q_norm or q_norm in lbl_norm:
                        hits.append((node, lbl, cuis))
                else:
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
