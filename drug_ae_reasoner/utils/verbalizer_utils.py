import networkx as nx
import os

# Note: The `pickle` import was in the notebook's main/example usage, 
# but not directly used by the core functions needed here.
# It's used by build_cadec_kg.py, which imports these utils.

def extract_spans(tokens, col_index):
    """
    Given tokens (a list of lists) and a column index, extract spans using BIO tagging.
    Returns a list of (span_text, tag_id) tuples.
    """
    spans = []
    current_span = []
    current_tag = None
    for parts in tokens:
        word = parts[0]
        tag = parts[col_index]
        if tag.startswith("B-"):
            if current_span:
                spans.append((" ".join(current_span), current_tag))
            current_span = [word]
            current_tag = tag[2:]
        elif tag.startswith("I-") and current_span:
            if tag[2:] == current_tag:
                current_span.append(word)
            else:
                spans.append((" ".join(current_span), current_tag))
                current_span = []
                current_tag = None
        else:
            if current_span:
                spans.append((" ".join(current_span), current_tag))
                current_span = []
                current_tag = None
    if current_span:
        spans.append((" ".join(current_span), current_tag))
    return spans

def process_doc(doc_id, tokens, G):
    """
    Process a single document from the CADEC file.
    
    Extracts spans from the ADR column (index 1) and Drug column (index 3).
    If no drug span is found, uses the document ID’s prefix (e.g. "LIPITOR") as the drug.
    Adds nodes and edges (both directions) to the graph G.
    
    The PMID is set to the full document ID (e.g., "LIPITOR.408" or "ARTHROTEC.36").
    """
    pmid = doc_id
    adr_spans = extract_spans(tokens, 1)
    drug_spans = extract_spans(tokens, 3)
    
    if not drug_spans:
        fallback_drug = doc_id.split(".")[0]
        drug_spans = [(fallback_drug, None)]
    
    for drug_text, tag in drug_spans:
        drug_node = f"drug_{drug_text.lower()}"
        if drug_node not in G:
            G.add_node(drug_node, label=drug_text, type="drug", doc=doc_id)
    for adr_text, tag in adr_spans:
        adr_node = f"adr_{adr_text.lower()}"
        if adr_node not in G:
            G.add_node(adr_node, label=adr_text, type="adverse_effect", doc=doc_id)
    
    for drug_text, _ in drug_spans:
        drug_node = f"drug_{drug_text.lower()}"
        for adr_text, _ in adr_spans:
            adr_node = f"adr_{adr_text.lower()}"
            G.add_edge(drug_node, adr_node, relation="causes", pmid=pmid)
            G.add_edge(adr_node, drug_node, relation="adr_of", pmid=pmid)

def read_cadec_documents(filepath):
    """
    Read CADEC file and return a list of (doc_id, tokens) pairs.
    Each document is represented as (doc_id, list_of_tokens).
    """
    documents = []
    doc_id = None
    tokens = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if doc_id is not None and tokens:
                    documents.append((doc_id, tokens))
                doc_id = None
                tokens = []
            elif "\t" not in line:
                if doc_id is not None and tokens:
                    documents.append((doc_id, tokens))
                    tokens = []
                doc_id = line.strip()
            else:
                parts = line.split("\t")
                if len(parts) >= 6: # Ensure enough parts before appending
                    tokens.append(parts)
        if doc_id is not None and tokens: # Process the last document
            documents.append((doc_id, tokens))
    return documents

def build_cadec_kg_from_docs(documents):
    """
    Build a KG from a list of (doc_id, tokens) pairs.
    """
    G = nx.MultiDiGraph()
    for doc_id, tokens in documents:
        process_doc(doc_id, tokens, G)
    return G

def dedupe_cadec(G):
    """
    Return a new MultiDiGraph with only one edge per unique
    (u, v, relation, pmid) tuple.
    """
    H = nx.MultiDiGraph()
    H.add_nodes_from(G.nodes(data=True))
    seen = set()
    for u, v, data in G.edges(data=True):
        # Ensure 'relation' and 'pmid' are present in data, otherwise use a default or skip
        relation = data.get("relation")
        pmid = data.get("pmid")
        key = (u, v, relation, pmid)
        if key not in seen:
            seen.add(key)
            H.add_edge(u, v, **data)
    return H

def list_all_unique_drugs(G):
    """
    Returns a sorted list of unique drug names (by their label) in the KG.
    """
    drugs = set()
    for node, data in G.nodes(data=True):
        if data.get("type") == "drug":
            # Ensure 'label' is present and handle potential errors if it's not
            label = data.get("label")
            if label:
                drugs.add(label.upper())
    return sorted(list(drugs))
