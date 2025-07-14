import networkx as nx
import os

# Note: The `pickle` import was in the notebook's example usage, 
# but not directly needed in these utility functions.

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
            # Begin a new span
            if current_span:
                spans.append((" ".join(current_span), current_tag))
            current_span = [word]
            current_tag = tag[2:]
        elif tag.startswith("I-") and current_span:
            # Continue the current span if tag matches
            if tag[2:] == current_tag:
                current_span.append(word)
            else:
                # If we encounter an I- tag that doesn't match the current span, close the current span
                spans.append((" ".join(current_span), current_tag))
                current_span = []
                current_tag = None
        else:
            # Outside of a span ("O") or no current span to continue
            if current_span:
                spans.append((" ".join(current_span), current_tag))
                current_span = []
                current_tag = None
    # Catch any span still open at EOF
    if current_span:
        spans.append((" ".join(current_span), current_tag))
    return spans

def process_doc(doc_id, tokens, G):
    """
    Process a single document from the CADEC file.
    Extracts spans from the ADR column (index 1) and Drug column (index 3).
    If no drug span is found, uses the document ID’s prefix (e.g. "LIPITOR") as the drug.
    Adds nodes and edges (in both directions) to the graph G.
    The PMID is set to the full document ID (e.g., "LIPITOR.408").
    """
    pmid = doc_id
    adr_spans = extract_spans(tokens, col_index=1)      # ADR spans from column 2 in file (index 1)
    drug_spans = extract_spans(tokens, col_index=3)     # Drug spans from column 4 in file (index 3)
    if not drug_spans:
        # If no drug was tagged in the text, use the document prefix as a fallback drug name
        fallback_drug = doc_id.split(".")[0]
        drug_spans = [(fallback_drug, None)]
    # Add drug nodes
    for drug_text, tag in drug_spans:
        drug_node = f"drug_{drug_text.lower()}"
        if drug_node not in G:
            G.add_node(drug_node, label=drug_text, type="drug", doc=doc_id)
    # Add ADR nodes
    for adr_text, tag in adr_spans:
        adr_node = f"adr_{adr_text.lower()}"
        if adr_node not in G:
            G.add_node(adr_node, label=adr_text, type="adverse_effect", doc=doc_id)
    # Add edges between each drug and each ADR in this document (both directions)
    for drug_text, _ in drug_spans:
        drug_node = f"drug_{drug_text.lower()}"
        for adr_text, _ in adr_spans:
            adr_node = f"adr_{adr_text.lower()}"
            G.add_edge(drug_node, adr_node, relation="causes", pmid=pmid)
            G.add_edge(adr_node, drug_node, relation="adr_of", pmid=pmid)

def read_cadec_documents(filepath):
    """
    Read a CADEC .conll file and return a list of (doc_id, tokens) pairs.
    Each document begins with a line containing the doc_id, followed by token lines.
    """
    documents = []
    doc_id = None
    tokens = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # Empty line indicates end of a document
                if doc_id is not None and tokens:
                    documents.append((doc_id, tokens))
                doc_id = None
                tokens = []
            elif "\t" not in line:
                # A line with no tab is a new document ID
                if doc_id is not None and tokens:
                    documents.append((doc_id, tokens))
                    tokens = []
                doc_id = line.strip()
            else:
                # Token line with multiple tab-separated columns
                parts = line.split("\t")
                if len(parts) >= 6:  # ensure the expected number of columns
                    tokens.append(parts)
        # After loop, if last document didn't end with a blank line, add it
        if doc_id is not None and tokens:
            documents.append((doc_id, tokens))
    return documents

def build_cadec_kg_from_docs(documents):
    """
    Build a knowledge graph from a list of (doc_id, tokens) pairs using NetworkX.
    Returns a MultiDiGraph where drug and adverse_effect nodes are connected.
    """
    G = nx.MultiDiGraph()
    for doc_id, tokens in documents:
        process_doc(doc_id, tokens, G)
    return G

def dedupe_cadec(G):
    """
    Return a new MultiDiGraph with only one edge per unique (u, v, relation, pmid) tuple.
    This removes duplicate edges between the same nodes with the same relation and PMID.
    """
    H = nx.MultiDiGraph()
    H.add_nodes_from(G.nodes(data=True))  # copy all nodes with data
    seen = set()
    for u, v, data in G.edges(data=True):
        relation = data.get("relation")
        pmid = data.get("pmid")
        key = (u, v, relation, pmid)
        if key not in seen:
            seen.add(key)
            H.add_edge(u, v, **data)
    return H

def list_all_unique_drugs(G):
    """
    Returns a sorted list of unique drug names present in the KG (case-insensitive unique labels).
    """
    drugs = set()
    for node, data in G.nodes(data=True):
        if data.get("type") == "drug":
            label = data.get("label")
            if label:
                drugs.add(label.upper())
    return sorted(drugs)
