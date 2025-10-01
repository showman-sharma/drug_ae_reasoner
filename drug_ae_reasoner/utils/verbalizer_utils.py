import networkx as nx
import os

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
    Process one CADEC document: extract drug & ADR spans,
    add nodes and bidirectional edges to graph G.
    """
    pmid = doc_id
    adr_spans = extract_spans(tokens, col_index=1)
    drug_spans = extract_spans(tokens, col_index=3)
    if not drug_spans:
        fallback = doc_id.split(".")[0]
        drug_spans = [(fallback, None)]
    # Add drug nodes
    import re
    def normalize_drug_name(name):
        # Remove punctuation, extra spaces, and modifiers like '#', numbers
        name = name.lower()
        name = re.sub(r"[^a-z0-9 ]", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        # Remove trailing numbers and hash (e.g., 'tylenol # 3' -> 'tylenol')
        name = re.sub(r"\s*#?\s*\d+$", "", name)
        return name

    for drug, _ in drug_spans:
        norm_drug = normalize_drug_name(drug)
        node = f"drug_{norm_drug}"
        if node not in G:
            G.add_node(node, label=norm_drug, type="drug", doc=doc_id)
    # Add ADR nodes
    for adr, _ in adr_spans:
        node = f"adr_{adr.lower()}"
        if node not in G:
            G.add_node(node, label=adr, type="adverse_effect", doc=doc_id)
    # Add edges
    for drug, _ in drug_spans:
        dnode = f"drug_{drug.lower()}"
        for adr, _ in adr_spans:
            anode = f"adr_{adr.lower()}"
            G.add_edge(dnode, anode, relation="causes", pmid=pmid)
            G.add_edge(anode, dnode, relation="adr_of", pmid=pmid)

def read_cadec_documents(filepath):
    """
    Read a CADEC .conll file and return a list of (doc_id, tokens) pairs.
    """
    documents, doc_id, tokens = [], None, []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if doc_id and tokens:
                    documents.append((doc_id, tokens))
                doc_id, tokens = None, []
            elif "\t" not in line:
                if doc_id and tokens:
                    documents.append((doc_id, tokens))
                doc_id = line
                tokens = []
            else:
                parts = line.split("\t")
                if len(parts) >= 6:
                    tokens.append(parts)
        if doc_id and tokens:
            documents.append((doc_id, tokens))
    return documents

def build_cadec_kg_from_docs(documents):
    """
    Build a NetworkX MultiDiGraph from a list of (doc_id, tokens).
    """
    G = nx.MultiDiGraph()
    for doc_id, tokens in documents:
        process_doc(doc_id, tokens, G)
    return G

def dedupe_cadec(G):
    """
    Remove duplicate edges, keeping only unique (u, v, relation, pmid).
    """
    H = nx.MultiDiGraph()
    H.add_nodes_from(G.nodes(data=True))
    seen = set()
    for u, v, data in G.edges(data=True):
        key = (u, v, data.get("relation"), data.get("pmid"))
        if key not in seen:
            seen.add(key)
            H.add_edge(u, v, **data)
    return H

def list_all_unique_drugs(G):
    """
    Return sorted list of unique drug labels (uppercase) in the KG.
    """
    drugs = {
        data["label"].upper()
        for node, data in G.nodes(data=True)
        if data.get("type") == "drug" and data.get("label")
    }
    return sorted(drugs)
