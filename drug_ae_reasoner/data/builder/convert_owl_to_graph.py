import os
import pickle
import rdflib
import networkx as nx
from rdflib.namespace import RDFS

def owl_to_graph(owl_path):
    g = rdflib.Graph()
    g.parse(owl_path, format='xml')
    G = nx.MultiDiGraph()
    ent_to_label = {}

    for s, _, o in g.triples((None, RDFS.label, None)):
        lbl = str(o).strip().lower()
        ent_to_label[s] = lbl
        G.add_node(lbl)

    for c, _, p in g.triples((None, RDFS.subClassOf, None)):
        if c in ent_to_label and p in ent_to_label:
            G.add_edge(ent_to_label[c], ent_to_label[p], relation='subClassOf')

    return G

def main():
    oae_dir = os.path.join("drug_ae_reasoner", "data", "oae")
    owl_path = os.path.join(oae_dir, "oae_merged.owl")
    graph_path = os.path.join(oae_dir, "oae_graph.gpickle")

    G = owl_to_graph(owl_path)
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    print(f"OAE graph saved to: {graph_path}")

if __name__ == "__main__":
    main()
