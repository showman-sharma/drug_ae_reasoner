import os
import pickle
import faiss
import numpy as np
import rdflib
from rdflib.namespace import RDFS
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

def encode(text):
    vec = model.encode([text])[0]
    return vec / np.linalg.norm(vec)

def extract_labels(owl_path):
    g = rdflib.Graph()
    g.parse(owl_path, format='xml')
    return sorted({str(o).strip().lower() for s, _, o in g.triples((None, RDFS.label, None))})

def main():
    oae_dir = os.path.join("drug_ae_reasoner", "data", "oae")
    owl_path = os.path.join(oae_dir, "oae_merged.owl")
    labels = extract_labels(owl_path)

    vecs = [encode(label).astype(np.float32) for label in tqdm(labels, desc="Encoding")]
    xb = np.vstack(vecs)

    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)

    index_path = os.path.join(oae_dir, "oae_sapbert_index.faiss")
    label_path = os.path.join(oae_dir, "oae_labels.pkl")

    faiss.write_index(index, index_path)
    with open(label_path, "wb") as f:
        pickle.dump(labels, f)

    print(f"FAISS index saved: {index_path}")
    print(f"Label map saved: {label_path}")

if __name__ == "__main__":
    main()
