import os

PACKAGE_DIR = os.path.dirname(__file__)

# Data folders
RX_PATH = os.path.join(PACKAGE_DIR, "data", "rxnorm")
CADEC_KG_PATH = os.path.join(PACKAGE_DIR, "data", "cadec", "cadec_normalized_kg.gpickle")
OAE_INDEX_PATH = os.path.join(PACKAGE_DIR, "data", "oae", "oae_sapbert_index.faiss")
OAE_LABEL_MAP_PATH = os.path.join(PACKAGE_DIR, "data", "oae", "oae_labels.pkl")
OAE_GRAPH_PATH = os.path.join(PACKAGE_DIR, "data", "oae", "oae_graph.gpickle")  # ✅ Add this line
