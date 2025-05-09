# 🧠 `drug_ae_reasoner`

*Semantic Path Tracing from Drugs to Adverse Events using CADEC, RxNorm, and OAE Knowledge Graphs*

---

## 🚀 Overview

`drug_ae_reasoner` is a Python package that connects **drug mentions** to **user-reported adverse effects (AEs)** through a multi-source reasoning pipeline involving:

* RxNorm-based normalization of drug names
* CADEC forum-based adverse effect knowledge graph
* OAE ontology graph & SapBERT semantic mapping
* Path discovery and verbalization from CADEC→OAE→Input AEs

---

## 📦 Installation

### ▶️ Clone the Repo and Set Up Virtual Environment

```bash
git clone https://github.com/your-username/drug_ae_reasoner.git
cd drug_ae_reasoner
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -e .
```

---

## 📥 Dataset Requirements & Setup

To use the system, you must manually download and place three external biomedical datasets into designated folders.

| Dataset | File Needed      | Download Link                                                                                    | Destination Folder              |
| ------- | ---------------- | ------------------------------------------------------------------------------------------------ | ------------------------------- |
| CADEC   | `train.conll`    | [CADEC GitHub](https://github.com/gabrielStanovsky/CADEC-for-NLP/tree/master/data)               | `drug_ae_reasoner/data/cadec/`  |
| RxNorm  | `RXNCONSO.RRF`   | [RxNorm UMLS](https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html)                | `drug_ae_reasoner/data/rxnorm/` |
| OAE     | `oae_merged.owl` | [OAE Ontology OWL](https://raw.githubusercontent.com/OAE-ontology/OAE/master/src/oae_merged.owl) | `drug_ae_reasoner/data/oae/`    |

> ⚠️ These files are **not bundled** in the repo due to licensing and distribution terms. Please download them yourself.

---

### ✅ Folder Structure After Download

```
drug_ae_reasoner/
└── data/
    ├── cadec/
    │   └── train.conll
    ├── rxnorm/
    │   └── RXNCONSO.RRF
    ├── oae/
    │   └── oae_merged.owl
```

---

## 🛠️ Building the Knowledge Resources

Once all files are in place, run the unified data setup pipeline:

```bash
python -m drug_ae_reasoner.data.builder.run_all
```

This script performs:

* CADEC KG creation + split
* Drug normalization using RxNorm
* OAE embedding + FAISS indexing
* OWL to NetworkX graph conversion

### 🔄 Output Files

| File                          | Folder        | Description                          |
| ----------------------------- | ------------- | ------------------------------------ |
| `cadec_verbalizer_kg.gpickle` | `data/cadec/` | Raw CADEC drug–AE graph              |
| `cadec_normalized_kg.gpickle` | `data/cadec/` | Normalized with RxNorm CUIs          |
| `train_30.jsonl`              | `data/cadec/` | Raw training subset (30%)            |
| `oae_sapbert_index.faiss`     | `data/oae/`   | FAISS index for OAE label embeddings |
| `oae_labels.pkl`              | `data/oae/`   | Label map for FAISS vectors          |
| `oae_graph.gpickle`           | `data/oae/`   | Directed ontology graph from OAE.owl |

---

## 🧪 CLI Usage

Once installed and built, run the reasoning CLI:

```bash
drug_ae_reasoner --drug metformin --aes nausea vomiting
```

This will:

* Normalize the drug
* Trace paths from drug → CADEC AE → OAE node → input AE
* Rank based on semantic similarity
* Print verbalized paths with similarity scores

---

## 📚 Python API Usage

You can also use the system directly in Python:

```python
from drug_ae_reasoner.reasoning import find_top_drug_to_input_ae_paths
from drug_ae_reasoner.config import RX_PATH, CADEC_KG_PATH, OAE_INDEX_PATH, OAE_LABEL_MAP_PATH, OAE_GRAPH_PATH

connected, top_paths, fb_drug, fb_ae, verb = find_top_drug_to_input_ae_paths(
    drug="metformin",
    ae_input_list=["nausea", "vomiting"],
    rx_path=RX_PATH,
    cadec_kg_path=CADEC_KG_PATH,
    oae_index_path=OAE_INDEX_PATH,
    oae_label_map_path=OAE_LABEL_MAP_PATH,
    oae_graph_path=OAE_GRAPH_PATH,
    n_cadec=5,
    cadec_ae_threshold=0.7,
    n_input=5,
    input_ae_threshold=0.7,
    n_paths=5,
    n_disconnect=3
)

print("\n".join(verb))
```

---

## 📎 Notes

* SapBERT is automatically downloaded on first use (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`)
* All paths and configs are centralized in `drug_ae_reasoner/config.py`
* Model caching is handled under `~/.cache/torch/sentence_transformers/`

---

## 🧾 License & Attribution

* CADEC Corpus: [G. Stanovsky et al.](https://github.com/gabrielStanovsky/CADEC-for-NLP)
* RxNorm Data: [U.S. National Library of Medicine](https://www.nlm.nih.gov/)
* OAE Ontology: [Ontology of Adverse Events](http://www.oae-ontology.org/)

