# 🧠 `drug_ae_reasoner`

*Semantic Path Tracing from Drugs to Adverse Events using CADEC, RxNorm, and OAE Knowledge Graphs*

---

## 🚀 Overview

`drug_ae_reasoner` is a Python package that connects **drug mentions** to **user-reported adverse effects (AEs)** through a multi-source reasoning pipeline involving:

* RxNorm-based normalization of drug names with SapBERT MEL fallback
* CADEC forum-based adverse effect knowledge graph
* OAE ontology graph & SapBERT semantic mapping
* Path discovery and verbalization from CADEC→OAE→Input AEs

---

## 📦 Installation

### ▶️ Clone the Repo and Set Up Virtual Environment

```bash
git clone https://github.com/showman-sharma/drug_ae_reasoner.git
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


## 🧬 Drug Mention-Entity Linking (MEL)

When you provide a drug name, the system tries to map it to a node in the CADEC knowledge graph using a multi-step process:

1. **RxNorm Normalization:**
    Attempts to match the input drug to a standard RxNorm CUI using exact and synonym matches.

2. **SapBERT Embedding Search (MEL):**
    If no exact match is found, the system uses SapBERT (a biomedical language model) to embed your input and searches for the most similar drug nodes using a FAISS index.
    - You can control the number of candidates and similarity threshold with `--mel_top_k` and `--mel_threshold`.
    - To disable embedding-based search, use `--no_embedding`.

3. **Fallbacks:**
    If neither method finds a match, the system will report that no drug node was found.

**Tip:**
If your drug input is not found, try using a generic name, check spelling, or lower the MEL threshold for fuzzier matching.

---

Once all files are in place, run the unified data setup pipeline:

```bash
python -m drug_ae_reasoner.data.builder.run_all
```


This script performs:

* CADEC KG creation from the full dataset
* Drug normalization using RxNorm
* OAE embedding + FAISS indexing
* OWL to NetworkX graph conversion

### 🔄 Output Files

| File                          | Folder        | Description                          |
| ----------------------------- | ------------- | ------------------------------------ |
| `cadec_verbalizer_kg.gpickle` | `data/cadec/` | CADEC drug–AE graph (full dataset)   |
| `cadec_normalized_kg.gpickle` | `data/cadec/` | Normalized with RxNorm CUIs          |
| `oae_sapbert_index.faiss`     | `data/oae/`   | FAISS index for OAE label embeddings |
| `oae_labels.pkl`              | `data/oae/`   | Label map for FAISS vectors          |
| `oae_graph.gpickle`           | `data/oae/`   | Directed ontology graph from OAE.owl |

---

## 🧪 CLI Usage


Once installed and built, run the reasoning CLI:



```bash
python -m drug_ae_reasoner.main \
    --drug lipitor \
    --aes pain joints \
    --mel_top_k 5 \
    --mel_threshold 0.5 \
    --n_paths 3 \
    --n_cadec 3 \
    --n_input 3 \
    --cadec_ae_thresh 0.7 \
    --input_ae_thresh 0.7 \
    --n_disconnect 2
```

You can also override the default OAE resources, e.g.:

This will:

* Normalize the drug
* Trace paths from drug → CADEC AE → OAE node → input AE
* Rank based on semantic similarity
* Print verbalized paths with similarity scores

You can also override the default OAE resources, e.g.:


```bash
python -m drug_ae_reasoner.main \
    --drug lipitor \
    --aes pain \
    --oae_index_path path/to/index.faiss \
    --oae_label_map_path path/to/labels.pkl \
    --mel_top_k 5 \
    --mel_threshold 0.5
```

---

## 📚 Python API Usage

You can also use the system directly in Python:

```python
from drug_ae_reasoner.utils.path_reasoner import find_top_drug_to_input_ae_paths
from drug_ae_reasoner.config import (
    RX_PATH,
    CADEC_KG_PATH,
    OAE_INDEX_PATH,
    OAE_LABEL_MAP_PATH,
    OAE_GRAPH_PATH,
)

# find_top_drug_to_input_ae_paths performs the full reasoning pipeline, including drug MEL.
# You can control MEL behavior with mel_top_k, mel_threshold, and use_embedding.
connected, top_paths, fb_drug, fb_ae, verb = find_top_drug_to_input_ae_paths(
    drug="lipitor",
    ae_input_list=["pain", "joints"],
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
    n_disconnect=3,
    mel_top_k=5,           # MEL: number of drug candidates to consider
    mel_threshold=0.5,     # MEL: similarity threshold for drug mapping
    use_embedding=True,    # MEL: use SapBERT embedding-based search
)

print("\n".join(verb))
```

## 🔬 Research Features

### Three-Path Classification System

The system includes advanced research capabilities for ADR dataset annotation with three-class path classification:

- **Positive paths** (`score ≥ 0.6`): Strong evidence for drug-AE relationship
- **Negative paths** (`0 < score < 0.6`): Weak/contradictory evidence  
- **Random paths** (`score ≤ 0`): Control/baseline paths

Each drug-AE pair generates exactly 3 paths (one per class) with structured output:

```json
{
  "drug": "lipitor",
  "adr": "muscle pain", 
  "paths": [
    {
      "mapped_drug_concept": "lipitor",
      "mapped_ae_concept": "muscle problems",
      "oae_nodes": ["muscle problems"],
      "path_type": "positive",
      "path_weight": 0.87,
      "relation": "ADR",
      "verbalization": "lipitor → muscle problems (OAE) ~ 'muscle pain' [sim(cadec→oae_from)=0.89; sim(oae_to→input)=0.85; score=0.87]"
    }
  ]
}
```

This enables systematic evaluation of knowledge graph reasoning performance across different evidence strength levels.

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

