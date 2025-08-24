# Patch only the tqdm progress bar display methods, not the whole module
# --- Quiet library progress bars, keep our own --------------------------------
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")
os.environ.setdefault("SENTENCE_TRANSFORMERS_PROGRESS_BAR", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BAR", "1")

# Provide a subclassable no-op tqdm for libs that import tqdm.auto and/or subclass it
import tqdm.auto as _tqa

class _NoOpTQDM:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        self.disable = True
    def __iter__(self):
        return iter(self.iterable or [])
    def update(self, *a, **k): pass
    def close(self, *a, **k): pass
    def set_description(self, *a, **k): pass
    def set_postfix(self, *a, **k): pass
    def write(self, *a, **k): pass
    def reset(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): pass

# Patch only the auto variant used by libraries (keeps std tqdm intact)
_tqa.tqdm = _NoOpTQDM

# Use the real tqdm for *our* progress bars
from tqdm.std import tqdm  # IMPORTANT: don't re-import from tqdm.auto later

# Logging: keep ours; quiet down libraries
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
for _name in ("transformers", "sentence_transformers", "huggingface_hub", "datasets", "urllib3", "httpx"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Force sentence-transformers encode() to hide bars even if caller forgets
try:
    from sentence_transformers import SentenceTransformer as _ST
    _orig_encode = _ST.encode
    def _quiet_encode(self, sentences, *args, **kwargs):
        kwargs.setdefault("show_progress_bar", False)
        return _orig_encode(self, sentences, *args, **kwargs)
    _ST.encode = _quiet_encode
except Exception:
    pass
# ------------------------------------------------------------------------------

import glob
import xml.etree.ElementTree as ET
from datasets import load_dataset
from functools import lru_cache

# ------------------------------------------------------------------
# Update these paths for your environment
# ------------------------------------------------------------------
KG_PATH      = r"D:/projects/CADEC-for-NLP/drug_ae_reasoner_package/drug_ae_reasoner/data/cadec/cadec_normalized_kg.gpickle"
RXN_RRF_PATH = r"D:/projects/CADEC-for-NLP/drug_ae_reasoner_package/drug_ae_reasoner/data/rxnorm/RXNCONSO.RRF"
TAC2017_DIR  = r"D:/projects/CADEC-for-NLP/drug_ae_reasoner_package/data/tac2017/train_xml"   # folder with *.xml

# ------------------------------------------------------------------
from drug_ae_reasoner.data.cadec_loader import get_cadec_drug_nodes, get_cadec_ae_pairs

@lru_cache(maxsize=1024)
def cached_get_cadec_drug_nodes(drug, kg_path, rxn_rrf_path, mel_top_k=5, mel_threshold=0.8):
    return get_cadec_drug_nodes(drug, kg_path, rxn_rrf_path, mel_top_k=mel_top_k, mel_threshold=mel_threshold)

@lru_cache(maxsize=4096)
def cached_get_cadec_ae_pairs(drug_nodes_tuple, kg_path):
    # lru_cache requires hashable arguments, so drug_nodes must be a tuple
    return get_cadec_ae_pairs(list(drug_nodes_tuple), kg_path)

def count_links(records_iterable):
    """
    Single global tqdm bar that covers:
      1) Preprocessing each unique drug (node mapping)
      2) Checking each unique (drug, AE) pair for linkage
    """
    # Materialize records once, compute work totals for a single global bar
    records = list(records_iterable)
    unique_drugs = {}
    unique_pairs = set()
    for rec in records:
        d = rec["drug"]; ae = rec["adverse_event"]
        unique_drugs.setdefault(d.lower(), d)
        unique_pairs.add((d.lower(), ae.lower()))

    total_steps = len(unique_drugs) + len(unique_pairs)

    seen = set()
    total = linked = 0
    drug_nodes_cache = {}

    with tqdm(total=total_steps, desc="Global pipeline progress", unit="step") as pbar:
        # Phase 1: preprocess drugs → cache mapped nodes
        for drug_l, drug in unique_drugs.items():
            logger.info(f"Processing drug: {drug}")
            drug_nodes = cached_get_cadec_drug_nodes(drug, KG_PATH, RXN_RRF_PATH, 5, 0.8)
            drug_nodes_cache[drug_l] = drug_nodes
            if not drug_nodes:
                logger.warning(f"No match for drug: {drug}")
            else:
                logger.info(f"Matched nodes for drug {drug}: {[n[1] for n in drug_nodes]}")
            pbar.update(1)  # one step per drug

        # Phase 2: evaluate unique (drug, AE) pairs
        for (drug_l, ae_l) in unique_pairs:
            total += 1
            drug_nodes = drug_nodes_cache.get(drug_l, [])
            ae_pairs = cached_get_cadec_ae_pairs(tuple([(n[0], n[1], frozenset(n[2])) for n in drug_nodes]), KG_PATH) if drug_nodes else []
            if any(ae_l == ae for _, ae, _ in ae_pairs):
                logger.info(f"Linked: ({unique_drugs.get(drug_l, drug_l)}, {ae_l})")
                linked += 1
            pbar.update(1)  # one step per pair

    return linked, total

# ------------------------------------------------------------------
def load_ade(split="train"):
    ds = load_dataset("SetFit/ade_corpus_v2_classification", split=split)
    for row in ds:
        yield {"drug": row["drug"], "adverse_event": row["adverse_effect"]}

def load_tac2017(xml_dir):
    for path in glob.glob(f"{xml_dir}/*.xml"):
        root = ET.parse(path).getroot()
        drug = root.attrib["drug"]
        for m in root.findall("./Mentions/Mention"):
            if m.attrib.get("type") == "AdverseReaction":
                yield {"drug": drug, "adverse_event": m.attrib["str"]}

# ------------------------------------------------------------------
if __name__ == "__main__":
    # ade_links, ade_total = count_links(load_ade("train"))
    tac_links, tac_total = count_links(load_tac2017(TAC2017_DIR))
    # print(f"ADE: {ade_links}/{ade_total} pairs linked to CADEC")
    print(f"TAC2017: {tac_links}/{tac_total} pairs linked to CADEC")
