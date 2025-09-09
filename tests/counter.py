import os
import argparse
import logging
from functools import lru_cache
from tqdm import tqdm
from datasets import load_dataset

# Suppress library progress bars
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")
os.environ.setdefault("SENTENCE_TRANSFORMERS_PROGRESS_BAR", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BAR", "1")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Count ADE drug-AE pairs linked to CADEC KG.")
    parser.add_argument("--kg_path", type=str, default=os.getenv("KG_PATH", "drug_ae_reasoner/data/cadec/cadec_normalized_kg.gpickle"),
                        help="Path to CADEC KG .gpickle file")
    parser.add_argument("--rxn_rrf_path", type=str, default=os.getenv("RXN_RRF_PATH", "drug_ae_reasoner/data/rxnorm/RXNCONSO.RRF"),
                        help="Path to RxNorm RXNCONSO.RRF file")
    parser.add_argument("--mel_top_k", type=int, default=5, help="MEL top_k")
    parser.add_argument("--mel_threshold", type=float, default=0.8, help="MEL similarity threshold")
    return parser.parse_args()

from drug_ae_reasoner.data.cadec_loader import get_cadec_drug_nodes, get_cadec_ae_pairs

@lru_cache(maxsize=1024)
def cached_get_cadec_drug_nodes(drug, kg_path, rxn_rrf_path, mel_top_k, mel_threshold):
    return get_cadec_drug_nodes(drug, kg_path, rxn_rrf_path, mel_top_k=mel_top_k, mel_threshold=mel_threshold)

@lru_cache(maxsize=4096)
def cached_get_cadec_ae_pairs(drug_nodes_tuple, kg_path):
    return get_cadec_ae_pairs(list(drug_nodes_tuple), kg_path)

def count_links(records_iterable, kg_path, rxn_rrf_path, mel_top_k, mel_threshold):
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

    with tqdm(total=total_steps, desc="Matching ADE pairs", unit="step") as pbar:
        for drug_l, drug in unique_drugs.items():
            drug_nodes = cached_get_cadec_drug_nodes(drug, kg_path, rxn_rrf_path, mel_top_k, mel_threshold)
            drug_nodes_cache[drug_l] = drug_nodes
            if not drug_nodes:
                logger.warning(f"No match for drug: {drug}")
            pbar.update(1)

        for (drug_l, ae_l) in unique_pairs:
            total += 1
            drug_nodes = drug_nodes_cache.get(drug_l, [])
            ae_pairs = cached_get_cadec_ae_pairs(tuple([(n[0], n[1], frozenset(n[2])) for n in drug_nodes]), kg_path) if drug_nodes else []
            if any(ae_l == ae for _, ae, _ in ae_pairs):
                linked += 1
            pbar.update(1)

    return linked, total

def load_ade(split="train"):
    ds = load_dataset("SetFit/ade_corpus_v2_classification", split=split)
    print("Columns:", ds.column_names)
    for i, row in enumerate(ds):
        if i == 0:
            print("Sample row:", row)
        # Patch this line after inspecting the columns
        yield {"drug": row.get("drug", row.get("entity_1", "")), "adverse_event": row.get("adverse_effect", row.get("entity_2", ""))}

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"KG: {args.kg_path}")
    logger.info(f"RxNorm: {args.rxn_rrf_path}")
    logger.info(f"MEL: top_k={args.mel_top_k}, threshold={args.mel_threshold}")
    ade_links, ade_total = count_links(
        load_ade("train"),
        args.kg_path,
        args.rxn_rrf_path,
        args.mel_top_k,
        args.mel_threshold
    )
    print(f"\nADE: {ade_links}/{ade_total} pairs linked to CADEC ({ade_links/ade_total:.2%})")