import os
from typing import Set, Dict
import logging

logger = logging.getLogger(__name__)

def load_rxnorm(rrf_dir: str) -> Dict[str, Set[str]]:
    logger.info("Loading RxNorm mappings...")
    cui_to_names = {}
    with open(os.path.join(rrf_dir, "RXNCONSO.RRF"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split("|")
            cui, lang, suppress, name = parts[0], parts[1], parts[16], parts[14]
            if lang != "ENG" or suppress == "Y":
                continue
            cui_to_names.setdefault(cui, set()).add(name.lower())
    logger.info(f"Loaded {len(cui_to_names)} CUIs")
    return cui_to_names

def get_input_cuis(drug: str, rrf_dir: str) -> Set[str]:
    mapping = load_rxnorm(rrf_dir)
    norm = drug.lower()
    matched = {c for c, names in mapping.items() if any(norm in nm for nm in names)}
    if not matched:
        raise ValueError(f"No RxNorm CUI found for '{drug}'")
    logger.info(f"Matched CUIs for drug '{drug}': {matched}")
    return matched
