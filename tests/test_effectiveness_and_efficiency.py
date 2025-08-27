import time
import pytest
from drug_ae_reasoner.main import find_top_drug_to_input_ae_paths
from drug_ae_reasoner.config import RX_PATH, CADEC_KG_PATH, OAE_INDEX_PATH, OAE_LABEL_MAP_PATH, OAE_GRAPH_PATH

# Realistic drug-AE test cases (effectiveness)
EXAMPLES = [
    ("paracetamol", ["rash"]),
    ("atorvastatin", ["muscle pain"]),
    ("amoxicillin", ["diarrhea"]),
    ("ibuprofen", ["gastric ulcer"]),
    ("metformin", ["nausea"]),
]

@pytest.mark.parametrize("drug,aes", EXAMPLES)
def test_effectiveness_and_efficiency(drug, aes):
    # Cold run (includes model/index load)
    t0 = time.perf_counter()
    connected, top_paths, *_ = find_top_drug_to_input_ae_paths(
        drug=drug,
        ae_input_list=aes,
        rx_path=RX_PATH,
        cadec_kg_path=CADEC_KG_PATH,
        oae_index_path=OAE_INDEX_PATH,
        oae_label_map_path=OAE_LABEL_MAP_PATH,
        oae_graph_path=OAE_GRAPH_PATH,
        n_paths=3,
        n_cadec=3,
        n_input=3,
        cadec_ae_threshold=0.7,
        input_ae_threshold=0.7,
        n_disconnect=2,
        mel_top_k=3,
        mel_threshold=0.6,
        use_embedding=True,
    )
    t1 = time.perf_counter()
    # Warm run (should be much faster)
    t2 = time.perf_counter()
    connected2, top_paths2, *_ = find_top_drug_to_input_ae_paths(
        drug=drug,
        ae_input_list=aes,
        rx_path=RX_PATH,
        cadec_kg_path=CADEC_KG_PATH,
        oae_index_path=OAE_INDEX_PATH,
        oae_label_map_path=OAE_LABEL_MAP_PATH,
        oae_graph_path=OAE_GRAPH_PATH,
        n_paths=3,
        n_cadec=3,
        n_input=3,
        cadec_ae_threshold=0.7,
        input_ae_threshold=0.7,
        n_disconnect=2,
        mel_top_k=3,
        mel_threshold=0.6,
        use_embedding=True,
    )
    t3 = time.perf_counter()

    # Effectiveness: Should find at least one path for known pairs
    assert connected or connected2, f"No path found for {drug} + {aes}"
    assert top_paths or top_paths2, f"No top paths for {drug} + {aes}"

    # Efficiency: Warm run should be fast (<1s)
    warm_time = t3 - t2
    assert warm_time < 1.0, f"Warm inference too slow: {warm_time:.2f}s"

    print(f"{drug} + {aes}: cold={t1-t0:.2f}s, warm={warm_time:.2f}s, paths={len(top_paths2)}")
