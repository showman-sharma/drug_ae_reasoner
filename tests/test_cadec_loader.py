import os
import sys
import types
from unittest.mock import patch

import networkx as nx
import numpy as np

_dummy_enc = types.ModuleType('drug_ae_reasoner.utils.encoding')
_dummy_enc.encode_text = lambda *args, **kwargs: np.array([0.0])
sys.modules['drug_ae_reasoner.utils.encoding'] = _dummy_enc

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from drug_ae_reasoner.data import cadec_loader


def test_embedding_similarity_returns_drug_node():
    G = nx.MultiDiGraph()
    G.add_node('d1', type='drug', label='Aspirin')

    cadec_loader._DRUG_EMB_CACHE.clear()
    cadec_loader._CADEC_KG_CACHE.clear()

    with patch('drug_ae_reasoner.data.cadec_loader._load_cadec_graph', return_value=G), \
         patch('drug_ae_reasoner.data.cadec_loader.encode_text') as mock_encode, \
         patch('drug_ae_reasoner.data.cadec_loader._rxnorm_cuis_for', return_value=set()):
        mock_encode.side_effect = lambda text: np.array([1.0]) if 'aspirin' in text.lower() else np.array([0.0])
        res = cadec_loader.get_cadec_drug_nodes('aspirin generic', kg_path='dummy')

    assert res == [('d1', 'aspirin', set())]
    assert mock_encode.call_count == 2


def test_brand_name_matches_via_rxnorm_synonyms(tmp_path):
    rx = tmp_path / "rx"
    rx.mkdir()
    conso = rx / "RXNCONSO.RRF"
    rel = rx / "RXNREL.RRF"
    generic = "|".join(["CUI-G", "ENG", "", "", "", "", "", "", "", "", "", "", "", "", "Ibuprofen", "", "N", ""])
    brand = "|".join(["CUI-B", "ENG", "", "", "", "", "", "", "", "", "", "", "", "", "Advil", "", "N", ""])
    conso.write_text("\n".join([generic, brand]))
    rel_line = "|".join(["CUI-B", "", "", "", "CUI-G", "", "", "tradename_of", ""])
    rel.write_text(rel_line)

    G = nx.MultiDiGraph()
    G.add_node("d1", type="drug", label="Ibuprofen", cuis={"CUI-G"})

    cadec_loader._DRUG_EMB_CACHE.clear()
    cadec_loader._CADEC_KG_CACHE.clear()

    with patch("drug_ae_reasoner.data.cadec_loader._load_cadec_graph", return_value=G), \
         patch("drug_ae_reasoner.data.cadec_loader._rxnorm_cuis_for", return_value=set()):
        res = cadec_loader.get_cadec_drug_nodes("Advil", kg_path="dummy", rxn_rrf_path=str(rx))

    assert res == [("d1", "advil", {"CUI-G"})]


def test_embedding_requires_confirmation():
    G = nx.MultiDiGraph()
    G.add_node("d1", type="drug", label="Aspirin")

    cadec_loader._DRUG_EMB_CACHE.clear()
    cadec_loader._CADEC_KG_CACHE.clear()

    with patch("drug_ae_reasoner.data.cadec_loader._load_cadec_graph", return_value=G), \
         patch("drug_ae_reasoner.data.cadec_loader.encode_text", return_value=np.array([1.0])), \
         patch("drug_ae_reasoner.data.cadec_loader._rxnorm_cuis_for", return_value=set()):
        res = cadec_loader.get_cadec_drug_nodes(
            "unmatched", kg_path="dummy", rxn_rrf_path="missing", mel_top_k=5,
            mel_threshold=0.0, mel_require_confirmation=True
        )

    assert res == []


def test_disable_embedding_search():
    G = nx.MultiDiGraph()
    G.add_node("d1", type="drug", label="Aspirin")

    cadec_loader._DRUG_EMB_CACHE.clear()
    cadec_loader._CADEC_KG_CACHE.clear()

    with patch("drug_ae_reasoner.data.cadec_loader._load_cadec_graph", return_value=G), \
         patch("drug_ae_reasoner.data.cadec_loader.encode_text", return_value=np.array([1.0])), \
         patch("drug_ae_reasoner.data.cadec_loader._rxnorm_cuis_for", return_value=set()):
        res = cadec_loader.get_cadec_drug_nodes(
            "aspirin generic", kg_path="dummy", use_embedding=False
        )

    assert res == []
