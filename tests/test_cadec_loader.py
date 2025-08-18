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
