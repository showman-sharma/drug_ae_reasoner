import os
import sys
import types
from unittest.mock import patch


def test_rx_path_forwarding():
    # Stub out similarity_search module to avoid loading large FAISS index
    dummy_sim = types.ModuleType('drug_ae_reasoner.utils.similarity_search')
    dummy_sim.build_input_ae_oae_list = lambda *args, **kwargs: []
    dummy_sim.build_cadec_ae_oae_mapping = lambda *args, **kwargs: {}
    sys.modules['drug_ae_reasoner.utils.similarity_search'] = dummy_sim

    # Ensure package root is on the path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from drug_ae_reasoner.utils import path_reasoner

    with patch('drug_ae_reasoner.utils.path_reasoner.get_cadec_drug_nodes') as mock_drug_nodes, \
         patch('drug_ae_reasoner.utils.path_reasoner.get_cadec_ae_pairs', return_value=[]), \
         patch('drug_ae_reasoner.utils.path_reasoner.find_drug_to_input_ae_paths', return_value=[]), \
         patch('drug_ae_reasoner.utils.path_reasoner.rank_drug_ae_paths', return_value=[]), \
         patch('drug_ae_reasoner.utils.path_reasoner.generate_fallback_drug_paths', return_value=[]), \
         patch('drug_ae_reasoner.utils.path_reasoner.generate_fallback_ae_paths', return_value=[]), \
         patch('drug_ae_reasoner.utils.path_reasoner.verbalize_drug_to_input_ae_paths', return_value=[]):
        path_reasoner.find_top_drug_to_input_ae_paths(
            drug='aspirin',
            ae_input_list=[],
            rx_path='custom_rx',
            cadec_kg_path='kg_path',
            oae_index_path='idx',
            oae_label_map_path='lbl',
            oae_graph_path='graph',
            n_paths=5,
            n_cadec=5,
            n_input=5,
            cadec_ae_threshold=0.7,
            input_ae_threshold=0.7,
            n_disconnect=3,
        )
    mock_drug_nodes.assert_called_once_with('aspirin', 'kg_path', 'custom_rx')
