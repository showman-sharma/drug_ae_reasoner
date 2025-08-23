import os
import sys
import types
from unittest.mock import patch

import networkx as nx
import pytest

# Stub out heavy modules to avoid network and large loads during import
_dummy_sim = types.ModuleType('drug_ae_reasoner.utils.similarity_search')
_dummy_sim.build_input_ae_oae_list = lambda *args, **kwargs: []
_dummy_sim.build_cadec_ae_oae_mapping = lambda *args, **kwargs: {}
sys.modules['drug_ae_reasoner.utils.similarity_search'] = _dummy_sim

_dummy_enc = types.ModuleType('drug_ae_reasoner.utils.encoding')
_dummy_enc.encode_text = lambda *args, **kwargs: [0.0]
sys.modules['drug_ae_reasoner.utils.encoding'] = _dummy_enc

# Ensure package root on path and import target module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from drug_ae_reasoner.utils import path_reasoner


def test_rx_path_forwarding():
    """find_top_drug_to_input_ae_paths should pass through the rx_path argument."""
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


def test_oae_paths_forwarding():
    """find_top_drug_to_input_ae_paths should pass through OAE resource paths."""
    with patch('drug_ae_reasoner.utils.path_reasoner.build_cadec_ae_oae_mapping') as mock_cadec, \
         patch('drug_ae_reasoner.utils.path_reasoner.build_input_ae_oae_list') as mock_input, \
         patch('drug_ae_reasoner.utils.path_reasoner.get_cadec_drug_nodes'), \
         patch('drug_ae_reasoner.utils.path_reasoner.get_cadec_ae_pairs', return_value=[]), \
         patch('drug_ae_reasoner.utils.path_reasoner.find_drug_to_input_ae_paths', return_value=[]), \
         patch('drug_ae_reasoner.utils.path_reasoner.rank_drug_ae_paths', return_value=[]), \
         patch('drug_ae_reasoner.utils.path_reasoner.generate_fallback_drug_paths', return_value=[]), \
         patch('drug_ae_reasoner.utils.path_reasoner.generate_fallback_ae_paths', return_value=[]), \
         patch('drug_ae_reasoner.utils.path_reasoner.verbalize_drug_to_input_ae_paths', return_value=[]):
        path_reasoner.find_top_drug_to_input_ae_paths(
            drug='aspirin',
            ae_input_list=['nausea'],
            rx_path='rx',
            cadec_kg_path='kg',
            oae_index_path='idx_path',
            oae_label_map_path='lbl_path',
            oae_graph_path='graph',
            n_paths=5,
            n_cadec=5,
            n_input=5,
            cadec_ae_threshold=0.7,
            input_ae_threshold=0.7,
            n_disconnect=3,
        )

    kwargs_cadec = mock_cadec.call_args.kwargs
    assert kwargs_cadec['index_path'] == 'idx_path'
    assert kwargs_cadec['label_map_path'] == 'lbl_path'

    kwargs_input = mock_input.call_args.kwargs
    assert kwargs_input['index_path'] == 'idx_path'
    assert kwargs_input['label_map_path'] == 'lbl_path'

def test_generate_fallback_drug_paths_case_insensitive():
    """Mixed-case drug labels should match lowercase entries in cadec_pairs."""
    cadec_pairs = [('metformin', 'nausea', 'ae1')]
    cadec_ae_oae_dict = {'nausea': [('OAE:0001', 0.9)]}

    res = path_reasoner.generate_fallback_drug_paths('Metformin', cadec_pairs, cadec_ae_oae_dict)
    assert res and res[0][1] == 'nausea'


def test_generate_fallback_drug_paths_unknown_drug():
    """Unknown drugs should yield an empty fallback list."""
    cadec_pairs = [('metformin', 'nausea', 'ae1')]
    cadec_ae_oae_dict = {'nausea': [('OAE:0001', 0.9)]}

    res = path_reasoner.generate_fallback_drug_paths('Ibuprofen', cadec_pairs, cadec_ae_oae_dict)
    assert res == []


def test_find_drug_to_input_ae_paths_zero_and_one_hop():
    """find_drug_to_input_ae_paths should capture 0-hop and 1-hop relations."""
    G = nx.MultiDiGraph()
    G.add_edge('OAE:2', 'OAE:1')
    with patch('drug_ae_reasoner.utils.path_reasoner._load_oae_graph', return_value=G):
        res = path_reasoner.find_drug_to_input_ae_paths(
            'drug',
            {'ae1': [('OAE:1', 0.9)], 'ae2': [('OAE:2', 0.8)]},
            [('input', 'OAE:1', 0.7)],
        )
    assert ('drug', 'input', ['OAE:1']) in res  # 0-hop
    assert ('drug', 'input', ['OAE:2', 'OAE:1']) in res  # 1-hop


def test_rank_drug_ae_paths_scoring_and_dedup():
    """rank_drug_ae_paths should score paths and keep the highest per OAE route."""
    raw_paths = [
        ('drug', 'inp_high', ['O1']),
        ('drug', 'inp_low', ['O1']),
    ]
    cadec_dict = {'ae': [('O1', 0.8)]}
    oae_inputs = [('inp_high', 'O1', 0.9), ('inp_low', 'O1', 0.5)]
    res = path_reasoner.rank_drug_ae_paths(raw_paths, cadec_dict, oae_inputs)
    assert res == [('drug', 'inp_high', ['O1'], pytest.approx((0.8 + 0.9) / 2))]


def test_generate_fallback_ae_paths_returns_expected():
    """generate_fallback_ae_paths should link inputs back to CADEC drugs."""
    ae_inputs = ['nausea']
    cadec_pairs = [('Aspirin', 'cadec_nausea', 'ae1')]
    cadec_dict = {'cadec_nausea': [('OAE:1', 0.8)]}
    oae_inputs = [('nausea', 'OAE:1', 0.6)]
    res = path_reasoner.generate_fallback_ae_paths(ae_inputs, cadec_pairs, cadec_dict, oae_inputs)
    assert res and res[0][0] == 'Aspirin'
    assert res[0][1] == 'nausea'
    assert res[0][2] == ['OAE:1']
    assert res[0][3] == pytest.approx((0.6 + 0.8 + 1.0) / 3)
