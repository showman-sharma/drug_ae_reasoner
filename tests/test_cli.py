import sys
import types
from unittest.mock import MagicMock

import pytest


def test_cli_forwards_oae_paths(monkeypatch):
    dummy_pr = types.ModuleType('drug_ae_reasoner.utils.path_reasoner')
    dummy_pr.find_top_drug_to_input_ae_paths = MagicMock(return_value=(False, [], [], [], []))
    monkeypatch.setitem(sys.modules, 'drug_ae_reasoner.utils.path_reasoner', dummy_pr)

    import importlib
    sys.modules.pop('drug_ae_reasoner.main', None)
    from drug_ae_reasoner import main

    argv = ['drug_ae_reasoner', '--drug', 'aspirin', '--aes', 'nausea', '--oae_index_path', 'idx', '--oae_label_map_path', 'lbl']
    monkeypatch.setattr(sys, 'argv', argv)
    main.main()

    dummy_pr.find_top_drug_to_input_ae_paths.assert_called_once()
    kwargs = dummy_pr.find_top_drug_to_input_ae_paths.call_args.kwargs
    assert kwargs['oae_index_path'] == 'idx'
    assert kwargs['oae_label_map_path'] == 'lbl'
