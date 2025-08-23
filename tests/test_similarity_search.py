import sys
import types
import io
import pickle
import numpy as np
import builtins

import pytest

def setup_module(module):
    # Replace heavy encoding module with a lightweight stub before importing
    dummy_enc = types.ModuleType('drug_ae_reasoner.utils.encoding')
    class DummyModel:
        def encode(self, seq, convert_to_tensor=False, normalize_embeddings=True):
            return np.zeros((len(seq), 1), dtype=np.float32)
    dummy_enc.model = DummyModel()
    sys.modules['drug_ae_reasoner.utils.encoding'] = dummy_enc


def test_custom_paths(monkeypatch):
    # Ensure we import a fresh copy of similarity_search
    sys.modules.pop('drug_ae_reasoner.utils.similarity_search', None)
    import drug_ae_reasoner.utils.similarity_search as sim

    calls = []
    class DummyIndex:
        def search(self, queries, k):
            return np.zeros((len(queries), k)), np.zeros((len(queries), k), dtype=np.int64)
    def fake_read_index(path):
        calls.append(('index', path))
        return DummyIndex()
    def fake_open(path, mode='rb'):
        calls.append(('label', path))
        return io.BytesIO(pickle.dumps(['LBL']))

    monkeypatch.setattr(sim.faiss, 'read_index', fake_read_index)
    monkeypatch.setattr(builtins, 'open', fake_open)

    sim.build_cadec_ae_oae_mapping(['ae1'], index_path='idx1', label_map_path='lbl1')
    sim.build_input_ae_oae_list(['ae2'], index_path='idx2', label_map_path='lbl2')

    assert ('index', 'idx1') in calls
    assert ('label', 'lbl1') in calls
    assert ('index', 'idx2') in calls
    assert ('label', 'lbl2') in calls
