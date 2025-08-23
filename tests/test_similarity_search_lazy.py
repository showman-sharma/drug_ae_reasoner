import os
import sys
import importlib
import types
import numpy as np


def test_lazy_loading_and_cache(monkeypatch):
    # Ensure package root on path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    # Stub heavy encoding module before importing similarity_search
    dummy_enc = types.ModuleType('drug_ae_reasoner.utils.encoding')
    dummy_enc.model = types.SimpleNamespace(encode=lambda texts, **kw: np.zeros((len(texts), 1)))
    sys.modules['drug_ae_reasoner.utils.encoding'] = dummy_enc

    import drug_ae_reasoner.utils.similarity_search as ss
    importlib.reload(ss)
    ss.clear_similarity_caches()
    assert ss._INDEX_CACHE == {}
    assert ss._LABEL_CACHE == {}

    calls = {'index': 0, 'labels': 0}

    class DummyIndex:
        def search(self, queries, k):
            return np.zeros((len(queries), k)), -np.ones((len(queries), k), dtype=np.int64)

    def fake_read_index(path):
        calls['index'] += 1
        return DummyIndex()

    def fake_open(path, mode):
        calls['labels'] += 1
        class Dummy:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return b''
        return Dummy()

    monkeypatch.setattr(ss.faiss, 'read_index', fake_read_index)
    monkeypatch.setattr('builtins.open', fake_open)
    monkeypatch.setattr(ss.pickle, 'load', lambda f: ['L'])

    # First call loads index and labels
    ss.build_cadec_ae_oae_mapping(['a'], index_path='idx', label_map_path='lbl')
    assert calls == {'index': 1, 'labels': 1}

    # Second call with same paths uses cache
    ss.build_input_ae_oae_list(['b'], index_path='idx', label_map_path='lbl')
    assert calls == {'index': 1, 'labels': 1}

    # Different paths trigger another load
    ss.build_cadec_ae_oae_mapping(['c'], index_path='idx2', label_map_path='lbl2')
    assert calls == {'index': 2, 'labels': 2}
