import json
import sys
import types
from pathlib import Path

from medical_qa_platform.retrieval.kg_backend import KGRetrieval


def test_runtime_source_does_not_reference_baseline():
    src_root = Path(__file__).parents[2] / "src" / "medical_qa_platform"
    forbidden = [
        "baseline",
        "BASELINE_ROOT",
        "scripts.serve",
        "MedicalKnowledgeTool",
        "/home/vcsai/minhlbq",
    ]
    for path in src_root.rglob("*.py"):
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f"{token!r} leaked into {path}"


def test_kg_retrieval_loads_local_artifacts_without_baseline(monkeypatch, tmp_path):
    data_dir = tmp_path / "kg"
    data_dir.mkdir()
    (data_dir / "medical_hg.json").write_text(
        json.dumps(
            {
                "hyperedges": [
                    {
                        "id": "h1",
                        "description": "metformin treats diabetes",
                        "relation": "treats",
                        "type": "drug",
                        "anchor": "metformin",
                        "entities": ["diabetes"],
                    },
                    {
                        "id": "h2",
                        "description": "insulin treats diabetes",
                        "relation": "treats",
                        "type": "drug",
                        "anchor": "insulin",
                        "entities": ["diabetes"],
                    },
                ],
                "entities": {
                    "diabetes": {"type": "disease"},
                    "metformin": {"type": "drug"},
                    "insulin": {"type": "drug"},
                },
                "entity_to_hedges": {"diabetes": ["h2"], "metformin": ["h1"]},
            }
        )
    )

    class FakeArray:
        def __init__(self, value):
            self._value = value

        def tolist(self):
            return self._value

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.float32 = "float32"

    def fake_load(path, allow_pickle=False):
        filename = Path(path).name
        if filename == "hedge_ids.npy":
            return FakeArray(["h1"])
        if filename == "entity_names.npy":
            return FakeArray(["diabetes"])
        raise AssertionError(f"unexpected np.load path: {path}")

    fake_numpy.load = fake_load

    class FakeEmbedding:
        def astype(self, dtype):
            assert dtype == "float32"
            return self

    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True):
            assert texts == ["diabetes"]
            assert normalize_embeddings is True
            return FakeEmbedding()

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = lambda model_name, device: FakeEncoder()

    class FakeIndex:
        def __init__(self, ids):
            self._ids = ids

        def search(self, embedding, top_k):
            return [[1.0]], [self._ids]

    fake_faiss = types.ModuleType("faiss")

    def fake_read_index(path):
        filename = Path(path).name
        if filename == "index_hyperedge.bin":
            return FakeIndex([0])
        if filename == "index_entity.bin":
            return FakeIndex([0])
        raise AssertionError(f"unexpected faiss index path: {path}")

    fake_faiss.read_index = fake_read_index

    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    backend = KGRetrieval(data_dir=str(data_dir), device="cpu")

    assert backend.search("diabetes", top_k=2) == [
        "metformin treats diabetes",
        "insulin treats diabetes",
    ]
