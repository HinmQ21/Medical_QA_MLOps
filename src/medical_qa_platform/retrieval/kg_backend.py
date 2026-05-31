"""Self-contained FAISS-backed medical knowledge retrieval."""

import json
import os
from pathlib import Path

from .backends import RetrievalBackend
from .device import resolve_device


class KGRetrieval(RetrievalBackend):  # pragma: no cover
    """Load local KG artifacts and retrieve hyperedge descriptions."""

    def __init__(
        self,
        data_dir: str | None = None,
        device: str | None = None,
        encoder_model: str | None = None,
    ):
        self.data_dir = Path(data_dir or os.environ.get("KG_DATA_DIR", "data/"))
        self.device = resolve_device(device)
        self.encoder_model = encoder_model or os.environ.get(
            "KG_ENCODER_MODEL", "abhinand/MedEmbed-large-v0.1"
        )

        faiss, np, sentence_transformer = self._load_runtime_dependencies()
        self._np = np
        self.encoder = sentence_transformer(self.encoder_model, device=self.device)
        self.idx_he = faiss.read_index(str(self.data_dir / "index_hyperedge.bin"))
        self.idx_ent = faiss.read_index(str(self.data_dir / "index_entity.bin"))
        self.hedge_ids = np.load(
            self.data_dir / "hedge_ids.npy", allow_pickle=True
        ).tolist()
        self.ent_names = np.load(
            self.data_dir / "entity_names.npy", allow_pickle=True
        ).tolist()

        with open(self.data_dir / "medical_hg.json") as handle:
            graph = json.load(handle)
        self.hedge_by_id = {
            item["id"]: item["description"] for item in graph["hyperedges"]
        }
        self.entity_to_hedges = graph.get("entity_to_hedges", {})

    @staticmethod
    def _load_runtime_dependencies():
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "KGRetrieval requires the optional runtime extra: "
                "pip install 'medical_qa_platform[runtime]'"
            ) from exc
        return faiss, np, SentenceTransformer

    @staticmethod
    def _first_row(values) -> list:
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not values:
            return []
        return list(values[0])

    def _encode_query(self, query: str):
        embedding = self.encoder.encode([query], normalize_embeddings=True)
        if hasattr(embedding, "astype"):
            return embedding.astype(self._np.float32)
        return embedding

    def _append_hyperedge(self, results: dict[str, str], hedge_id: str) -> None:
        description = self.hedge_by_id.get(hedge_id)
        if description is not None:
            results.setdefault(hedge_id, description)

    def search(self, query: str, top_k: int) -> list[str]:
        if top_k <= 0:
            return []

        embedding = self._encode_query(query)
        results: dict[str, str] = {}

        _, he_ids = self.idx_he.search(embedding, top_k)
        for raw_idx in self._first_row(he_ids):
            idx = int(raw_idx)
            if 0 <= idx < len(self.hedge_ids):
                self._append_hyperedge(results, self.hedge_ids[idx])
            if len(results) >= top_k:
                return list(results.values())[:top_k]

        _, ent_ids = self.idx_ent.search(embedding, 3)
        for raw_idx in self._first_row(ent_ids):
            idx = int(raw_idx)
            if not (0 <= idx < len(self.ent_names)):
                continue
            entity_name = self.ent_names[idx]
            for hedge_id in self.entity_to_hedges.get(entity_name, []):
                self._append_hyperedge(results, hedge_id)
                if len(results) >= top_k:
                    return list(results.values())[:top_k]

        return list(results.values())[:top_k]
