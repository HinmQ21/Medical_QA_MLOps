"""Self-contained FAISS-backed medical knowledge retrieval (retrieve_v1 parity).

Heavy I/O only: load artifacts, encode the query, run FAISS, resolve indices to
(rank, hid/name, sim) hits, then delegate ranking to the pure `ranker` module.
"""

import json
import os
from pathlib import Path

from . import ranker
from .backends import RetrievalBackend
from .device import resolve_device


class KGRetrieval(RetrievalBackend):  # pragma: no cover
    """Load local KG artifacts and retrieve hyperedge descriptions via retrieve_v1."""

    def __init__(self, data_dir=None, device=None, encoder_model=None):
        self.data_dir = Path(data_dir or os.environ.get("KG_DATA_DIR", "data/"))
        self.device = resolve_device(device)
        self.encoder_model = encoder_model or os.environ.get(
            "KG_ENCODER_MODEL", "abhinand/MedEmbed-small-v0.1"
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
        self.hedge_meta = {
            item["id"]: {
                "description": item["description"],
                "relation": item.get("relation", ""),
                "type": item.get("type", ""),
                "anchor": item.get("anchor", ""),
                "entities": item.get("entities", []),
            }
            for item in graph["hyperedges"]
        }
        self.hedge_by_id = {
            hid: meta["description"] for hid, meta in self.hedge_meta.items()
        }
        self.entity_to_hedges = graph.get("entity_to_hedges", {})
        self.hedge_token_sets = {
            hid: set(
                ranker.tokenize(
                    " ".join(
                        [
                            meta["description"],
                            meta["relation"],
                            meta["type"],
                            meta["anchor"],
                            *meta["entities"],
                        ]
                    )
                )
            )
            for hid, meta in self.hedge_meta.items()
        }
        self.entity_token_sets = {
            name: set(ranker.tokenize(name)) for name in graph.get("entities", {})
        }

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
    def _first_row(values):
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not values:
            return []
        return list(values[0])

    def _encode_query(self, query):
        embedding = self.encoder.encode([query], normalize_embeddings=True)
        if hasattr(embedding, "astype"):
            return embedding.astype(self._np.float32)
        return embedding

    def search(self, query, top_k=ranker.DEFAULT_TOP_K):
        if top_k <= 0:
            return []

        embedding = self._encode_query(query)
        he_k, ent_k = ranker.candidate_budgets(top_k)
        he_scores, he_ids = self.idx_he.search(embedding, he_k)
        ent_scores, ent_ids = self.idx_ent.search(embedding, ent_k)

        he_scores_row = self._first_row(he_scores)
        he_ids_row = self._first_row(he_ids)
        ent_scores_row = self._first_row(ent_scores)
        ent_ids_row = self._first_row(ent_ids)

        he_hits = []
        for rank, (score, raw_idx) in enumerate(zip(he_scores_row, he_ids_row)):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(self.hedge_ids):
                continue
            hid = self.hedge_ids[idx]
            if hid not in self.hedge_meta:
                continue
            he_hits.append((rank, hid, float(score)))

        ent_hits = []
        for ent_rank, (score, raw_idx) in enumerate(zip(ent_scores_row, ent_ids_row)):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(self.ent_names):
                continue
            ent_hits.append((ent_rank, self.ent_names[idx], float(score)))

        ranked_hids = ranker.fuse_candidates(
            he_hits=he_hits,
            ent_hits=ent_hits,
            hedge_meta=self.hedge_meta,
            hedge_token_sets=self.hedge_token_sets,
            entity_token_sets=self.entity_token_sets,
            entity_to_hedges=self.entity_to_hedges,
            query_tokens=set(ranker.tokenize(query)),
            top_k=top_k,
        )
        return [self.hedge_by_id[hid] for hid in ranked_hids]
