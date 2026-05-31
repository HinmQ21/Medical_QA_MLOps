"""Production retrieval backed by the existing baseline KG tool.

Heavy dependencies are imported lazily inside ``KGRetrieval.__init__`` so unit
tests can import this module without FAISS or sentence-transformers installed.
"""

import os
import sys

from .backends import RetrievalBackend
from .device import resolve_device


class KGRetrieval(RetrievalBackend):  # pragma: no cover
    def __init__(self, data_dir: str | None = None, device: str | None = None):
        self.data_dir = data_dir or os.environ.get("KG_DATA_DIR", "data/")
        self.device = resolve_device(device)
        baseline_root = os.environ.get("BASELINE_ROOT", "/home/vcsai/minhlbq/baseline")
        if baseline_root not in sys.path:
            sys.path.insert(0, baseline_root)
        from scripts.serve.retrieval_tool import MedicalKnowledgeTool

        self._tool = MedicalKnowledgeTool.load(self.data_dir, device=self.device)

    def search(self, query: str, top_k: int) -> list[str]:
        return self._tool.retrieve(query, top_k)
