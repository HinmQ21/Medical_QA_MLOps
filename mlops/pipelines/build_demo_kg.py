"""Build a tiny but real FAISS-indexed KG compatible with KGRetrieval.

Unlike the smoke KG (JSON fixtures for FixtureRetrieval/eval), this produces the
exact artifacts the deployed `KGRetrieval` backend loads:
  - index_hyperedge.bin, index_entity.bin  (FAISS IndexFlatIP over MedEmbed-small)
  - hedge_ids.npy, entity_names.npy         (row-aligned id arrays, object dtype)
  - medical_hg.json                         (hyperedges + entities + entity_to_hedges)

A handful of common clinical relations gives a demo that exercises the real
retrieval path (encoder -> FAISS -> fusion ranker) without the full 56.8K-hedge
KG. Embedding needs the `runtime` extra (faiss, sentence-transformers, numpy);
graph construction is pure and unit-tested without those deps.
"""

import argparse
import json
from pathlib import Path

ENCODER_MODEL = "abhinand/MedEmbed-small-v0.1"

HYPEREDGES = [
    {"id": "h-metformin-t2dm", "relation": "first_line_for", "type": "drug", "anchor": "Metformin",
     "entities": ["type 2 diabetes mellitus"],
     "description": "Metformin is the first-line pharmacologic treatment for type 2 diabetes mellitus."},
    {"id": "h-insulin-t1dm", "relation": "deficient_in", "type": "hormone", "anchor": "Insulin",
     "entities": ["type 1 diabetes mellitus"],
     "description": "Type 1 diabetes mellitus results from autoimmune destruction of pancreatic beta cells causing insulin deficiency."},
    {"id": "h-lisinopril-htn", "relation": "treats", "type": "drug", "anchor": "Lisinopril",
     "entities": ["hypertension"],
     "description": "Lisinopril is an ACE inhibitor used to treat hypertension and reduce afterload."},
    {"id": "h-amoxicillin-otitis", "relation": "treats", "type": "antibiotic", "anchor": "Amoxicillin",
     "entities": ["acute otitis media"],
     "description": "Amoxicillin is the first-line antibiotic for acute otitis media in children."},
    {"id": "h-aspirin-mi", "relation": "prevents", "type": "drug", "anchor": "Aspirin",
     "entities": ["myocardial infarction"],
     "description": "Aspirin is an antiplatelet agent given to reduce mortality in acute myocardial infarction."},
    {"id": "h-warfarin-afib", "relation": "prevents_stroke_in", "type": "anticoagulant", "anchor": "Warfarin",
     "entities": ["atrial fibrillation"],
     "description": "Warfarin is an anticoagulant used to prevent thromboembolic stroke in atrial fibrillation."},
    {"id": "h-albuterol-asthma", "relation": "relieves", "type": "bronchodilator", "anchor": "Albuterol",
     "entities": ["asthma"],
     "description": "Albuterol is a short-acting beta-2 agonist that relieves acute asthma bronchospasm."},
    {"id": "h-levothyroxine-hypothyroid", "relation": "treats", "type": "hormone", "anchor": "Levothyroxine",
     "entities": ["hypothyroidism"],
     "description": "Levothyroxine is synthetic T4 used as replacement therapy for hypothyroidism."},
    {"id": "h-omeprazole-gerd", "relation": "treats", "type": "drug", "anchor": "Omeprazole",
     "entities": ["gastroesophageal reflux disease"],
     "description": "Omeprazole is a proton pump inhibitor that treats gastroesophageal reflux disease."},
    {"id": "h-atorvastatin-cholesterol", "relation": "lowers", "type": "statin", "anchor": "Atorvastatin",
     "entities": ["hypercholesterolemia"],
     "description": "Atorvastatin is a statin that lowers LDL cholesterol in hypercholesterolemia."},
    {"id": "h-furosemide-chf", "relation": "treats", "type": "diuretic", "anchor": "Furosemide",
     "entities": ["heart failure"],
     "description": "Furosemide is a loop diuretic used to relieve fluid overload in heart failure."},
    {"id": "h-ceftriaxone-meningitis", "relation": "treats", "type": "antibiotic", "anchor": "Ceftriaxone",
     "entities": ["bacterial meningitis"],
     "description": "Ceftriaxone is a third-generation cephalosporin used empirically for bacterial meningitis."},
    {"id": "h-prednisone-inflammation", "relation": "suppresses", "type": "corticosteroid", "anchor": "Prednisone",
     "entities": ["inflammation"],
     "description": "Prednisone is a corticosteroid that suppresses inflammation and immune response."},
    {"id": "h-acetaminophen-fever", "relation": "reduces", "type": "drug", "anchor": "Acetaminophen",
     "entities": ["fever"],
     "description": "Acetaminophen is an antipyretic and analgesic that reduces fever and mild pain."},
    {"id": "h-salbutamol-copd", "relation": "relieves", "type": "bronchodilator", "anchor": "Ipratropium",
     "entities": ["chronic obstructive pulmonary disease"],
     "description": "Ipratropium is an inhaled anticholinergic bronchodilator used in chronic obstructive pulmonary disease."},
]


def build_graph() -> dict:
    """Pure construction of the KGRetrieval graph (no embedding). Unit-tested."""
    entities: dict[str, dict] = {}
    entity_to_hedges: dict[str, list[str]] = {}
    for hedge in HYPEREDGES:
        names = [hedge["anchor"], *hedge["entities"]]
        for name in names:
            entities.setdefault(name, {"type": "entity"})
            bucket = entity_to_hedges.setdefault(name, [])
            if hedge["id"] not in bucket:
                bucket.append(hedge["id"])
    return {
        "hyperedges": HYPEREDGES,
        "entities": entities,
        "entity_to_hedges": entity_to_hedges,
    }


def build_demo_kg(out_dir: Path, encoder_model: str = ENCODER_MODEL) -> dict:
    """Write medical_hg.json + FAISS indices + row-aligned id arrays."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    out_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph()
    (out_dir / "medical_hg.json").write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n"
    )

    encoder = SentenceTransformer(encoder_model)

    hedge_ids = [hedge["id"] for hedge in graph["hyperedges"]]
    hedge_texts = [hedge["description"] for hedge in graph["hyperedges"]]
    he_emb = encoder.encode(hedge_texts, normalize_embeddings=True).astype("float32")
    idx_he = faiss.IndexFlatIP(he_emb.shape[1])
    idx_he.add(he_emb)
    faiss.write_index(idx_he, str(out_dir / "index_hyperedge.bin"))
    np.save(out_dir / "hedge_ids.npy", np.array(hedge_ids, dtype=object))

    ent_names = list(graph["entities"])
    ent_emb = encoder.encode(ent_names, normalize_embeddings=True).astype("float32")
    idx_ent = faiss.IndexFlatIP(ent_emb.shape[1])
    idx_ent.add(ent_emb)
    faiss.write_index(idx_ent, str(out_dir / "index_entity.bin"))
    np.save(out_dir / "entity_names.npy", np.array(ent_names, dtype=object))

    manifest = {
        "artifact_type": "demo_kg",
        "encoder_model": encoder_model,
        "embedding_dim": int(he_emb.shape[1]),
        "n_hyperedges": len(hedge_ids),
        "n_entities": len(ent_names),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts/demo/kg")
    parser.add_argument("--encoder-model", default=ENCODER_MODEL)
    args = parser.parse_args()
    manifest = build_demo_kg(Path(args.out_dir), args.encoder_model)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
