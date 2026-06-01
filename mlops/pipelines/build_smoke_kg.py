"""Generate tiny deterministic KG artifacts for the smoke pipeline."""

import argparse
import json
from pathlib import Path

from .profiles import PipelineProfile, load_profile


HYPEREDGES = [
    {
        "id": "h-metformin-diabetes",
        "description": "Metformin is commonly used first-line for type 2 diabetes.",
        "relation": "treats",
        "type": "drug",
        "anchor": "Metformin",
        "entities": ["type 2 diabetes"],
    },
    {
        "id": "h-insulin-diabetes",
        "description": "Type 1 diabetes is characterized by insulin deficiency.",
        "relation": "deficient_in",
        "type": "hormone",
        "anchor": "Insulin",
        "entities": ["type 1 diabetes"],
    },
]

ENTITIES = {
    "Metformin": {"type": "drug"},
    "Insulin": {"type": "hormone"},
    "type 1 diabetes": {"type": "disease"},
    "type 2 diabetes": {"type": "disease"},
}

ENTITY_TO_HEDGES = {
    "Metformin": ["h-metformin-diabetes"],
    "Insulin": ["h-insulin-diabetes"],
    "type 1 diabetes": ["h-insulin-diabetes"],
    "type 2 diabetes": ["h-metformin-diabetes"],
}

RETRIEVAL_FIXTURE = {
    "type 2 diabetes first line medication": [
        "Metformin is commonly used first-line for type 2 diabetes."
    ],
    "type 1 diabetes deficient hormone": [
        "Type 1 diabetes is characterized by insulin deficiency."
    ],
}


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_smoke_kg(profile: PipelineProfile) -> dict:
    graph = {
        "hyperedges": HYPEREDGES,
        "entities": ENTITIES,
        "entity_to_hedges": ENTITY_TO_HEDGES,
    }
    _write_json(profile.kg_dir / "medical_hg.json", graph)
    _write_json(profile.kg_dir / "hedge_ids.json", [item["id"] for item in HYPEREDGES])
    _write_json(profile.kg_dir / "entity_names.json", list(ENTITIES))
    _write_json(profile.retrieval_fixture_path, RETRIEVAL_FIXTURE)

    manifest = {
        "artifact_type": "smoke_kg",
        "profile": profile.name,
        "n_hyperedges": len(HYPEREDGES),
        "n_entities": len(ENTITIES),
        "retrieval_fixture_path": str(profile.retrieval_fixture_path),
    }
    _write_json(profile.kg_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke")
    args = parser.parse_args()
    build_smoke_kg(load_profile(args.profile))


if __name__ == "__main__":
    main()
