"""Generate the retrieve_v1 golden parity fixture from the reference ranker.

Run once (and after any intentional algorithm/encoder change) in a reference
venv that has faiss + sentence-transformers, e.g.:

    cd /home/vcsai/minhlbq/baseline
    ./training_venv312/bin/python \
        ../mlops-platform/scripts/gen_retrieval_golden.py \
        --baseline-root /home/vcsai/minhlbq/baseline \
        --data-dir /home/vcsai/minhlbq/baseline/data \
        --device cpu \
        --out ../mlops-platform/tests/retrieval/golden/retrieve_v1_golden.json

The fixture captures, per query: the resolved FAISS hits (he_hits/ent_hits with
original ranks), the reference's expected ranked descriptions, and a minimal KG
slice covering every referenced hyperedge/entity so the pure ranker can be
replayed in CI without faiss/encoder.
"""

import argparse
import json
import os
import subprocess
import sys

QUERIES = [
    "What is the mechanism of action of metformin in type 2 diabetes?",
    "first line treatment for essential hypertension",
    "classic symptoms of acute myocardial infarction",
    "cerebrospinal fluid findings in bacterial meningitis",
    "warfarin drug interactions and mechanism",
    "pathophysiology of asthma airway inflammation",
    "emergency management of severe hyperkalemia",
    "complications of diabetic ketoacidosis",
]

TOP_K = 5


def _git_commit(root):
    try:
        return (
            subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embed-model", default="abhinand/MedEmbed-large-v0.1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    os.environ["KG_ENCODER_MODEL"] = args.embed_model
    sys.path.insert(0, args.baseline_root)
    from scripts.serve.retrieval_tool import MedicalKnowledgeTool

    tool = MedicalKnowledgeTool.load(data_dir=args.data_dir, device=args.device)
    he_k = max(40, args.top_k * 8)
    ent_k = max(12, args.top_k * 4)

    cases = []
    ref_hids = set()
    ref_entities = set()

    for query in QUERIES:
        q_emb = tool.encoder.encode([query], normalize_embeddings=True).astype("float32")
        he_scores, he_ids = tool.idx_he.search(q_emb, he_k)
        ent_scores, ent_ids = tool.idx_ent.search(q_emb, ent_k)

        he_hits = []
        for rank, (score, raw_idx) in enumerate(zip(he_scores[0], he_ids[0])):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(tool.hedge_ids):
                continue
            hid = tool.hedge_ids[idx]
            if hid not in tool.hedge_meta:
                continue
            he_hits.append([rank, hid, float(score)])
            ref_hids.add(hid)

        ent_hits = []
        for ent_rank, (score, raw_idx) in enumerate(zip(ent_scores[0], ent_ids[0])):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(tool.ent_names):
                continue
            name = tool.ent_names[idx]
            ent_hits.append([ent_rank, name, float(score)])
            ref_entities.add(name)
            for hid in tool.entity_to_hedges.get(name, []):
                if hid in tool.hedge_meta:
                    ref_hids.add(hid)

        expected = tool.retrieve_v1(query, args.top_k)
        cases.append(
            {
                "query": query,
                "baseline_query_tokens": sorted(set(tool._tokenize(query))),
                "he_hits": he_hits,
                "ent_hits": ent_hits,
                "expected_ranked_descriptions": expected,
            }
        )

    # entity_token_sets must also cover anchors/entities referenced by sliced hedges
    token_entities = set(ref_entities)
    for hid in ref_hids:
        meta = tool.hedge_meta[hid]
        token_entities.add(meta.get("anchor", ""))
        token_entities.update(meta.get("entities", []))

    kg_slice = {
        "hedge_meta": {hid: tool.hedge_meta[hid] for hid in sorted(ref_hids)},
        "entity_to_hedges": {
            name: tool.entity_to_hedges.get(name, []) for name in sorted(ref_entities)
        },
        "hedge_token_sets": {
            hid: sorted(tool.hedge_token_sets[hid]) for hid in sorted(ref_hids)
        },
        "entity_token_sets": {
            name: sorted(tool.entity_token_sets[name])
            for name in sorted(token_entities)
            if name in tool.entity_token_sets
        },
    }

    fixture = {
        "manifest": {
            "encoder_model": args.embed_model,
            "device": args.device,
            "data_dir": args.data_dir,
            "top_k": args.top_k,
            "baseline_commit": _git_commit(args.baseline_root),
        },
        "cases": cases,
        "kg_slice": kg_slice,
    }

    with open(args.out, "w") as handle:
        json.dump(fixture, handle, indent=2, sort_keys=True)
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
