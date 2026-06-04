from mlops.pipelines.build_demo_kg import HYPEREDGES, build_graph


def test_hyperedges_have_kgretrieval_fields():
    required = {"id", "description", "relation", "type", "anchor", "entities"}
    for hedge in HYPEREDGES:
        assert required <= set(hedge), hedge
        assert hedge["id"] and hedge["description"]
        assert isinstance(hedge["entities"], list)


def test_hedge_ids_are_unique():
    ids = [hedge["id"] for hedge in HYPEREDGES]
    assert len(ids) == len(set(ids))


def test_build_graph_derives_entities_and_entity_to_hedges():
    graph = build_graph()
    assert set(graph) == {"hyperedges", "entities", "entity_to_hedges"}
    # anchors and member entities are all registered as entities
    assert "Metformin" in graph["entities"]
    assert "type 2 diabetes mellitus" in graph["entities"]
    # entity_to_hedges maps an entity name to the hedges referencing it
    assert graph["entity_to_hedges"]["Metformin"] == ["h-metformin-t2dm"]
    assert "h-metformin-t2dm" in graph["entity_to_hedges"]["type 2 diabetes mellitus"]
    # every referenced hedge id exists
    all_ids = {hedge["id"] for hedge in graph["hyperedges"]}
    for hids in graph["entity_to_hedges"].values():
        assert set(hids) <= all_ids
