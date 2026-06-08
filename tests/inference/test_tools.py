from medical_qa_platform.inference.tools import MEDICAL_TOOL_DEF, SEARCH_TOOL_NAME


def test_tool_def_shape_matches_training_schema():
    assert SEARCH_TOOL_NAME == "search_medical_knowledge"
    assert MEDICAL_TOOL_DEF["type"] == "function"
    fn = MEDICAL_TOOL_DEF["function"]
    assert fn["name"] == SEARCH_TOOL_NAME
    assert fn["parameters"]["properties"]["query"]["type"] == "string"
    assert fn["parameters"]["required"] == ["query"]
