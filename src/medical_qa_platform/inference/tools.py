"""The tool schema advertised to the model for native OpenAI function-calling.

Mirrors the training/eval tool definition used in the RL pipeline so serving
behaves like the agentic rollout the model was trained against. Kept in its own
module so a Llama-family variant can be added later without touching the loop.
"""

SEARCH_TOOL_NAME = "search_medical_knowledge"

MEDICAL_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Search the medical knowledge base for relevant clinical information "
            "about diseases, drugs, symptoms, and treatments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Medical query to search for.",
                }
            },
            "required": ["query"],
        },
    },
}
