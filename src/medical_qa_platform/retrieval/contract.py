"""The retrieval contract: tool-response formatting + contract version.

format_evidence reproduces the exact string the reference search_medical_knowledge
tool returns, so the tokens the model sees at serving time match training.
"""

RETRIEVAL_CONTRACT_VERSION = "v1-medembed-small"

NO_RESULTS = "No relevant knowledge found."


def format_evidence(results):
    if not results:
        return NO_RESULTS
    return "\n".join(f"- {item}" for item in results)
