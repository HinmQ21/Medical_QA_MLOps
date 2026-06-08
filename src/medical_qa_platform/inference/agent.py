"""Model-driven agentic tool-call loop for serving.

The model decides when to call ``search_medical_knowledge``; this loop executes
the call against the retrieval service, feeds the result back, and repeats until
the model answers or the tool-round budget is exhausted. Mirrors the training/eval
rollout from the RL pipeline.
"""

import json
from dataclasses import dataclass, field

# build_prompt is reused intentionally so the loop's initial [system, user] messages
# stay in sync with the rest of the API. This inference->api import is acyclic
# (api.prompt imports nothing from inference).
from ..api.prompt import build_prompt
from ..retrieval.backends import SupportsSearch
from .base import ModelBackend
from .tools import MEDICAL_TOOL_DEF


@dataclass
class LoopResult:
    trace: list[dict] = field(default_factory=list)
    final_content: str = ""
    evidence: list[str] = field(default_factory=list)
    tool_call_count: int = 0


def _query_of(tool_call: dict) -> str:
    """Extract the ``query`` argument from a raw OpenAI tool call; "" on bad JSON."""
    args = tool_call.get("function", {}).get("arguments", "")
    if isinstance(args, dict):
        return str(args.get("query", ""))
    try:
        return str(json.loads(args).get("query", ""))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return ""


def run_agentic_loop(
    backend: ModelBackend,
    retrieval: SupportsSearch,
    question: str,
    *,
    top_k: int,
    max_tokens: int,
    max_iterations: int,
) -> LoopResult:
    messages = build_prompt(question, [])
    result = LoopResult()

    turn = None
    # indices 0..max_iterations-1 are tool rounds; index max_iterations is the
    # forced final round (no tools) that must produce the answer.
    for i in range(max_iterations + 1):
        if i == max_iterations:
            # Final round: offer no tools so the model must answer in text.
            turn = backend.chat(messages, tools=None, max_tokens=max_tokens)
        else:
            turn = backend.chat(
                messages,
                tools=[MEDICAL_TOOL_DEF],
                tool_choice="auto",
                max_tokens=max_tokens,
            )

        # Resend with OpenAI-correct content: null (None) on a tool-call-only turn,
        # the text otherwise. The trace entry below always uses a string for display.
        assistant_msg = {"role": "assistant", "content": turn.content}
        if turn.tool_calls:
            assistant_msg["tool_calls"] = turn.tool_calls
        messages.append(assistant_msg)
        result.trace.append(
            {
                "role": "assistant",
                "content": turn.content or "",
                "tool_calls": turn.tool_calls or None,
            }
        )

        if turn.tool_calls and i < max_iterations:
            for tc in turn.tool_calls:
                result.tool_call_count += 1
                query = _query_of(tc)
                hits = retrieval.search(query, top_k) if query else []
                result.evidence.extend(hits)
                content = "\n".join(f"- {h}" for h in hits) or "No relevant knowledge found."
                messages.append(
                    {"role": "tool", "tool_call_id": tc.get("id", ""), "content": content}
                )
                result.trace.append({"role": "tool", "content": content, "tool_calls": None})
            continue

        break

    result.final_content = (turn.content if turn else "") or ""
    return result
