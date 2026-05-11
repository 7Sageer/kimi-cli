from __future__ import annotations

from kosong.message import Message, ToolCall

from kimi_cli.soul.history_normalization import (
    needs_thinking_switch_compaction,
    normalize_history_for_llm,
)
from kimi_cli.wire.types import TextPart, ThinkPart


def _tool_call(call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        function=ToolCall.FunctionBody(name="lookup", arguments='{"q":"kimi"}'),
    )


def test_thinking_off_strips_think_parts_without_losing_visible_content():
    history = [
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content=[ThinkPart(think="private reasoning"), TextPart(text="visible answer")],
        ),
    ]

    normalized = normalize_history_for_llm(history, thinking_enabled=False)

    assert len(normalized) == 2
    assert normalized[1].role == "assistant"
    assert normalized[1].content == [TextPart(text="visible answer")]


def test_thinking_off_keeps_tool_call_message_after_stripping_reasoning():
    tool_call = _tool_call()
    history = [
        Message(role="user", content="use a tool"),
        Message(
            role="assistant",
            content=[ThinkPart(think="I should call lookup")],
            tool_calls=[tool_call],
        ),
        Message(role="tool", content="result", tool_call_id=tool_call.id),
    ]

    normalized = normalize_history_for_llm(history, thinking_enabled=False)

    assert len(normalized) == 3
    assert normalized[1].role == "assistant"
    assert normalized[1].content == []
    assert normalized[1].tool_calls == [tool_call]
    assert normalized[2].tool_call_id == tool_call.id


def test_thinking_off_drops_orphan_thinking_only_assistant_message():
    history = [
        Message(role="user", content="hello"),
        Message(role="assistant", content=[ThinkPart(think="partial hidden thought")]),
        Message(role="user", content="next"),
    ]

    normalized = normalize_history_for_llm(history, thinking_enabled=False)

    assert [message.role for message in normalized] == ["user"]
    assert normalized[0].content == [TextPart(text="hello"), TextPart(text="next")]


def test_thinking_on_requires_compaction_for_legacy_tool_call_without_think_part():
    history = [
        Message(role="user", content="use a tool"),
        Message(role="assistant", content=[], tool_calls=[_tool_call()]),
        Message(role="tool", content="result", tool_call_id="call_1"),
        Message(role="user", content="continue"),
    ]

    assert needs_thinking_switch_compaction(history, thinking_enabled=True)


def test_thinking_on_does_not_compact_tool_call_with_think_part():
    history = [
        Message(role="user", content="use a tool"),
        Message(
            role="assistant",
            content=[ThinkPart(think="I should call lookup")],
            tool_calls=[_tool_call()],
        ),
        Message(role="tool", content="result", tool_call_id="call_1"),
        Message(role="user", content="continue"),
    ]

    assert not needs_thinking_switch_compaction(history, thinking_enabled=True)


def test_thinking_off_never_needs_thinking_switch_compaction():
    history = [
        Message(role="assistant", content=[], tool_calls=[_tool_call()]),
        Message(role="tool", content="result", tool_call_id="call_1"),
    ]

    assert not needs_thinking_switch_compaction(history, thinking_enabled=False)
