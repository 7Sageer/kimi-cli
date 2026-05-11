from __future__ import annotations

from collections.abc import Sequence

from kosong.message import Message

from kimi_cli.llm import LLM
from kimi_cli.notifications import is_notification_message
from kimi_cli.wire.types import TextPart, ThinkPart


def llm_uses_thinking(llm: LLM) -> bool:
    """Return whether the concrete chat provider will send thinking parameters."""
    effort = getattr(llm.chat_provider, "thinking_effort", None)
    return effort is not None and effort != "off"


def needs_thinking_switch_compaction(history: Sequence[Message], *, thinking_enabled: bool) -> bool:
    """Whether thinking-on replay needs a compaction boundary.

    Kimi/OpenAI-compatible endpoints reject replaying assistant tool-call
    messages without reasoning content when the next request has thinking
    enabled. Those missing blocks cannot be reconstructed, so the safe move is
    to compact the old raw trajectory before sending the next request.
    """
    if not thinking_enabled:
        return False

    for message in history:
        if message.role != "assistant" or not message.tool_calls:
            continue
        if not any(isinstance(part, ThinkPart) for part in message.content):
            return True
    return False


def normalize_history_for_llm(
    history: Sequence[Message], *, thinking_enabled: bool
) -> list[Message]:
    """Prepare persisted conversation history for the current LLM request.

    This is intentionally lossy only for protocol-only data:
    - when thinking is disabled or unsupported, strip ThinkPart blocks so the
      provider does not send reasoning content with a non-thinking request;
    - drop orphan assistant messages that only contain thinking and no tool
      calls, because they carry no replayable visible content;
    - merge adjacent user messages after filtering, preserving notification
      boundaries.
    """
    filtered: list[Message] = []
    for message in history:
        normalized = _normalize_message_for_thinking(message, thinking_enabled=thinking_enabled)
        if normalized is None:
            continue
        filtered.append(normalized)
    return _merge_adjacent_user_messages(filtered)


def _normalize_message_for_thinking(message: Message, *, thinking_enabled: bool) -> Message | None:
    content = message.content
    if not thinking_enabled:
        content = [part for part in content if not isinstance(part, ThinkPart)]

    if message.role == "assistant" and not message.tool_calls and not _has_visible_content(content):
        return None

    if content is message.content:
        return message
    return message.model_copy(update={"content": content}, deep=True)


def _has_visible_content(content: Sequence[object]) -> bool:
    for part in content:
        if isinstance(part, ThinkPart):
            continue
        if isinstance(part, TextPart):
            if part.text.strip():
                return True
            continue
        return True
    return False


def _merge_adjacent_user_messages(history: Sequence[Message]) -> list[Message]:
    result: list[Message] = []
    for message in history:
        if (
            result
            and result[-1].role == "user"
            and message.role == "user"
            and not is_notification_message(result[-1])
            and not is_notification_message(message)
        ):
            merged_content = list(result[-1].content) + list(message.content)
            result[-1] = Message(role="user", content=merged_content)
        else:
            result.append(message)
    return result
