"""Tests for the session-pinned thinking mode in ``KimiCLI.create``.

Switching thinking mode mid-session corrupts the conversation history that gets
replayed to the API (assistant tool-call messages generated without thinking
blocks cannot be sent back with thinking enabled, and vice versa). The fix is to
pin the thinking mode to the session on first invocation and enforce it on
resume.

Legacy sessions (created before the pin existed) carry no recorded mode but may
have replayable history. To avoid reproducing the original API 400, those are
pinned to ``False`` on first resume — the safe default that matches existing
tool-call messages without reasoning blocks.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import kimi_cli.app as app_module
from kimi_cli.app import KimiCLI
from kimi_cli.session import Session
from kimi_cli.session_state import SessionState


def _simulate_resumed_with_content(session) -> None:
    """Write a placeholder line so ``session.wire_file.is_empty()`` returns False.

    A non-empty wire file is what distinguishes "legacy resumed session with
    replayable history" (must pin thinking=False) from "resumed session that
    never actually wrote a turn" (treated as fresh).
    """
    path: Path = session.wire_file.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"placeholder": "non-metadata content"}\n', encoding="utf-8")


def _patch_create_deps(monkeypatch):
    """Patch heavy dependencies so KimiCLI.create() runs without I/O.

    Returns ``create_llm_kwargs`` (list of dicts captured per call) so tests can
    assert what thinking value was actually passed down to the LLM provider.
    """

    create_llm_kwargs: list[dict] = []

    class FakeSoul:
        def __init__(self, agent, context):
            self.plan_mode = False
            self._set_plan_mode_calls: list[tuple[bool, str]] = []

        async def set_plan_mode_from_manual(self, enabled: bool) -> bool:
            self._set_plan_mode_calls.append((enabled, "manual"))
            self.plan_mode = enabled
            return enabled

        def schedule_plan_activation_reminder(self) -> None: ...
        def set_hook_engine(self, engine) -> None: ...

    fake_context = SimpleNamespace(system_prompt=None)
    fake_context.restore = AsyncMock()
    fake_context.write_system_prompt = AsyncMock()

    async def fake_runtime_create(config, _oauth, _llm, session, yolo, **kwargs):
        return SimpleNamespace(
            session=session,
            config=config,
            llm=None,
            approval=SimpleNamespace(
                is_yolo=lambda: yolo,
                is_afk=lambda: kwargs.get("afk", False) or kwargs.get("runtime_afk", False),
            ),
            notifications=SimpleNamespace(recover=lambda: None),
            background_tasks=SimpleNamespace(reconcile=lambda: None),
        )

    def fake_create_llm(*args, **kwargs):
        create_llm_kwargs.append(kwargs)
        return None

    monkeypatch.setattr(app_module, "load_config", lambda conf: conf)
    monkeypatch.setattr(app_module, "augment_provider_with_env_vars", lambda p, m: {})
    monkeypatch.setattr(app_module, "create_llm", fake_create_llm)
    monkeypatch.setattr(app_module.Runtime, "create", fake_runtime_create)
    monkeypatch.setattr(
        app_module,
        "load_agent",
        AsyncMock(return_value=SimpleNamespace(name="test", system_prompt="sp")),
    )
    monkeypatch.setattr(app_module, "Context", lambda _path: fake_context)
    monkeypatch.setattr(app_module, "KimiSoul", FakeSoul)

    return create_llm_kwargs


def _suppress_save_state(monkeypatch):
    """Patch Session.save_state to a no-op recorder so tests don't touch disk."""
    saves: list[bool | None] = []

    def _save(self) -> None:
        saves.append(self.state.thinking)

    monkeypatch.setattr(Session, "save_state", _save)
    return saves


class TestThinkingPinNewSession:
    """First-time pinning when ``state.thinking`` is None."""

    @pytest.mark.asyncio
    async def test_pins_requested_value_when_explicit(self, session, config, monkeypatch):
        create_llm_kwargs = _patch_create_deps(monkeypatch)
        saves = _suppress_save_state(monkeypatch)
        assert session.state.thinking is None

        await KimiCLI.create(session, config=config, thinking=True, resumed=False)

        assert session.state.thinking is True
        assert saves == [True]  # save_state was called exactly once
        assert create_llm_kwargs[0]["thinking"] is True

    @pytest.mark.asyncio
    async def test_pins_config_default_when_not_explicit(self, session, config, monkeypatch):
        config.default_thinking = True
        create_llm_kwargs = _patch_create_deps(monkeypatch)
        saves = _suppress_save_state(monkeypatch)

        await KimiCLI.create(session, config=config, thinking=None, resumed=False)

        assert session.state.thinking is True
        assert saves == [True]
        assert create_llm_kwargs[0]["thinking"] is True

    @pytest.mark.asyncio
    async def test_pins_false_for_default_off(self, session, config, monkeypatch):
        # default_thinking defaults to False in get_default_config
        assert config.default_thinking is False
        create_llm_kwargs = _patch_create_deps(monkeypatch)
        saves = _suppress_save_state(monkeypatch)

        await KimiCLI.create(session, config=config, resumed=False)

        assert session.state.thinking is False
        assert saves == [False]
        assert create_llm_kwargs[0]["thinking"] is False


class TestThinkingPinLegacySession:
    """Resumed sessions whose state was written before this field existed."""

    @pytest.mark.asyncio
    async def test_resumed_empty_session_pins_requested(self, session, config, monkeypatch):
        """A 'resumed' session whose wire file has no content (e.g. after /new
        but before the first turn) is treated as fresh and pins to requested."""
        create_llm_kwargs = _patch_create_deps(monkeypatch)
        saves = _suppress_save_state(monkeypatch)
        assert session.state.thinking is None
        assert session.wire_file.is_empty()

        await KimiCLI.create(session, config=config, thinking=True, resumed=True)

        assert session.state.thinking is True
        assert saves == [True]
        assert create_llm_kwargs[0]["thinking"] is True

    @pytest.mark.asyncio
    async def test_resumed_with_content_pins_to_false(self, session, config, monkeypatch):
        """A legacy session that already has replayable history must pin to
        False regardless of the user's requested value — the existing tool-call
        messages have no reasoning blocks, so enabling thinking would reproduce
        the API 400 the pin is meant to prevent."""
        create_llm_kwargs = _patch_create_deps(monkeypatch)
        saves = _suppress_save_state(monkeypatch)
        _simulate_resumed_with_content(session)
        assert session.state.thinking is None

        await KimiCLI.create(session, config=config, thinking=True, resumed=True)

        assert session.state.thinking is False
        assert saves == [False]
        assert create_llm_kwargs[0]["thinking"] is False

    @pytest.mark.asyncio
    async def test_resumed_with_content_pins_false_when_requested_false(
        self, session, config, monkeypatch
    ):
        """Same legacy path when the user requested False already — no surprise,
        still pins to False."""
        create_llm_kwargs = _patch_create_deps(monkeypatch)
        saves = _suppress_save_state(monkeypatch)
        _simulate_resumed_with_content(session)

        await KimiCLI.create(session, config=config, thinking=False, resumed=True)

        assert session.state.thinking is False
        assert saves == [False]
        assert create_llm_kwargs[0]["thinking"] is False


class TestThinkingPinPersistence:
    """Failure-mode tests around when the pin is persisted to disk."""

    @pytest.mark.asyncio
    async def test_pin_not_persisted_when_create_llm_raises(self, session, config, monkeypatch):
        """If create_llm raises, the in-memory pin must not be written to disk.
        Leaving the pin would lock the user out of retrying with a different
        thinking mode after fixing whatever made create_llm fail."""
        _patch_create_deps(monkeypatch)
        saves = _suppress_save_state(monkeypatch)

        def raising_create_llm(*args, **kwargs):
            raise RuntimeError("simulated LLM init failure")

        monkeypatch.setattr(app_module, "create_llm", raising_create_llm)

        with pytest.raises(RuntimeError, match="simulated LLM init failure"):
            await KimiCLI.create(session, config=config, thinking=True, resumed=False)

        # save_state must never have run — the pin stays unpersisted on disk.
        assert saves == []


class TestThinkingPinResumedSession:
    """Resumed sessions with a recorded thinking mode."""

    @pytest.mark.asyncio
    async def test_pinned_true_overrides_requested_false(self, session, config, monkeypatch):
        create_llm_kwargs = _patch_create_deps(monkeypatch)
        saves = _suppress_save_state(monkeypatch)
        session.state = SessionState(thinking=True)

        await KimiCLI.create(session, config=config, thinking=False, resumed=True)

        # Pin enforced — LLM receives the pinned value, not the requested one.
        assert create_llm_kwargs[0]["thinking"] is True
        assert session.state.thinking is True
        # No re-save when the pin already exists.
        assert saves == []

    @pytest.mark.asyncio
    async def test_pinned_false_overrides_config_default_true(self, session, config, monkeypatch):
        config.default_thinking = True
        create_llm_kwargs = _patch_create_deps(monkeypatch)
        _suppress_save_state(monkeypatch)
        session.state = SessionState(thinking=False)

        await KimiCLI.create(session, config=config, thinking=None, resumed=True)

        assert create_llm_kwargs[0]["thinking"] is False
        assert session.state.thinking is False

    @pytest.mark.asyncio
    async def test_matching_request_does_not_warn(self, session, config, monkeypatch, caplog):
        create_llm_kwargs = _patch_create_deps(monkeypatch)
        _suppress_save_state(monkeypatch)
        session.state = SessionState(thinking=True)

        with caplog.at_level("WARNING"):
            await KimiCLI.create(session, config=config, thinking=True, resumed=True)

        assert create_llm_kwargs[0]["thinking"] is True
        # No warning about ignoring a thinking request.
        assert not any("ignoring requested thinking" in r.message for r in caplog.records)
