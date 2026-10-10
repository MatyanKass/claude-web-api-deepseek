from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from claude_web_api import completions, runtime
from claude_web_api.providers.contracts import (
    ProviderEvent,
    ProviderEventKind,
    ProviderTurn,
)
from claude_web_api.providers.deepseek_web import DEEPSEEK_WEB_PROVIDER_ID


class FakeDeepSeekProvider:
    def __init__(self) -> None:
        self.request = None
        self.requests: list = []
        self.conversation_id: str | None = None
        self._chats = 0

    async def complete(self, request, *, event_sink=None):
        self.request = request
        self.requests.append(request)
        if request.new_conversation or self.conversation_id is None:
            self._chats += 1
            self.conversation_id = f"chat-{self._chats}"
        if event_sink is not None:
            event_sink(
                ProviderEvent(
                    kind=ProviderEventKind.TEXT_DELTA,
                    text="hello",
                )
            )
        return ProviderTurn(
            content="hello",
            thinking="reasoning",
            model="deepseek-web",
            stop_reason="end_turn",
        )


class DeepSeekCompletionRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        completions._deepseek_lineage.clear()
        self.addCleanup(completions._deepseek_lineage.clear)

    async def test_active_deepseek_profile_uses_provider_registry(self) -> None:
        provider = FakeDeepSeekProvider()
        registry = SimpleNamespace(resolve=lambda **_: provider)
        body = completions.CompletionsIn(
            model="deepseek-web",
            messages=[{"role": "user", "content": "Say hello"}],
        )
        emitted: list[dict[str, object]] = []
        with (
            patch.object(
                runtime,
                "active_provider_id",
                return_value=DEEPSEEK_WEB_PROVIDER_ID,
            ),
            patch.object(runtime, "active_profile_id", return_value="deepseek-a"),
            patch.object(runtime, "provider_registry", registry),
            patch.object(
                runtime.control,
                "behavior_snapshot",
                return_value=(
                    {"thinking": "show", "privacy": "keep"},
                    "",
                ),
            ),
        ):
            turn = await completions.run_native_with_limits(
                body,
                client_session_id="client-a",
                event_sink=emitted.append,
            )

        self.assertEqual("hello", turn.content)
        self.assertEqual("deepseek-web", turn.model)
        self.assertIsNotNone(provider.request)
        self.assertEqual("Say hello", provider.request.message)
        self.assertTrue(provider.request.new_conversation)
        self.assertEqual("client-a", provider.request.client_session_id)
        self.assertEqual("text_delta", emitted[0]["type"])

    async def test_deepseek_rejects_openai_tools_before_browser_request(self) -> None:
        provider = FakeDeepSeekProvider()
        registry = SimpleNamespace(resolve=lambda **_: provider)
        body = completions.CompletionsIn(
            model="deepseek-web",
            messages=[{"role": "user", "content": "Use a tool"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "example", "parameters": {}},
                }
            ],
        )
        with (
            patch.object(
                runtime,
                "active_provider_id",
                return_value=DEEPSEEK_WEB_PROVIDER_ID,
            ),
            patch.object(runtime, "active_profile_id", return_value="deepseek-a"),
            patch.object(runtime, "provider_registry", registry),
            patch.object(
                runtime.control,
                "behavior_snapshot",
                return_value=(
                    {"thinking": "auto", "privacy": "keep"},
                    "",
                ),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await completions.run_native_with_limits(
                    body,
                    client_session_id="client-a",
                    event_sink=None,
                )

        self.assertEqual(400, raised.exception.status_code)
        self.assertIsNone(provider.request)


class DeepSeekConversationLineageTests(unittest.IsolatedAsyncioTestCase):
    """A coding client's system prompt and history must reach DeepSeek."""

    SYSTEM = {
        "role": "system",
        "content": "You are Roo.\n\nTOOL USE\n<read_file><path>x</path></read_file>",
    }

    def setUp(self) -> None:
        completions._deepseek_lineage.clear()
        self.addCleanup(completions._deepseek_lineage.clear)
        self.provider = FakeDeepSeekProvider()
        registry = SimpleNamespace(resolve=lambda **_: self.provider)
        patches = (
            patch.object(
                runtime,
                "active_provider_id",
                return_value=DEEPSEEK_WEB_PROVIDER_ID,
            ),
            patch.object(runtime, "active_profile_id", return_value="deepseek-a"),
            patch.object(runtime, "provider_registry", registry),
            patch.object(
                runtime.control,
                "behavior_snapshot",
                return_value=({"thinking": "auto", "privacy": "keep"}, ""),
            ),
        )
        for active in patches:
            active.start()
            self.addCleanup(active.stop)

    async def _send(self, messages, *, new_chat: bool = False):
        body = completions.CompletionsIn(
            model="deepseek-reasoner",
            messages=messages,
            new_chat=new_chat,
        )
        await completions.run_native_with_limits(
            body,
            client_session_id=None,
            event_sink=None,
        )
        return self.provider.request

    async def test_first_turn_carries_the_client_system_prompt(self) -> None:
        request = await self._send(
            [self.SYSTEM, {"role": "user", "content": "<task>fix bug</task>"}]
        )
        self.assertTrue(request.new_conversation)
        self.assertIn("<read_file><path>x</path></read_file>", request.message)
        self.assertIn("<task>fix bug</task>", request.message)
        # Code and XML must reach the model unquoted.
        self.assertNotIn("> <task>", request.message)

    async def test_continuation_sends_only_the_new_messages(self) -> None:
        first = [self.SYSTEM, {"role": "user", "content": "<task>fix bug</task>"}]
        await self._send(first)
        second = await self._send(
            [
                *first,
                {"role": "assistant", "content": "client-rewritten reply"},
                {"role": "user", "content": "[read_file] Result: print(1)"},
            ]
        )
        self.assertFalse(second.new_conversation)
        self.assertEqual("[read_file] Result: print(1)", second.message)
        self.assertEqual(1, self.provider._chats)

    async def test_diverged_history_rebuilds_a_fresh_chat(self) -> None:
        await self._send(
            [self.SYSTEM, {"role": "user", "content": "<task>first task</task>"}]
        )
        request = await self._send(
            [
                self.SYSTEM,
                {"role": "user", "content": "<task>other task</task>"},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "continue"},
            ]
        )
        self.assertTrue(request.new_conversation)
        self.assertIn("TOOL USE", request.message)
        self.assertIn("other task", request.message)
        self.assertIn("earlier answer", request.message)
        self.assertIn("continue", request.message)

    async def test_changed_system_prompt_rebuilds_a_fresh_chat(self) -> None:
        first = [self.SYSTEM, {"role": "user", "content": "hi"}]
        await self._send(first)
        request = await self._send(
            [
                {"role": "system", "content": "Architect mode"},
                first[1],
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "plan it"},
            ]
        )
        self.assertTrue(request.new_conversation)
        self.assertIn("Architect mode", request.message)

    async def test_lost_browser_chat_rebuilds_from_history(self) -> None:
        first = [self.SYSTEM, {"role": "user", "content": "hi"}]
        await self._send(first)
        self.provider.conversation_id = None  # browser restarted
        request = await self._send(
            [
                *first,
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "next"},
            ]
        )
        self.assertTrue(request.new_conversation)
        self.assertIn("EARLIER CONVERSATION", request.message)

    async def test_failed_turn_forgets_the_lineage(self) -> None:
        first = [self.SYSTEM, {"role": "user", "content": "hi"}]
        await self._send(first)

        async def broken(request, *, event_sink=None):
            raise RuntimeError("stream died")

        original = self.provider.complete
        self.provider.complete = broken
        with self.assertRaises(RuntimeError):
            await self._send(
                [*first, {"role": "assistant", "content": "a"},
                 {"role": "user", "content": "b"}]
            )
        self.provider.complete = original
        request = await self._send(
            [*first, {"role": "assistant", "content": "a"},
             {"role": "user", "content": "b"}]
        )
        self.assertTrue(request.new_conversation)


if __name__ == "__main__":
    unittest.main()
