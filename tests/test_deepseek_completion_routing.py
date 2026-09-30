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

    async def complete(self, request, *, event_sink=None):
        self.request = request
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
        self.assertIn("Say hello", provider.request.message)
        self.assertFalse(provider.request.new_conversation)
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


if __name__ == "__main__":
    unittest.main()
