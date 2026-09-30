from __future__ import annotations

import unittest

from claude_web_api.providers.contracts import (
    ProviderEventKind,
    ProviderTurnRequest,
)
from claude_web_api.providers.deepseek_web import DeepSeekWebProviderError
from claude_web_api.session.deepseek import (
    DEEPSEEK_COMPLETION_PATH,
    DeepSeekCamoufoxTransport,
    _resolve_model,
)


class FakePage:
    def __init__(self) -> None:
        self.closed = False
        self.binding = None
        self.calls = []

    def is_closed(self):
        return self.closed

    async def evaluate(self, script, arg=None):
        self.calls.append((script, arg))
        if arg is None:
            return {"authenticated": True, "reason": None}
        operation_id = arg["operationId"]
        await self.binding(
            None,
            {
                "operationId": operation_id,
                "event": "ready",
                "data": {
                    "request_message_id": 20,
                    "response_message_id": 21,
                },
            },
        )
        await self.binding(
            None,
            {
                "operationId": operation_id,
                "event": "message",
                "data": {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [{"type": "THINK", "content": "why"}],
                },
            },
        )
        await self.binding(
            None,
            {
                "operationId": operation_id,
                "event": "message",
                "data": {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [{"type": "RESPONSE", "content": "answer"}],
                },
            },
        )
        await self.binding(
            None,
            {
                "operationId": operation_id,
                "event": "close",
                "data": {},
            },
        )
        return {"sessionId": "session-a"}


class DeepSeekTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.transport = DeepSeekCamoufoxTransport(
            profile_id="deepseek-a",
            profile_path="deepseek-test-profile",
            display_name="DeepSeek A",
        )
        self.page = FakePage()
        self.page.binding = self.transport._receive_stream
        self.transport._page = self.page
        self.transport._ready = True
        self.transport._phase = "idle"

    def test_model_aliases_are_explicit(self) -> None:
        self.assertEqual(("default", "deepseek-web"), _resolve_model("auto"))
        self.assertEqual(
            ("expert", "deepseek-reasoner"),
            _resolve_model("deepseek-v4-pro"),
        )
        with self.assertRaises(ValueError):
            _resolve_model("claude-opus")

    async def test_completion_stays_inside_page_and_preserves_lineage(self):
        events = []
        turn = await self.transport.complete(
            ProviderTurnRequest(
                message="question",
                model="deepseek-chat",
                reasoning_mode="show",
                client_session_id="client-a",
            ),
            event_sink=events.append,
        )

        self.assertEqual("answer", turn.content)
        self.assertEqual("why", turn.thinking)
        self.assertEqual("deepseek-web", turn.model)
        self.assertEqual(
            [ProviderEventKind.THINKING_DELTA, ProviderEventKind.TEXT_DELTA],
            [event.kind for event in events],
        )
        self.assertEqual("session-a", self.transport._session_id)
        self.assertEqual(21, self.transport._parent_message_id)
        sent = self.page.calls[-1][1]
        self.assertEqual(DEEPSEEK_COMPLETION_PATH, sent["completionPath"])
        self.assertEqual("default", sent["modelType"])
        self.assertTrue(sent["thinkingEnabled"])
        self.assertNotIn("token", sent)
        self.assertNotIn("cookie", sent)

        await self.transport.complete(
            ProviderTurnRequest(
                message="follow-up",
                client_session_id="client-a",
            )
        )
        follow_up = self.page.calls[-1][1]
        self.assertEqual("session-a", follow_up["sessionId"])
        self.assertEqual(21, follow_up["parentMessageId"])
        self.assertFalse(follow_up["newConversation"])

    async def test_client_change_starts_fresh_conversation(self) -> None:
        self.transport._session_id = "old-session"
        self.transport._parent_message_id = 99
        self.transport._client_session_id = "client-a"

        await self.transport.complete(
            ProviderTurnRequest(
                message="new task",
                client_session_id="client-b",
            )
        )

        sent = self.page.calls[-1][1]
        self.assertIsNone(sent["sessionId"])
        self.assertIsNone(sent["parentMessageId"])
        self.assertTrue(sent["newConversation"])

    async def test_ephemeral_mode_fails_until_verified(self) -> None:
        with self.assertRaises(DeepSeekWebProviderError):
            await self.transport.complete(
                ProviderTurnRequest(
                    message="private",
                    privacy_mode="ephemeral",
                )
            )


if __name__ == "__main__":
    unittest.main()
