from __future__ import annotations

import unittest

from claude_web_api.providers.contracts import (
    ProviderEvent,
    ProviderEventKind,
    ProviderHealth,
    ProviderProfileIdentity,
    ProviderToolResult,
    ProviderTurn,
    ProviderTurnRequest,
)
from claude_web_api.providers.deepseek_web import (
    DEEPSEEK_WEB_PROVIDER_ID,
    DeepSeekProviderNotReadyError,
    DeepSeekStreamDecoder,
    DeepSeekUnsupportedToolsError,
    DeepSeekWebProvider,
)


class FakeDeepSeekTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    @property
    def profile_identity(self) -> ProviderProfileIdentity:
        return ProviderProfileIdentity(
            provider=DEEPSEEK_WEB_PROVIDER_ID,
            profile_id="deepseek-a",
            display_name="DeepSeek A",
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(live=True, ready=True, phase="idle")

    async def start(self) -> None:
        self.calls.append(("start",))

    async def stop(self) -> None:
        self.calls.append(("stop",))

    async def new_conversation(self) -> None:
        self.calls.append(("new_conversation",))

    async def complete(self, request, *, event_sink=None) -> ProviderTurn:
        self.calls.append(("complete", request))
        if event_sink is not None:
            event_sink(
                ProviderEvent(
                    kind=ProviderEventKind.THINKING_DELTA,
                    text="considering",
                )
            )
            event_sink(
                ProviderEvent(
                    kind=ProviderEventKind.TEXT_DELTA,
                    text="answer",
                )
            )
        return ProviderTurn(
            content="answer",
            thinking="considering",
            model="deepseek-web",
            stop_reason="end_turn",
        )


class DeepSeekWebProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_stream_decoder_keeps_reasoning_and_output_separate(self) -> None:
        decoder = DeepSeekStreamDecoder(model="deepseek-web")

        self.assertEqual(
            (),
            decoder.feed(
                "ready",
                {"request_message_id": 10, "response_message_id": 11},
            ),
        )
        first = decoder.feed(
            "message",
            {
                "v": {
                    "response": {
                        "fragments": [{"type": "THINK", "content": "why"}],
                        "accumulated_token_usage": 2,
                    }
                }
            },
        )
        second = decoder.feed(
            "message",
            {
                "p": "response/fragments",
                "o": "APPEND",
                "v": [{"type": "RESPONSE", "content": "ans"}],
            },
        )
        tail = decoder.feed(
            "message",
            {
                "p": "response/fragments/-1/content",
                "o": "APPEND",
                "v": "wer",
            },
        )
        decoder.feed("close", {})

        self.assertEqual(10, decoder.request_message_id)
        self.assertEqual(11, decoder.response_message_id)
        self.assertEqual(
            [ProviderEventKind.THINKING_DELTA, ProviderEventKind.USAGE],
            [event.kind for event in first],
        )
        self.assertEqual(ProviderEventKind.TEXT_DELTA, second[0].kind)
        self.assertEqual(ProviderEventKind.TEXT_DELTA, tail[0].kind)
        turn = decoder.turn()
        self.assertEqual("why", turn.thinking)
        self.assertEqual("answer", turn.content)
        self.assertEqual({"output_tokens": 2}, turn.usage)
        self.assertEqual("end_turn", turn.stop_reason)

    def test_stream_decoder_handles_search_and_batch_usage(self) -> None:
        decoder = DeepSeekStreamDecoder()
        search = decoder.feed(
            "message",
            {
                "p": "response/fragments",
                "o": "APPEND",
                "v": [{"type": "SEARCH", "content": "docs"}],
            },
        )
        usage = decoder.feed(
            "message",
            {
                "p": "response",
                "o": "BATCH",
                "v": [{"p": "accumulated_token_usage", "v": 7}],
            },
        )

        self.assertEqual(ProviderEventKind.THINKING_DELTA, search[0].kind)
        self.assertEqual("\n[search] docs\n", search[0].text)
        self.assertEqual(ProviderEventKind.USAGE, usage[0].kind)
        self.assertEqual(7, decoder.turn().usage["output_tokens"])

    def test_unconfigured_provider_fails_closed(self) -> None:
        provider = DeepSeekWebProvider()

        self.assertFalse(provider.capabilities.streaming)
        self.assertFalse(provider.capabilities.thinking)
        self.assertEqual("unsupported", provider.capabilities.tool_continuation)
        self.assertFalse(provider.health().ready)
        self.assertEqual("protocol_unverified", provider.health().phase)

    async def test_unconfigured_provider_refuses_start_and_completion(self) -> None:
        provider = DeepSeekWebProvider()

        with self.assertRaises(DeepSeekProviderNotReadyError):
            await provider.start()
        with self.assertRaises(DeepSeekProviderNotReadyError):
            await provider.complete(ProviderTurnRequest(message="hello"))
        await provider.stop()

    async def test_verified_transport_delegates_text_and_stream_events(self) -> None:
        transport = FakeDeepSeekTransport()
        provider = DeepSeekWebProvider(transport)
        events: list[ProviderEvent] = []

        await provider.start()
        await provider.new_conversation()
        turn = await provider.complete(
            ProviderTurnRequest(
                message="hello",
                reasoning_mode="show",
                client_session_id="client-a",
            ),
            event_sink=events.append,
        )
        await provider.stop()

        self.assertTrue(provider.capabilities.streaming)
        self.assertTrue(provider.capabilities.thinking)
        self.assertEqual("answer", turn.content)
        self.assertEqual("considering", turn.thinking)
        self.assertEqual(
            [
                ProviderEventKind.THINKING_DELTA,
                ProviderEventKind.TEXT_DELTA,
            ],
            [event.kind for event in events],
        )
        self.assertEqual(
            ["start", "new_conversation", "complete", "stop"],
            [str(call[0]) for call in transport.calls],
        )

    async def test_tools_are_rejected_without_text_parsing(self) -> None:
        provider = DeepSeekWebProvider(FakeDeepSeekTransport())
        request = ProviderTurnRequest(
            message="use a tool",
            tools=({"name": "Read"},),
        )

        with self.assertRaises(DeepSeekUnsupportedToolsError):
            await provider.complete(request)
        with self.assertRaises(DeepSeekUnsupportedToolsError):
            await provider.continue_with_tool_results(
                [ProviderToolResult(tool_use_id="tool-1", content="result")]
            )


if __name__ == "__main__":
    unittest.main()
