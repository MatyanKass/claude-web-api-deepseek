"""Provider boundary for an authenticated DeepSeek Web session.

DeepSeek's browser protocol is private and guarded by a proof-of-work
challenge.  This module keeps the public provider contract independent from
the browser implementation and refuses to claim capabilities until a verified
transport is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from claude_web_api.providers.contracts import (
    ProviderCapabilities,
    ProviderEvent,
    ProviderEventKind,
    ProviderEventSink,
    ProviderHealth,
    ProviderProfileIdentity,
    ProviderToolResult,
    ProviderTurn,
    ProviderTurnRequest,
    ToolContinuation,
)

DEEPSEEK_WEB_PROVIDER_ID = "deepseek_web"
DEEPSEEK_WEB_ORIGIN = "https://chat.deepseek.com"


class DeepSeekWebProviderError(RuntimeError):
    """Base error for the DeepSeek Web provider."""


class DeepSeekProviderNotReadyError(DeepSeekWebProviderError):
    """Raised when no verified browser transport is available."""


class DeepSeekUnsupportedToolsError(DeepSeekWebProviderError):
    """Raised when a client requests native tools from DeepSeek Web."""


class DeepSeekStreamDecoder:
    """Normalize the observed DeepSeek SSE patch stream.

    The Web response alternates typed fragments with content-only append
    patches.  Keeping the active fragment channel prevents reasoning text from
    leaking into the final answer.
    """

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model
        self.current_kind = ProviderEventKind.THINKING_DELTA
        self.thinking_parts: list[str] = []
        self.content_parts: list[str] = []
        self.output_tokens = 0
        self.request_message_id: str | int | None = None
        self.response_message_id: str | int | None = None
        self.closed = False

    def feed(
        self,
        event: str,
        data: Mapping[str, Any],
    ) -> tuple[ProviderEvent, ...]:
        if event == "ready":
            self.request_message_id = _message_id(data.get("request_message_id"))
            self.response_message_id = _message_id(data.get("response_message_id"))
            return ()
        if event == "close":
            self.closed = True
            return ()
        if event == "title":
            return ()

        updates: list[ProviderEvent] = []
        response = _nested_response(data)
        if response is not None:
            updates.extend(self._fragments(response.get("fragments")))
            self._set_usage(response.get("accumulated_token_usage"), updates)
            return tuple(updates)

        if data.get("p") == "response/fragments" and data.get("o") == "APPEND":
            updates.extend(self._fragments(data.get("v")))
            return tuple(updates)

        content_append = (
            data.get("p") == "response/fragments/-1/content"
            and data.get("o") == "APPEND"
        )
        root_append = "p" not in data and data.get("o") == "APPEND" and "v" in data
        bare_string = (
            "p" not in data and "o" not in data and isinstance(data.get("v"), str)
        )
        if content_append or root_append or bare_string:
            update = self._text_update(str(data.get("v") or ""))
            return (update,) if update is not None else ()

        if (
            data.get("p") == "response"
            and data.get("o") == "BATCH"
            and isinstance(data.get("v"), list)
        ):
            for patch in data["v"]:
                if (
                    isinstance(patch, Mapping)
                    and patch.get("p") == "accumulated_token_usage"
                ):
                    self._set_usage(patch.get("v"), updates)
        return tuple(updates)

    def turn(self) -> ProviderTurn:
        return ProviderTurn(
            content="".join(self.content_parts) or None,
            thinking="".join(self.thinking_parts) or None,
            usage={"output_tokens": self.output_tokens},
            model=self.model,
            stop_reason="end_turn" if self.closed else None,
        )

    def _fragments(self, value: Any) -> list[ProviderEvent]:
        if not isinstance(value, list):
            return []
        updates: list[ProviderEvent] = []
        for fragment in value:
            if not isinstance(fragment, Mapping):
                continue
            fragment_type = str(fragment.get("type") or "")
            content = str(fragment.get("content") or "")
            if fragment_type == "THINK":
                self.current_kind = ProviderEventKind.THINKING_DELTA
                update = self._text_update(content)
            elif fragment_type == "RESPONSE":
                self.current_kind = ProviderEventKind.TEXT_DELTA
                update = self._text_update(content)
            elif fragment_type in {"SEARCH", "SEARCH_REF"}:
                self.current_kind = ProviderEventKind.THINKING_DELTA
                note = f"\n[search] {content}\n" if content else "\n[search]\n"
                update = self._text_update(note)
            else:
                update = None
            if update is not None:
                updates.append(update)
        return updates

    def _text_update(self, text: str) -> ProviderEvent | None:
        if not text:
            return None
        if self.current_kind is ProviderEventKind.TEXT_DELTA:
            self.content_parts.append(text)
        else:
            self.thinking_parts.append(text)
        return ProviderEvent(
            kind=self.current_kind,
            text=text,
            metadata={"provider": DEEPSEEK_WEB_PROVIDER_ID},
        )

    def _set_usage(self, value: Any, updates: list[ProviderEvent]) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        self.output_tokens = max(0, int(value))
        updates.append(
            ProviderEvent(
                kind=ProviderEventKind.USAGE,
                metadata={"output_tokens": self.output_tokens},
            )
        )


def _message_id(value: Any) -> str | int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (str, int)) else None


def _nested_response(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = data.get("v")
    if not isinstance(value, Mapping):
        return None
    response = value.get("response")
    return response if isinstance(response, Mapping) else None


@runtime_checkable
class DeepSeekBrowserTransport(Protocol):
    """Verified browser-specific implementation used by the provider."""

    @property
    def profile_identity(self) -> ProviderProfileIdentity | None: ...

    def health(self) -> ProviderHealth: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def new_conversation(self) -> None: ...

    async def complete(
        self,
        request: ProviderTurnRequest,
        *,
        event_sink: ProviderEventSink | None = None,
    ) -> ProviderTurn: ...


class DeepSeekWebProvider:
    """Expose a verified DeepSeek Web browser transport to the API layer.

    DeepSeek Web does not expose native function calling.  The bridge keeps
    tool continuation disabled instead of deriving tool calls from model text.
    """

    def __init__(self, transport: DeepSeekBrowserTransport | None = None) -> None:
        if transport is not None and not isinstance(
            transport, DeepSeekBrowserTransport
        ):
            raise TypeError("transport does not implement DeepSeekBrowserTransport")
        self._transport = transport

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            tool_continuation=ToolContinuation.UNSUPPORTED,
            streaming=self._transport is not None,
            thinking=self._transport is not None,
            profiles=True,
        )

    @property
    def profile_identity(self) -> ProviderProfileIdentity | None:
        if self._transport is None:
            return None
        return self._transport.profile_identity

    def health(self) -> ProviderHealth:
        if self._transport is None:
            return ProviderHealth(
                live=False,
                ready=False,
                phase="protocol_unverified",
                detail=(
                    "DeepSeek Web transport is not configured; authenticated "
                    "browser auth, proof-of-work and SSE parsing must be "
                    "verified before activation"
                ),
            )
        return self._transport.health()

    async def start(self) -> None:
        await self._require_transport().start()

    async def stop(self) -> None:
        if self._transport is not None:
            await self._transport.stop()

    async def new_conversation(self) -> None:
        await self._require_transport().new_conversation()

    async def complete(
        self,
        request: ProviderTurnRequest,
        *,
        event_sink: ProviderEventSink | None = None,
    ) -> ProviderTurn:
        if request.tools:
            raise DeepSeekUnsupportedToolsError(
                "DeepSeek Web has no verified native function-calling channel"
            )
        return await self._require_transport().complete(
            request,
            event_sink=event_sink,
        )

    async def continue_with_tool_results(
        self,
        results: Sequence[ProviderToolResult],
        *,
        timeout_seconds: float = 300.0,
        client_session_id: str | None = None,
        event_sink: ProviderEventSink | None = None,
    ) -> ProviderTurn:
        del results, timeout_seconds, client_session_id, event_sink
        raise DeepSeekUnsupportedToolsError(
            "DeepSeek Web cannot continue native host tool calls"
        )

    def _require_transport(self) -> DeepSeekBrowserTransport:
        if self._transport is None:
            raise DeepSeekProviderNotReadyError(
                "DeepSeek Web transport is not configured"
            )
        return self._transport


__all__ = [
    "DEEPSEEK_WEB_ORIGIN",
    "DEEPSEEK_WEB_PROVIDER_ID",
    "DeepSeekBrowserTransport",
    "DeepSeekProviderNotReadyError",
    "DeepSeekStreamDecoder",
    "DeepSeekUnsupportedToolsError",
    "DeepSeekWebProvider",
    "DeepSeekWebProviderError",
]
