"""Authenticated DeepSeek Web transport backed by a Playwright page.

Authentication material never crosses the browser boundary.  The page reads
its own ``userToken``, obtains and solves the current proof-of-work challenge,
opens the private completion stream and forwards only parsed SSE envelopes to
Python.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

from camoufox.async_api import AsyncCamoufox

from claude_web_api.control import proxy_relay
from claude_web_api.providers.contracts import (
    ProviderEventSink,
    ProviderHealth,
    ProviderProfileIdentity,
    ProviderTurn,
    ProviderTurnRequest,
)
from claude_web_api.providers.deepseek_web import (
    DEEPSEEK_WEB_ORIGIN,
    DEEPSEEK_WEB_PROVIDER_ID,
    DeepSeekProviderNotReadyError,
    DeepSeekStreamDecoder,
    DeepSeekWebProviderError,
)

DEEPSEEK_COMPLETION_PATH = "/api/v0/chat/completion"
DEFAULT_POW_WORKER_URL = (
    "https://fe-static.deepseek.com/chat/static/76608.8f2a9fa413.js"
)

_AUTH_PROBE_SCRIPT = r"""
async () => {
  const raw = localStorage.getItem("userToken");
  if (!raw) return {authenticated: false, reason: "missing_user_token"};
  let token = raw;
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && typeof parsed.value === "string") {
      token = parsed.value;
    } else if (typeof parsed === "string") {
      token = parsed;
    }
  } catch {}
  if (!token) return {authenticated: false, reason: "empty_user_token"};
  try {
    const response = await fetch("/api/v0/chat_session/fetch_page?count=1", {
      cache: "no-store",
      credentials: "include",
      headers: {authorization: "Bearer " + token, accept: "application/json"}
    });
    return {
      authenticated: response.ok,
      reason: response.ok ? null : "auth_probe_http_" + response.status
    };
  } catch (error) {
    return {authenticated: false, reason: String(error)};
  }
}
"""

_COMPLETE_SCRIPT = r"""
async (input) => {
  const isRecord = (value) => (
    value !== null && typeof value === "object" && !Array.isArray(value)
  );
  const responseJson = async (response, label) => {
    let body;
    try {
      body = await response.json();
    } catch {
      throw new Error(label + " returned invalid JSON (HTTP " + response.status + ")");
    }
    if (!response.ok) {
      throw new Error(label + " failed with HTTP " + response.status);
    }
    return body;
  };
  const tokenFromPage = () => {
    const raw = localStorage.getItem("userToken");
    if (!raw) return "";
    try {
      const parsed = JSON.parse(raw);
      if (isRecord(parsed) && typeof parsed.value === "string") return parsed.value;
      if (typeof parsed === "string") return parsed;
    } catch {}
    return raw;
  };
  const token = tokenFromPage();
  if (!token) throw new Error("DeepSeek login is required");
  const headers = {
    authorization: "Bearer " + token,
    "content-type": "application/json",
    accept: "application/json",
    "x-client-platform": "web",
    "x-client-version": "2.2.0",
    "x-client-locale": "en_US",
    "x-client-bundle-id": "com.deepseek.chat"
  };

  const challengeResponse = await fetch("/api/v0/chat/create_pow_challenge", {
    method: "POST",
    credentials: "include",
    headers,
    body: JSON.stringify({target_path: input.completionPath})
  });
  const challengeBody = await responseJson(challengeResponse, "PoW challenge");
  const challenge = challengeBody?.data?.biz_data?.challenge;
  if (!isRecord(challenge)) throw new Error("PoW challenge is missing");

  const workerResponse = await fetch(input.powWorkerUrl, {cache: "no-store"});
  if (!workerResponse.ok) {
    throw new Error("PoW worker failed with HTTP " + workerResponse.status);
  }
  const workerSource = await workerResponse.text();
  const workerObjectUrl = URL.createObjectURL(
    new Blob([workerSource], {type: "application/javascript"})
  );
  let answer;
  try {
    answer = await new Promise((resolve, reject) => {
      const worker = new Worker(workerObjectUrl);
      const timer = setTimeout(() => {
        worker.terminate();
        reject(new Error("PoW worker timed out"));
      }, input.powTimeoutMs);
      worker.onmessage = (event) => {
        clearTimeout(timer);
        worker.terminate();
        const payload = isRecord(event.data) ? event.data : null;
        if (payload?.type === "pow-answer" && isRecord(payload.answer)) {
          resolve(payload.answer);
        } else {
          reject(new Error("PoW worker returned an invalid answer"));
        }
      };
      worker.onerror = (event) => {
        clearTimeout(timer);
        worker.terminate();
        reject(new Error(event.message || "PoW worker failed"));
      };
      worker.postMessage({
        type: "pow-challenge",
        challenge: {
          algorithm: challenge.algorithm,
          challenge: challenge.challenge,
          salt: challenge.salt,
          difficulty: challenge.difficulty,
          signature: challenge.signature,
          expireAt: challenge.expire_at
        }
      });
    });
  } finally {
    URL.revokeObjectURL(workerObjectUrl);
  }

  const powPayload = {
    algorithm: answer.algorithm,
    challenge: answer.challenge,
    salt: answer.salt,
    answer: answer.answer,
    signature: answer.signature,
    target_path: input.completionPath
  };
  const powBytes = new TextEncoder().encode(JSON.stringify(powPayload));
  let powBinary = "";
  for (const byte of powBytes) powBinary += String.fromCharCode(byte);
  const powHeader = btoa(powBinary);

  let sessionId = input.sessionId;
  if (input.newConversation || !sessionId) {
    const createResponse = await fetch("/api/v0/chat_session/create", {
      method: "POST",
      credentials: "include",
      headers,
      body: "{}"
    });
    const createBody = await responseJson(createResponse, "Session creation");
    sessionId = createBody?.data?.biz_data?.chat_session?.id;
    if (typeof sessionId !== "string" || !sessionId) {
      throw new Error("Session creation returned no session id");
    }
  }

  const completionResponse = await fetch(input.completionPath, {
    method: "POST",
    credentials: "include",
    headers: {
      ...headers,
      accept: "text/event-stream",
      "x-ds-pow-response": powHeader
    },
    body: JSON.stringify({
      chat_session_id: sessionId,
      parent_message_id: input.parentMessageId,
      model_type: input.modelType,
      prompt: input.prompt,
      ref_file_ids: [],
      thinking_enabled: input.thinkingEnabled,
      search_enabled: false,
      action: null,
      preempt: false
    })
  });
  if (!completionResponse.ok) {
    throw new Error("Completion failed with HTTP " + completionResponse.status);
  }
  if (!completionResponse.body) throw new Error("Completion returned no stream");

  const emitBlock = async (block) => {
    if (!block.trim()) return;
    let event = "message";
    const dataLines = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    let data;
    try {
      data = JSON.parse(dataLines.join("\n"));
    } catch {
      return;
    }
    if (!isRecord(data)) return;
    await globalThis.__deepseek_stream({operationId: input.operationId, event, data});
  };
  const reader = completionResponse.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const step = await reader.read();
    if (step.done) break;
    buffer += decoder.decode(step.value, {stream: true});
    for (;;) {
      const boundary = /\r?\n\r?\n/.exec(buffer);
      if (!boundary) break;
      const block = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary[0].length);
      await emitBlock(block);
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) await emitBlock(buffer);
  return {sessionId};
}
"""


class DeepSeekCamoufoxTransport:
    """One persistent DeepSeek browser profile and its conversation lineage."""

    def __init__(
        self,
        *,
        profile_id: str,
        profile_path: str | Path,
        display_name: str,
        headless: bool = True,
        pow_worker_url: str | None = None,
        outbound_proxy: dict[str, Any] | None = None,
    ) -> None:
        self.profile_id = str(profile_id)
        self.profile_path = Path(profile_path).expanduser().resolve()
        self.display_name = str(display_name)
        self.headless = bool(headless)
        self.outbound_proxy = outbound_proxy
        self.pow_worker_url = (
            pow_worker_url
            or os.getenv("DEEPSEEK_POW_WORKER_URL")
            or DEFAULT_POW_WORKER_URL
        )
        self._lock = asyncio.Lock()
        self._camoufox: Any = None
        self._relay: proxy_relay.Socks5Relay | None = None
        self._context: Any = None
        self._page: Any = None
        self._ready = False
        self._phase = "stopped"
        self._last_error: str | None = None
        self._session_id: str | None = None
        self._parent_message_id: str | int | None = None
        self._client_session_id: str | None = None
        self._active_operation_id: str | None = None
        self._active_decoder: DeepSeekStreamDecoder | None = None
        self._active_sink: ProviderEventSink | None = None

    @property
    def profile_identity(self) -> ProviderProfileIdentity:
        return ProviderProfileIdentity(
            provider=DEEPSEEK_WEB_PROVIDER_ID,
            profile_id=self.profile_id,
            display_name=self.display_name,
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            live=self._phase not in {"stopped", "browser_dead"},
            ready=self._ready,
            phase=self._phase,
            detail=self._last_error,
        )

    async def start(self) -> None:
        async with self._lock:
            if self._page is not None and not self._page.is_closed():
                await self._probe_auth_unlocked()
                return
            self.profile_path.mkdir(parents=True, exist_ok=True)
            self._phase = "starting_browser"
            try:
                self._relay = await proxy_relay.open_relay(
                    self.outbound_proxy
                )
                browser_proxy = proxy_relay.browser_proxy(
                    self.outbound_proxy,
                    self._relay,
                )
                launch: dict[str, Any] = {
                    "headless": self.headless,
                    "persistent_context": True,
                    "user_data_dir": str(self.profile_path),
                    "humanize": False,
                }
                if browser_proxy:
                    launch["proxy"] = browser_proxy
                    launch["geoip"] = True
                self._camoufox = AsyncCamoufox(
                    **launch,
                )
                self._context = await asyncio.wait_for(
                    self._camoufox.__aenter__(), timeout=90
                )
                pages = self._context.pages
                self._page = pages[0] if pages else await self._context.new_page()
                await self._page.expose_binding(
                    "__deepseek_stream", self._receive_stream
                )
                await self._page.goto(
                    DEEPSEEK_WEB_ORIGIN,
                    wait_until="domcontentloaded",
                    timeout=120_000,
                )
                await self._probe_auth_unlocked()
            except Exception as exc:
                self._ready = False
                self._phase = "browser_dead"
                self._last_error = f"{type(exc).__name__}: {exc}"
                camoufox = self._camoufox
                self._camoufox = None
                self._context = None
                self._page = None
                relay = self._relay
                self._relay = None
                if camoufox is not None:
                    try:
                        await asyncio.wait_for(
                            camoufox.__aexit__(None, None, None), timeout=15
                        )
                    except Exception:
                        pass
                if relay is not None:
                    await relay.stop()
                raise DeepSeekProviderNotReadyError(self._last_error) from exc

    async def stop(self) -> None:
        async with self._lock:
            camoufox = self._camoufox
            relay = self._relay
            self._camoufox = None
            self._relay = None
            self._context = None
            self._page = None
            self._ready = False
            self._phase = "stopped"
            self._clear_active()
            if camoufox is not None:
                try:
                    await asyncio.wait_for(
                        camoufox.__aexit__(None, None, None), timeout=15
                    )
                except Exception:
                    pass
            if relay is not None:
                await relay.stop()

    async def new_conversation(self) -> None:
        async with self._lock:
            self._reset_conversation()

    async def complete(
        self,
        request: ProviderTurnRequest,
        *,
        event_sink: ProviderEventSink | None = None,
    ) -> ProviderTurn:
        async with self._lock:
            if self._page is None or self._page.is_closed() or not self._ready:
                raise DeepSeekProviderNotReadyError(
                    self._last_error or "DeepSeek browser is not ready"
                )
            if request.privacy_mode == "ephemeral":
                raise DeepSeekWebProviderError(
                    "DeepSeek Web temporary-chat semantics are not verified"
                )
            if request.new_conversation or (
                request.client_session_id
                and self._client_session_id
                and request.client_session_id != self._client_session_id
            ):
                self._reset_conversation()
            self._client_session_id = request.client_session_id
            model_type, public_model = _resolve_model(request.model)
            operation_id = uuid.uuid4().hex
            decoder = DeepSeekStreamDecoder(model=public_model)
            self._active_operation_id = operation_id
            self._active_decoder = decoder
            self._active_sink = event_sink
            self._phase = "generating"
            try:
                result = await asyncio.wait_for(
                    self._page.evaluate(
                        _COMPLETE_SCRIPT,
                        {
                            "operationId": operation_id,
                            "completionPath": DEEPSEEK_COMPLETION_PATH,
                            "powWorkerUrl": self.pow_worker_url,
                            "powTimeoutMs": min(
                                120_000,
                                max(5_000, int(request.timeout_seconds * 1000)),
                            ),
                            "sessionId": self._session_id,
                            "newConversation": self._session_id is None,
                            "parentMessageId": self._parent_message_id,
                            "modelType": model_type,
                            "prompt": request.message,
                            "thinkingEnabled": request.reasoning_mode != "off",
                        },
                    ),
                    timeout=request.timeout_seconds,
                )
                if not isinstance(result, dict) or not result.get("sessionId"):
                    raise DeepSeekWebProviderError(
                        "DeepSeek completion returned invalid session state"
                    )
                if not decoder.closed:
                    raise DeepSeekWebProviderError(
                        "DeepSeek completion stream ended without a close event"
                    )
                self._session_id = str(result["sessionId"])
                self._parent_message_id = decoder.response_message_id
                self._phase = "idle"
                self._last_error = None
                return decoder.turn()
            except asyncio.TimeoutError as exc:
                self._phase = "idle"
                self._last_error = "DeepSeek completion timed out"
                self._reset_conversation()
                raise DeepSeekWebProviderError(self._last_error) from exc
            except Exception as exc:
                self._phase = "idle"
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._clear_active()

    async def _probe_auth_unlocked(self) -> None:
        result = await self._page.evaluate(_AUTH_PROBE_SCRIPT)
        authenticated = bool(
            isinstance(result, dict) and result.get("authenticated")
        )
        self._ready = authenticated
        if authenticated:
            self._phase = "idle"
            self._last_error = None
        else:
            reason = (
                str(result.get("reason") or "login_required")
                if isinstance(result, dict)
                else "invalid_auth_probe"
            )
            self._phase = "auth_required"
            self._last_error = f"DeepSeek authentication required: {reason}"

    async def _receive_stream(self, source: Any, payload: Any) -> None:
        del source
        if not isinstance(payload, dict):
            return
        if payload.get("operationId") != self._active_operation_id:
            return
        decoder = self._active_decoder
        data = payload.get("data")
        if decoder is None or not isinstance(data, dict):
            return
        events = decoder.feed(str(payload.get("event") or "message"), data)
        sink = self._active_sink
        if sink is not None:
            for event in events:
                sink(event)

    def _reset_conversation(self) -> None:
        self._session_id = None
        self._parent_message_id = None
        self._client_session_id = None

    def _clear_active(self) -> None:
        self._active_operation_id = None
        self._active_decoder = None
        self._active_sink = None


def _resolve_model(model: str | None) -> tuple[str, str]:
    normalized = str(model or "").strip().lower()
    if normalized in {
        "",
        "auto",
        "deepseek-web",
        "deepseek-chat",
        "deepseek-v4-flash",
        "default",
        "flash",
    }:
        return "default", "deepseek-web"
    if normalized in {
        "deepseek-reasoner",
        "deepseek-v4-pro",
        "expert",
        "pro",
    }:
        return "expert", "deepseek-reasoner"
    raise ValueError(f"unsupported DeepSeek Web model: {model!r}")


__all__ = [
    "DEEPSEEK_COMPLETION_PATH",
    "DEFAULT_POW_WORKER_URL",
    "DeepSeekCamoufoxTransport",
]
