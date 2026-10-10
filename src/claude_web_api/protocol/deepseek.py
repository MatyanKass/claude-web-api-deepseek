"""Chat Completions history -> the text a DeepSeek Web chat receives.

DeepSeek Web keeps its own conversation server-side and accepts one plain
text prompt per turn. It has no system role and no native tools, so a coding
client such as Roo Code (XML tool protocol) carries its whole contract in the
system prompt. That prompt must reach the model, or the agent loop has
nothing to work with.

Sending the full transcript every turn would waste the web context window,
so the bridge remembers which client messages the live DeepSeek chat has
already seen (its *lineage*). A request whose history extends that lineage
sends only the new messages; anything else — a different task, an edited or
condensed history, a changed system prompt, a restarted browser — opens a
fresh chat with the system prompt and the full transcript.

Pure translation: no I/O, no routing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from claude_web_api.protocol.openai import (
    raw_text_content,
    strip_client_scaffolding,
    text_content,
)

# DeepSeek Web truncates nothing for us and rejects nothing up front either,
# so the bridge bounds a rebuilt transcript itself. Whole messages are dropped
# from the oldest end and the omission is stated in the prompt.
MAX_TRANSCRIPT_CHARS = 240_000

_SYSTEM_ROLES = frozenset({"system", "developer"})
_ASSISTANT_KEY = "assistant"


@dataclass(frozen=True)
class DeepSeekLineage:
    """What one live DeepSeek chat has already been shown."""

    conversation_id: str
    client_session_id: str | None
    system_digest: str
    message_keys: tuple[str, ...]


@dataclass(frozen=True)
class DeepSeekTurnPlan:
    prompt: str
    new_conversation: bool
    system_digest: str
    message_keys: tuple[str, ...]

    def lineage_after_reply(
        self,
        conversation_id: str,
        client_session_id: str | None,
    ) -> DeepSeekLineage:
        return DeepSeekLineage(
            conversation_id=conversation_id,
            client_session_id=client_session_id,
            system_digest=self.system_digest,
            message_keys=(*self.message_keys, _ASSISTANT_KEY),
        )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _message_text(message: dict[str, Any]) -> str:
    if message.get("role") == "tool":
        return raw_text_content(message)
    return strip_client_scaffolding(text_content(message))


def _message_key(message: dict[str, Any]) -> str:
    role = str(message.get("role", "")).lower()
    if role == "assistant":
        # Clients store the reply in their own shape (trimmed, reasoning
        # removed, XML re-serialised). DeepSeek wrote it, so only its position
        # matters for continuity.
        return _ASSISTANT_KEY
    return role + ":" + _digest(_message_text(message))


def _system_text(messages: list[dict[str, Any]]) -> str:
    parts = [
        text_content(message)
        for message in messages
        if str(message.get("role", "")).lower() in _SYSTEM_ROLES
    ]
    return "\n\n".join(part for part in parts if part)


def _dialogue(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        message
        for message in messages
        if str(message.get("role", "")).lower() not in _SYSTEM_ROLES
    ]


def _render(message: dict[str, Any]) -> str:
    role = str(message.get("role", "")).lower()
    text = _message_text(message)
    if role == "tool":
        name = str(message.get("name", "") or "tool")
        return f"TOOL RESULT ({name}):\n{text}"
    return f"{role.upper()}:\n{text}"


def _transcript(messages: list[dict[str, Any]]) -> str:
    rows = [_render(message) for message in messages if _message_text(message)]
    omitted = 0
    while rows and sum(len(row) + 2 for row in rows) > MAX_TRANSCRIPT_CHARS:
        rows.pop(0)
        omitted += 1
    if omitted:
        rows.insert(
            0,
            f"[{omitted} earliest message(s) omitted: the transcript exceeded "
            "the size the bridge sends to DeepSeek Web]",
        )
    return "\n\n".join(rows)


def _new_messages_text(messages: list[dict[str, Any]]) -> str:
    if len(messages) == 1 and messages[0].get("role") == "user":
        return _message_text(messages[0])
    return "\n\n".join(
        _render(message) for message in messages if _message_text(message)
    )


def _with_style(prompt: str, persona_instruction: str) -> str:
    persona = str(persona_instruction or "").strip()
    if not persona:
        return prompt
    return (
        "RESPONSE STYLE SELECTED BY THE USER:\n"
        + persona
        + "\n\n"
        + prompt
    )


def plan_turn(
    messages: list[dict[str, Any]],
    *,
    lineage: DeepSeekLineage | None,
    live_conversation_id: str | None,
    client_session_id: str | None,
    force_new: bool = False,
    persona_instruction: str = "",
) -> DeepSeekTurnPlan:
    """Decide whether this request continues the live chat, and what to send."""
    dialogue = _dialogue(messages)
    if not dialogue or dialogue[-1].get("role") not in ("user", "tool"):
        raise ValueError("request has no actionable user message")
    if not _message_text(dialogue[-1]):
        raise ValueError("request has no actionable user message")

    system = _system_text(messages)
    system_digest = _digest(system)
    keys = tuple(_message_key(message) for message in dialogue)

    continues = bool(
        not force_new
        and lineage is not None
        and live_conversation_id
        and lineage.conversation_id == live_conversation_id
        and lineage.client_session_id == client_session_id
        and lineage.system_digest == system_digest
        and len(keys) > len(lineage.message_keys)
        and keys[: len(lineage.message_keys)] == lineage.message_keys
    )
    if continues:
        assert lineage is not None
        new = dialogue[len(lineage.message_keys) :]
        if all(message.get("role") in ("user", "tool") for message in new):
            return DeepSeekTurnPlan(
                prompt=_with_style(_new_messages_text(new), persona_instruction),
                new_conversation=False,
                system_digest=system_digest,
                message_keys=keys,
            )

    sections: list[str] = []
    if system:
        sections.append(
            "SYSTEM INSTRUCTIONS FROM THE CLIENT APPLICATION "
            "(follow them for the whole conversation):\n" + system
        )
    earlier = _transcript(dialogue[:-1])
    if earlier:
        sections.append("EARLIER CONVERSATION:\n" + earlier)
    current = _message_text(dialogue[-1])
    if sections:
        sections.append(
            "CURRENT MESSAGE:\n"
            + (current if dialogue[-1].get("role") == "user" else _render(dialogue[-1]))
        )
        prompt = "\n\n---\n\n".join(sections)
    else:
        prompt = current
    return DeepSeekTurnPlan(
        prompt=_with_style(prompt, persona_instruction),
        new_conversation=True,
        system_digest=system_digest,
        message_keys=keys,
    )


__all__ = [
    "MAX_TRANSCRIPT_CHARS",
    "DeepSeekLineage",
    "DeepSeekTurnPlan",
    "plan_turn",
]
