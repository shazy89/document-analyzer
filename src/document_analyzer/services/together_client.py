from __future__ import annotations

from collections.abc import Sequence
import os
from typing import TYPE_CHECKING, Protocol, cast

from document_analyzer.core.config import Settings
from document_analyzer.models.chat import ChatMessage, ChatResponse, ChatRole

if TYPE_CHECKING:
    from together import Together


class MissingTogetherAPIKeyError(RuntimeError):
    """Raised when a Together request is made without an API key."""


class _TogetherMessage(Protocol):
    content: object


class _TogetherChoice(Protocol):
    message: _TogetherMessage


class _TogetherResponse(Protocol):
    choices: Sequence[_TogetherChoice]


class TogetherChatService:
    def __init__(self, *, api_key: str | None, default_model: str) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._client: Together | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "TogetherChatService":
        api_key = None
        if settings.together_api_key is not None:
            api_key = settings.together_api_key.get_secret_value()
        return cls(api_key=api_key, default_model=settings.together_model)

    def ask(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        if not self._api_key:
            raise MissingTogetherAPIKeyError("TOGETHER_API_KEY is not set.")

        if self._client is None:
            self._client = self._build_client()

        selected_model = model or self._default_model
        messages = self._build_messages(prompt=prompt, system_prompt=system_prompt)
        response = cast(
            _TogetherResponse,
            self._client.chat.completions.create(
                model=selected_model,
                messages=[message.model_dump() for message in messages],
            ),
        )
        answer = self._extract_answer(response)
        return ChatResponse(model=selected_model, answer=answer)

    def _build_client(self) -> Together:
        os.environ.setdefault("TOGETHER_NO_BANNER", "1")

        from together import Together

        return Together(api_key=self._api_key)

    def _build_messages(self, *, prompt: str, system_prompt: str | None) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role=ChatRole.SYSTEM, content=system_prompt))
        messages.append(ChatMessage(role=ChatRole.USER, content=prompt))
        return messages

    def _extract_answer(self, response: _TogetherResponse) -> str:
        if not response.choices:
            return ""

        content = response.choices[0].message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence):
            parts = [self._stringify_part(part) for part in content]
            return "".join(part for part in parts if part)
        return str(content)

    def _stringify_part(self, part: object) -> str:
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            text = part.get("text")
            return text if isinstance(text, str) else ""
        return str(part)
