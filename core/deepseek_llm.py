"""Minimal DeepSeek Chat Completions adapter for the ReAct graph.

The graph only needs ``bind_tools`` and ``invoke``. Keeping that small surface
area avoids coupling the regression workflow to a provider-specific LangChain
package while still returning normal LangChain ``AIMessage`` objects.
"""

from __future__ import annotations

import json
from copy import copy
from typing import Any, Iterable

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekChatModel:
    """Small synchronous DeepSeek client compatible with this project's graph."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        temperature: float = 0.1,
        thinking: str = "enabled",
        reasoning_effort: str = "high",
        timeout_seconds: float = 120.0,
    ):
        if thinking not in {"enabled", "disabled"}:
            raise ValueError("thinking must be 'enabled' or 'disabled'")
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort must be 'low', 'high', or 'max'")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self._tools: list[dict[str, Any]] = []

    def bind_tools(self, tools: Iterable[dict[str, Any]]):
        """Return a shallow copy carrying the OpenAI-format tool schemas."""
        bound = copy(self)
        bound._tools = list(tools)
        return bound

    def invoke(self, messages: str | Iterable[BaseMessage]) -> AIMessage:
        """Send one non-streaming turn and normalize tool calls for LangGraph."""
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not configured. Add it to the project .env file."
            )
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message_payload(message) for message in messages],
            "stream": False,
            "thinking": {"type": self.thinking},
            "reasoning_effort": self.reasoning_effort,
        }
        if self._tools:
            body["tools"] = self._tools
        # DeepSeek ignores sampling controls in thinking mode, so only send this
        # setting when non-thinking mode is explicitly selected.
        if self.thinking == "disabled":
            body["temperature"] = self.temperature

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    DEEPSEEK_CHAT_COMPLETIONS_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1000]
            raise RuntimeError(
                f"DeepSeek API returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("DeepSeek API returned an invalid chat response.") from exc

        normalized_calls = []
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"DeepSeek returned invalid JSON arguments for tool "
                    f"'{function.get('name', '')}'."
                ) from exc
            if not isinstance(arguments, dict):
                raise RuntimeError("DeepSeek tool arguments must decode to a JSON object.")
            normalized_calls.append(
                {
                    "name": function.get("name", ""),
                    "args": arguments,
                    "id": tool_call.get("id", ""),
                    "type": "tool_call",
                }
            )

        additional_kwargs = {}
        # Thinking-mode tool loops must send this value back on later requests.
        # It remains internal and is not included in the CLI's final summary.
        if "reasoning_content" in message:
            additional_kwargs["reasoning_content"] = message["reasoning_content"]

        response_metadata = {
            "model": payload.get("model", self.model),
            "finish_reason": choice.get("finish_reason"),
            "response_id": payload.get("id"),
        }
        ai_message_kwargs: dict[str, Any] = {
            "content": message.get("content") or "",
            "tool_calls": normalized_calls,
            "additional_kwargs": additional_kwargs,
            "response_metadata": response_metadata,
        }
        usage = payload.get("usage") or {}
        if all(
            isinstance(usage.get(key), int)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            ai_message_kwargs["usage_metadata"] = {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }
        return AIMessage(**ai_message_kwargs)

    @staticmethod
    def _message_payload(message: BaseMessage) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id,
            }
        if isinstance(message, AIMessage):
            payload: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            reasoning_content = message.additional_kwargs.get("reasoning_content")
            if reasoning_content is not None:
                payload["reasoning_content"] = reasoning_content
            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("args", {}), ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            return payload
        raise TypeError(f"Unsupported message type: {type(message).__name__}")
