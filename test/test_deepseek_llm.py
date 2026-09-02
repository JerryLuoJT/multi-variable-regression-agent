import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from core.deepseek_llm import DEEPSEEK_CHAT_COMPLETIONS_URL, DeepSeekChatModel


def test_deepseek_adapter_normalizes_and_replays_tool_calls(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "id": "chatcmpl-test",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "",
                            "reasoning_content": "internal reasoning",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "test_vif",
                                        "arguments": '{"candidate_id": 1}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

    class FakeClient:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def post(url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = json
            return FakeResponse()

    monkeypatch.setattr("core.deepseek_llm.httpx.Client", FakeClient)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_vif",
                "description": "Calculate VIF.",
                "parameters": {
                    "type": "object",
                    "properties": {"candidate_id": {"type": "integer"}},
                    "required": ["candidate_id"],
                },
            },
        }
    ]
    model = DeepSeekChatModel(
        model="deepseek-v4-pro",
        api_key="test-key",
        thinking="enabled",
        reasoning_effort="high",
        timeout_seconds=45,
    ).bind_tools(tools)

    response = model.invoke(
        [SystemMessage(content="Use one tool."), HumanMessage(content="Check VIF.")]
    )

    assert captured["url"] == DEEPSEEK_CHAT_COMPLETIONS_URL
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["timeout"] == 45
    assert captured["body"]["model"] == "deepseek-v4-pro"
    assert captured["body"]["thinking"] == {"type": "enabled"}
    assert captured["body"]["reasoning_effort"] == "high"
    assert captured["body"]["tools"] == tools
    assert "temperature" not in captured["body"]
    assert response.tool_calls == [
        {
            "name": "test_vif",
            "args": {"candidate_id": 1},
            "id": "call_123",
            "type": "tool_call",
        }
    ]
    assert response.additional_kwargs["reasoning_content"] == "internal reasoning"

    assistant_payload = model._message_payload(response)
    assert assistant_payload["reasoning_content"] == "internal reasoning"
    assert json.loads(assistant_payload["tool_calls"][0]["function"]["arguments"]) == {
        "candidate_id": 1
    }
    tool_payload = model._message_payload(
        ToolMessage(content='{"ok": true}', tool_call_id="call_123")
    )
    assert tool_payload == {
        "role": "tool",
        "content": '{"ok": true}',
        "tool_call_id": "call_123",
    }


def test_deepseek_adapter_requires_api_key():
    model = DeepSeekChatModel(model="deepseek-v4-pro", api_key=None)

    try:
        model.invoke("hello")
    except RuntimeError as exc:
        assert "DEEPSEEK_API_KEY" in str(exc)
    else:
        raise AssertionError("missing API key should fail before a network request")
