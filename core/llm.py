from configs.agent_config import (
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_THINKING_MODE,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)
from configs.setting import API_KEY, MODEL_NAME
from core.deepseek_llm import DeepSeekChatModel


def get_llm():
    return DeepSeekChatModel(
        model=MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        api_key=API_KEY,
        thinking=DEEPSEEK_THINKING_MODE,
        reasoning_effort=DEEPSEEK_REASONING_EFFORT,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
    )


if __name__ == "__main__":
    print(f"正在呼叫 DeepSeek（{MODEL_NAME}）...")
    model = get_llm()
    response = model.invoke("你好，请用一句话介绍一下什么是多元线性回归？")
    print(f"\n🤖 DeepSeek 的回答: \n{response.content}")
