from langchain_google_genai import ChatGoogleGenerativeAI
from configs.agent_config import LLM_TEMPERATURE
from configs.setting import API_KEY, MODEL_NAME

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        api_key=API_KEY)
    return llm

if __name__ == "__main__":
    print("正在呼叫 Google Gemini...")
    model = get_llm()
    response = model.invoke("你好，请用一句话介绍一下什么是多元线性回归？")
    print(f"\n🤖 Gemini 的回答: \n{response.content}")
