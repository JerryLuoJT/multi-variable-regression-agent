import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

MODEL_NAME = "deepseek-v4-pro"
