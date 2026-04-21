import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")

GEMINI_PRO      = "models/gemini-2.5-pro"
GEMINI_FLASH    = "models/gemini-2.5-flash"
GROQ_MODEL      = "llama-3.1-8b-instant"