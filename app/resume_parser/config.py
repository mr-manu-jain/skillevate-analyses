# config.py
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_MODEL = "phi3:mini"
OLLAMA_BASE_URL = "http://localhost:11434"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
# Extraction quality thresholds
MIN_LINES_FOR_GOOD_EXTRACTION = 10
MIN_SECTIONS_FOR_GOOD_EXTRACTION = 2

# DPI for Gemini image fallback
PDF_IMAGE_DPI = 150