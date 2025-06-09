#The api key is stored in a .env file but if it doesnt load then hardcode the api key here
#(
# ⚠️ WARNING: Do not hardcode API keys in production code
#GEMINI_API_KEY = "your_gemini_api_key_here"

#if not GEMINI_API_KEY or GEMINI_API_KEY.strip().lower() == "your_gemini_api_key_here":
    #raise ValueError("❌ API key missing or invalid.")
#)

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key.strip().lower() == "your_gemini_api_key_here":
    raise ValueError("❌ API key missing or invalid. Check your .env or config.")