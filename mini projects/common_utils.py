import os
from dotenv import load_dotenv
load_dotenv()

model = os.getenv("OPENAI_MODEL", "")
if not model:
	raise ValueError("OPENAI_MODEL environment variable is not set.")

base_url = os.getenv("OPENAI_BASE_URL", "")
if not base_url:
	raise ValueError("OPENAI_BASE_URL environment variable is not set.")

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
	raise ValueError("OPENAI_API_KEY environment variable is not set.")
