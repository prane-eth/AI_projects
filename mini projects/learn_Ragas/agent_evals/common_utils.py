import os
from dotenv import load_dotenv
load_dotenv()

model = os.getenv("OPENAI_MODEL", "")
if not model:
	raise ValueError("OPENAI_MODEL environment variable is not set.")
