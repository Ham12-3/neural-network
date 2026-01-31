import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MODEL_NAME = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-12-6")

MAX_INPUT_CHARS = 10_000
MIN_MAX_WORDS = 30
MAX_MAX_WORDS = 200
DEFAULT_MAX_WORDS = 80
