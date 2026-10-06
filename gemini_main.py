# Standard library import used to read environment variables (e.g. API keys).
import os
# Standard library import used to pause execution between retry attempts.
import time

# Loads key=value pairs from a .env file into the process environment.
from dotenv import load_dotenv
# Google's Gemini SDK client used to talk to the Gemini API.
from google import genai
# Error types raised by the Gemini SDK (used to catch server-side failures).
from google.genai import errors


# Helper that decides whether an exception represents a transient "model busy" error.
def is_model_busy_error(exc):
    # Returns True if the error text mentions HTTP 503 or an "UNAVAILABLE" status,
    # both of which indicate the model is temporarily overloaded rather than broken.
    return "503" in str(exc) or "UNAVAILABLE" in str(exc)


# Load environment variables from a .env file (if present) into os.environ.
load_dotenv()

# Fetch the Gemini API key from the environment.
api_key = os.getenv("GEMINI_API_KEY")
# Fail fast with a clear message if no API key was configured.
if not api_key:
    raise RuntimeError(
        "Set GEMINI_API_KEY in your environment or in a .env file before running this script."
    )

# Create the Gemini API client authenticated with the API key.
client = genai.Client(api_key=api_key)
# The prompt/message that will be sent to the model.
prompt = "Hi, I am Abhay. How can you assist me today?"

# Ordered list of models to attempt, with duplicates removed later via dict.fromkeys.
models_to_try = [
    # Preferred model, overridable via the GEMINI_MODEL environment variable.
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    # First fallback model.
    "gemini-2.5-flash",
    # Second fallback model.
    "gemini-2.0-flash",
]

# Tracks the most recent error so it can be reported if every model fails.
last_error = None
# Iterate over the unique models in models_to_try (dict.fromkeys preserves order and dedupes).
for model in dict.fromkeys(models_to_try):
    # Start a new chat session with the current candidate model.
    chat = client.chats.create(model=model)

    # Retry up to 3 times per model before giving up on it.
    for attempt in range(3):
        try:
            # Send the prompt to the model and get its response.
            response = chat.send_message(prompt)
            # Print the model's text reply to the console.
            print(response.text)
            # Exit the script successfully once we get a response.
            raise SystemExit(0)
        except errors.ServerError as exc:
            # Remember this error in case all models ultimately fail.
            last_error = exc
            # If this isn't a transient "busy" error, re-raise immediately.
            if not is_model_busy_error(exc):
                raise

            # Exponential backoff: wait longer after each failed attempt.
            wait_seconds = 2 ** attempt
            # Inform the user that we're retrying after a delay.
            print(
                f"{model} is busy right now. Retrying in {wait_seconds} seconds..."
            )
            # Pause before retrying the same model.
            time.sleep(wait_seconds)

    # All retry attempts for this model were exhausted; move on to the next model.
    print(f"Switching away from {model} because it is still unavailable.")

# If every model in the list failed, raise an error chained to the last exception seen.
raise RuntimeError("All configured Gemini models are unavailable right now.") from last_error
