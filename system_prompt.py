# Standard library import used to read environment variables (e.g. API keys).
import os
# Loads key=value pairs from a .env file into the process environment.
from dotenv import load_dotenv
# OpenAI SDK client, reused here to call Gemini's OpenAI-compatible endpoint.
from openai import OpenAI

# Load environment variables from a .env file (if present) into os.environ.
load_dotenv()
# Fetch the Gemini API key from the environment (Gemini exposes an OpenAI-compatible API).
api_key = os.getenv("GEMINI_API_KEY")
# Fail fast with a clear message if no API key was configured.
if not api_key:
    raise RuntimeError(
        "Set GEMINI_API_KEY in your environment or in a .env file before running this script."
    )


# Create an OpenAI client pointed at Google's Gemini OpenAI-compatible base URL.
client = OpenAI(
    # Authenticate using the Gemini API key.
    api_key= api_key,
    # Gemini's endpoint that mimics the OpenAI chat completions API.
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)



# Send a chat completion request that uses a restrictive system prompt to
# constrain the model to only answering Python-related questions.
response = client.chat.completions.create(
    # Target model name.
    model="gemini-3.8-flash",
    # Request a single completion choice.
    n=1,
    # Conversation messages: a restrictive system instruction followed by an
    # off-topic user question (about C++) to test that the restriction works.
    messages=[
        {"role": "system", "content": "You are a python expert and you are only supposed to answer questions related to python programming. Any other question about any other tool and you should say 'Sorry ! I am unable to answer what you are asking.'"},
        {
            "role": "user",
            "content": "Can yo help me write a program in C++"
        }
    ]
)

# Print the model's reply text to the console.
print(response.choices[0].message.content)