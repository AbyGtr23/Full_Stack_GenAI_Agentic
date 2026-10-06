# Earlier attempt kept for reference: using the official OpenAI SDK directly
# against OpenAI's own API (as opposed to Gemini's OpenAI-compatible endpoint below).
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()
# client = OpenAI()

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages= [{"role": "user", "content": "Hello, how are you?"}]
# )

# print(response.choices[0].message.content)

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



# Send a chat completion request to the Gemini model via the OpenAI-compatible API.
response = client.chat.completions.create(
    # Target model name.
    model="gemini-3.8-flash",
    # Request a single completion choice.
    n=1,
    # Conversation messages: a system instruction followed by the user's question.
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Explain to me briefly how AI works. Tell me who are you"
        }
    ]
)

# Print the model's reply text to the console.
print(response.choices[0].message.content)