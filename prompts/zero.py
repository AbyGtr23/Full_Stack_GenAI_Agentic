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

# Marks this script as a demo of Zero Shot prompting (instruction only, no examples).
#Zero Shot prompting

# System prompt that restricts the model to science topics with no examples
# provided (zero-shot), relying purely on the instruction itself.
SYSTEM_PROMPT = "You are a scientist and you should answer only science related questions. If the question is not related to science, you should say 'Sorry ! I am scientist and I don't have any other knowledge except Science'"

# Send a chat completion request using the zero-shot system prompt and an
# off-topic user question (general knowledge, not science) to test the restriction.
response = client.chat.completions.create(
    # Target model name.
    model="gemini-3.8-flash",
    # Request a single completion choice.
    n=1,
    # Conversation messages: the zero-shot system prompt plus the user's question.
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Please tell me who is the president of USA"
        }
    ]
)

# Print the model's reply text to the console.
print(response.choices[0].message.content)