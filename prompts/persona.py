# Loads key=value pairs from a .env file into the process environment.
from dotenv import load_dotenv
# OpenAI SDK client, reused here to call Groq's OpenAI-compatible endpoint.
from openai import OpenAI
# Standard library import used to read environment variables (e.g. API keys).
import os

# Load environment variables from a .env file (if present) into os.environ.
load_dotenv()
# Fetch the Groq API key from the environment.
api_key = os.getenv("GROQ_API_KEY")

# Create an OpenAI client pointed at Groq's OpenAI-compatible base URL.
client = OpenAI(
    # Authenticate using the Groq API key.
    api_key= api_key,
    # Groq's endpoint that mimics the OpenAI chat completions API.
    base_url="https://api.groq.com/openai/v1"
)

# System prompt that gives the model a named persona ("Abhay") with a
# friendly tone, plus one example exchange to demonstrate the desired style.
SYSTEM_PROMPT = """You are an AI persona Assistant named Abhay
You are a friendly and helpful peron who is currently learning GenAI

Examples:
Q. Hello
A. Hi there my dost.

"""
# Note on few-shot persona examples: more examples (or full sample
# conversations) can be added here to teach the model your talking style and tone.
#we can provide as many examples as we want and could also give entire conversations of ourself to make the model learn from our talking style and tone.

# Send a chat completion request with the persona system prompt and two
# user messages (both without an intervening assistant reply).
response = client.chat.completions.create(
    # Target model name.
    model="openai/gpt-oss-120b",
    # Conversation messages: persona system prompt plus two user turns.
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello, who are you ?"},
        {"role": "user", "content": "what all have you leart so far."}
    ]
)

# Print a labeled version of the model's reply text to the console.
print("Response:", response.choices[0].message.content)