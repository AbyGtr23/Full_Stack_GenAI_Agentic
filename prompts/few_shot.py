# Marks this script as a demo of Few Shot prompting.
#Few Shot prompting

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

# Explanation of what few-shot prompting is and why it's used here.
#Few Shot prompting is giving a direct instruction to the model along with a few examples of the task you want the model to perform. This helps the model understand the context and generate more accurate responses.

# System prompt that restricts the model to science topics and provides two
# worked examples (a refusal example and a correct-answer example) so the
# model learns the expected answer style and scope via few-shot examples.
SYSTEM_PROMPT = """You are a scientist and you should answer only science related questions. If the question is not related to science, you should say sorry

Examples:
Q. Who is the president of India?
A. I am not an expert in general Knowledge.

Q. What is the molecular orbital theory ?
A. The molecular orbital theory is a method for describing the electronic structure of molecules using quantum mechanics. It explains how atomic orbitals combine to form molecular orbitals, which can be occupied by electrons. This theory helps predict the bonding, stability, and properties of molecules.
"""

# Send a chat completion request using the few-shot system prompt and a
# science-related user question.
response = client.chat.completions.create(
    # Target model name.
    model="gemini-3.8-flash",
    # Request a single completion choice.
    n=1,
    # Conversation messages: the few-shot system prompt plus the user's question.
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "What is the molecular orbital theory ?"
        }
    ]
)

# Print the model's reply text to the console.
print(response.choices[0].message.content)