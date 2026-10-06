# Standard library import used to read environment variables (e.g. API keys).
import os
# Loads key=value pairs from a .env file into the process environment.
from dotenv import load_dotenv
# OpenAI SDK client, reused here to call Groq's OpenAI-compatible endpoint.
from openai import OpenAI

# Load environment variables from a .env file (if present) into os.environ.
load_dotenv()
# Fetch the Groq API key from the environment.
api_key = os.getenv("GROQ_API_KEY")
# Fail fast with a clear message if no API key was configured.
if not api_key:
    raise RuntimeError(
        "Set GROQ_API_KEY in your environment or in a .env file before running this script."
    )

# Create an OpenAI client pointed at Groq's OpenAI-compatible base URL.
client = OpenAI(
    # Authenticate using the Groq API key.
    api_key=api_key,
    # Groq's endpoint that mimics the OpenAI chat completions API.
    base_url="https://api.groq.com/openai/v1"
)

# System prompt instructing the model to reason step-by-step using a
# chain-of-thought (CoT) START -> PLAN -> OUTPUT protocol, responding only
# in a fixed JSON schema. Same prompt as COT.py, but targeting Groq instead
# of Gemini.
SYSTEM_PROMPT = """You are an expert AI assistant that is capable of answering or resolving problems and issues provided by a user, using a chain of thought.
Work on START PLAN OUTPUT steps to solve the problem.
You need to first PLAN what needs to be done. The PLAN could be of multiple steps. Once you think the PLAN is complete and appropriate for the problem statement, proceed to the OUTPUT.

Rules:
1. Strictly follow the given JSON output format.
2. Only perform one step at a time
3. The sequence of the steps is START (the user gives input), PLAN (you plan the steps and adapt an apt approach to solve the problem) and finally OUTPUT (you give the final output to be displayed to the user)

Output JSON format:
{"step": "START|PLAN|OUTPUT",  "content": "your output here in string format"}

Example:

    START: {"step": "START", "content": "Can you help me write a program in C++ to find the factorial of a number?"}
    PLAN: {"step": "PLAN", "content": "The user wants to write a code in C++ to find the factorial of a number."}
    PLAN: {"step": "PLAN", "content": "factorial of a number can be calculated using a recursive function or an iterative approach. I will choose the iterative approach for simplicity."}
    PLAN: {"step": "PLAN", "content": "In order to find the factorial of any number we need to multiply the number provided as input with a preceding number until we reach 1. And then we stop"}
    PLAN: {"step": "PLAN", "content": "For example if the input is 3 we multiply 3*2*1"}
    OUTPUT: {"step": "OUTPUT", "content": "The factorial of 3 is 6"}
"""

# Send a single chat completion request asking the model to compute a
# factorial using the CoT prompt, via Groq's gpt-oss model.
response = client.chat.completions.create(
    # Target model name.
    model="openai/gpt-oss-120b",
    # Force the model to respond with a JSON object matching the schema above.
    response_format={"type": "json_object"},
    # Conversation messages: the CoT system prompt plus the user's question.
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "What is the factorial of 6 ?"
        }
    ]
)

# Print the raw JSON response content returned by the model.
print(response.choices[0].message.content)
