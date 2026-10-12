# Standard library import used to read environment variables (e.g. API keys).
import os
import sys
# Loads key=value pairs from a .env file into the process environment.
from dotenv import load_dotenv
# Groq SDK client, used to call Groq's chat completions API.
from groq import Groq
import requests
import json

# On Windows, the terminal's default codepage (e.g. cp1252) can't encode
# characters like the weather-icon emoji wttr.in returns (e.g. "✨"), which
# crashes a plain print() with UnicodeEncodeError. Force UTF-8 output and
# fall back to replacing unencodable characters instead of crashing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    # reconfigure() is only available on Python 3.7+ text streams; if it's
    # missing for some reason, just continue with the default stream.
    pass

# Load environment variables from a .env file (if present) into os.environ.
load_dotenv()
# Fetch the Groq API key from the environment.
api_key = os.getenv("GROQ_API_KEY")
# Fail fast with a clear message if no API key was configured.
if not api_key:
    raise RuntimeError(
        "Set GROQ_API_KEY in your environment or in a .env file before running this script."
    )

# Create the Groq client. The SDK already points at Groq's endpoint, so no base_url is needed.
client = Groq(
    # Authenticate using the Groq API key.
    api_key=api_key,
)


def get_weather_info(city: str) -> str:
    """Fetch a one-line weather summary for `city` from wttr.in.

    Always returns a string - on any network error or non-200 response it
    returns a message describing the failure instead of raising, so the
    caller never has to special-case exceptions.
    """
    url = f"https://wttr.in/{city.lower()}?format=3"
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        return f"Failed to retrieve weather information for {city}: {exc}"

    if response.status_code == 200:
        return f"Weather information for {city}: {response.text.strip()}"
    return f"Failed to retrieve weather information for {city}. Please try again later."


# Python-side registry used to dispatch a tool call by name once the model asks for one.
available_tool = {
    "get_weather_info": get_weather_info,
}

# JSON-Schema declaration of the same tool, in the format Groq's API expects.
# This is what replaces agent1.py's hand-rolled "TOOL" step: instead of asking
# the model to *describe* a tool call as JSON text and hoping it follows the
# format, we register the tool natively and let the model use the API's
# built-in tool-calling mechanism.
#
# Why agent1.py's approach (and an early draft of this file) didn't work:
# gpt-oss-20b is trained heavily for agentic tool use. Even when told via the
# system prompt to just describe a "TOOL" step as plain JSON, it kept trying
# to emit a *native* tool call instead. Groq then rejected the request with
#   "Tool choice is none, but model called a tool"
# because no `tools` were declared. Declaring the tool here (and letting the
# model call it for real) works with the model's behaviour instead of
# fighting it, and removes an entire class of prompt-following bugs.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_info",
            "description": "Get the current weather summary for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get weather for, e.g. 'Sambalpur' or 'New Delhi'.",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a helpful assistant that answers questions step by step.
If a question requires current information you don't know (such as the weather), use the available tools to look it up before answering.
Once you have everything you need, give a clear, direct final answer to the user.
"""

MAX_ITERATIONS = 10

print("\n\n\n")
message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

user_query = input("Enter here: ")
message_history.append({"role": "user", "content": user_query})

for _ in range(MAX_ITERATIONS):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=message_history,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message
    # Append the assistant message exactly as returned (it may carry
    # tool_calls) so the conversation history stays valid for the next turn.
    message_history.append(message.model_dump(exclude_none=True))

    tool_calls = message.tool_calls

    if not tool_calls:
        # No further tool calls requested - this is the final answer.
        print("Final Output:", message.content)
        break

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            tool_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            tool_args = {}
        tool_input = tool_args.get("city", "")

        if tool_name not in available_tool:
            print(f"Model requested unknown tool: {tool_name!r}")
            tool_response = f"Error: tool '{tool_name}' does not exist. Available tools: {list(available_tool)}."
        else:
            print(f"tool_call: {tool_name}({tool_input})")
            tool_response = available_tool[tool_name](tool_input)
            print(f"tool_result: {tool_name}({tool_input}) = {tool_response}")

        # Tool results are reported back with role "tool" and must reference
        # the tool_call_id they answer, per Groq/OpenAI's function-calling format.
        message_history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_response,
        })
else:
    print(f"Stopped after {MAX_ITERATIONS} iterations without reaching a final answer.")
