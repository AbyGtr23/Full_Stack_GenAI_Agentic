# Standard library import used to read environment variables (e.g. API keys).
import os
# Loads key=value pairs from a .env file into the process environment.
from dotenv import load_dotenv
# Groq SDK client, used to call Groq's chat completions API.
from groq import Groq
import requests
import json

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



def get_weather_info(city: str):
    url = f"https://wttr.in/{city.lower()}?format=3"
    response  = requests.get(url)
    if response.status_code == 200:
        return f"Weather information for {city}: {response.text}"
    else:
        return f"Failed to retrieve weather information for {city}. Please try again later."

available_tool = {
    "get_weather_info": get_weather_info
}
# System prompt instructing the model to reason step-by-step using a
# chain-of-thought (CoT) START -> PLAN -> OUTPUT protocol, responding only
# in a fixed JSON schema. Based on prompts/COT.py, extended with a TOOL step
# so the model can call get_weather_info.
SYSTEM_PROMPT = """You are an expert AI assistant that is capable of answering or resolving problems and issues provided by a user, using a chain of thought.
Work on START PLAN OUTPUT steps to solve the problem.
You need to first PLAN what needs to be done. The PLAN could be of multiple steps. Once you think the PLAN is complete and appropriate for the problem statement, proceed to the OUTPUT.
You can also call a tool if required from the list of available tools.
For every tool call wait for the observe step, which is the output from the called tool.

Rules:
1. Strictly follow the given JSON output format.
2. Only perform one step at a time
3. The sequence of the steps is START (the user gives input), PLAN (you plan the steps and adapt an apt approach to solve the problem) and finally OUTPUT (you give the final output to be displayed to the user)

Output JSON format:
{"step": "START" | "PLAN" | "OUTPUT" | "TOOL",  "content": "string", "tool": "string", "input": "string"}

Available tools:
1. get_weather_info(city: str): Takes city names as an input string and returns the weather information about the city.

Example 1:

    START: {"step": "START", "content": "Can you help me write a program in C++ to find the factorial of a number?"}
    PLAN: {"step": "PLAN", "content": "The user wants to write a code in C++ to find the factorial of a number."}
    PLAN: {"step": "PLAN", "content": "factorial of a number can be calculated using a recursive function or an iterative approach. I will choose the iterative approach for simplicity."}
    PLAN: {"step": "PLAN", "content": "In order to find the factorial of any number we need to multiply the number provided as input with a preceding number until we reach 1. And then we stop"}
    PLAN: {"step": "PLAN", "content": "For example if the input is 3 we multiply 3*2*1"}
    OUTPUT: {"step": "OUTPUT", "content": "The factorial of 3 is 6"}

Example 2:
    START: Hey can you tell me how is the weather in Sambalpur?
    PLAN: {"step": "PLAN","content": "The user wants to know the weather information for Sambalpur in Odisha."}
    PLAN: {"step": "PLAN","content": "Let me see if there are any available tools to fetch the weather of Sambalpur."}
    PLAN: {"step": "PLAN","content": "Great! We have get_weather_info available for this query."}
    PLAN: {"step": "PLAN","content": "I need to call get_weather_info tool for Sambalpur as city."}
    PLAN: {"step": "TOOL","tool": "get_weather_info", "input": "Sambalpur"}
    PLAN: {"step": "OBSERVE","tool": "get_weather_info", "output": "The weather is humid with 30 degree Celcius in Sambalpur"}
    PLAN: {"step": "PLAN","content": "Great! I got the weather info about Sambalpur."}
    OUTPUT: {"step": "OUTPUT", "content": "The current weather of Sambalpur is humid and warm with a temperature of 30 degree Celcius"}

"""

print("\n\n\n")
message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

user_query = input("Enter here: ")
message_history.append({"role": "user", "content": user_query})

while True:
    response = client.chat.completions.create(
        model = "openai/gpt-oss-20b",
        response_format = {"type":"json_object"},
        messages = message_history

    )

    raw_result = response.choices[0].message.content
    message_history.append({"role": "assistant", "content": raw_result})

    print(raw_result)  

    parsed_result = json.loads(raw_result)

    if parsed_result.get("step") == "START":
        print("LLM Starts", parsed_result.get("content"))
        continue
    if parsed_result.get("step") == "TOOL":
        tool_to_call = parsed_result.get("tool")
        tool_input = parsed_result.get("input")
        print(f"tool_call: {tool_to_call}({tool_input})")

        tool_response = available_tool[tool_to_call](tool_input)
        print(f"tool_call: {tool_to_call}({tool_input}) = {tool_response}")
        message_history.append({"role": "user", "content": json.dumps(
            {"step": "OBSERVE", "tool": tool_to_call, "input": tool_input, "output": tool_response}
        )})
        continue
    # If the model is still planning, log the plan step and continue looping.
    if parsed_result.get("step") == "PLAN":
        print("Thinking", parsed_result.get("content"))
        continue
    # Once the model reaches the OUTPUT step, print the final answer and stop.
    if parsed_result.get("step") == "OUTPUT":
        print("Final Output", parsed_result.get("content"))
        break

# Earlier single-request test (commented out). It asked gpt-oss-120b for the
# factorial of 6; the while loop above replaces it.
# response = client.chat.completions.create(
#     # Target model name.
#     model="openai/gpt-oss-120b",
#     # Force the model to respond with a JSON object matching the schema above.
#     response_format={"type": "json_object"},
#     # Conversation messages: the CoT system prompt plus the user's question.
#     messages=[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {
#             "role": "user",
#             "content": "What is the factorial of 6 ?"
#         }
#     ]
# )

# Print the raw JSON response content returned by the model.
# print(response.choices[0].message.content)
