# Loads key=value pairs from a .env file into the process environment.
from dotenv import load_dotenv
# OpenAI SDK client used to call the chat completions API.
from openai import OpenAI

# Load environment variables from a .env file (if present) into os.environ.
load_dotenv()
# Create an OpenAI client; the API key is picked up automatically from the
# OPENAI_API_KEY environment variable.
client = OpenAI()

# Send a simple chat completion request with a single user message.
response = client.chat.completions.create(
    # Target model name.
    model="gpt-4o-mini",
    # Conversation messages: just one user greeting.
    messages= [{"role": "user", "content": "Hello, how are you?"}]
)

# Print the model's reply text to the console.
print(response.choices[0].message.content)