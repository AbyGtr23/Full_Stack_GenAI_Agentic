from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file
client = OpenAI()  # Initialize the OpenAI client

def main():
    user_query = input("Enter your question: ")
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "user", "content": user_query}
        ]
    )

    print("Response from the model:")
    print(response.choices[0].message.content)