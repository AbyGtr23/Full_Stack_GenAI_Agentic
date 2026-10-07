from groq import Groq
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file
client = Groq()  # Initialize the Groq client

def main():
    try:
        user_query = input("Enter your question: ")
        response = client.chat.completions.create(
            model = "openai/gpt-oss-20b",
            messages = [
                {"role": "user", "content": user_query}
            ]
        )

        print("Response from the model:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")
main()