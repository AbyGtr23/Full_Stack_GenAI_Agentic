from groq import Groq
from dotenv import load_dotenv
import requests


load_dotenv()  # Load environment variables from .env file
client = Groq()  # Initialize the Groq client


def get_weather_info(city: str):
    url = f"https://wttr.in/{city.lower()}?format=3"
    response  = requests.get(url)
    if response.status_code == 200:
        return f"Weather information for {city}: {response.text}"
    else:
        return f"Failed to retrieve weather information for {city}. Please try again later."





def main():
    try:
        user_query = input("Enter you query: ")
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

# print(get_weather_info("Sambalpur"))