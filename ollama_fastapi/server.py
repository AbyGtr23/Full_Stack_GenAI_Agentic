from fastapi import FastAPI, Body
from ollama import Client


app = FastAPI()
client = Client(
    host="http://localhost:11434"
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/contact")
def my_contact():
    return {"email": "abhay.pattnaik@elixgen.com", "Mob": "637XXXXX19"}

@app.post("/chat") #New 'chat' route created to send messages to the model and receive responses
def chat(message: str = Body(..., description="The message to send to the model")):
    response =client.chat(model="llama3.2:latest", messages=[
        {"role": "user", "content": message}
    ])
    return {"response": response.message.content} #Returns the response from the model



