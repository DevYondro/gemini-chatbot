import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

# Initialize FastAPI app
app = FastAPI()

# Verify API key exists
if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY environment variable is missing!")

# Initialize the official Google GenAI Client
client = genai.Client()

# Create a stateful chat session using the 2026 standard model
config = types.GenerateContentConfig(
    system_instruction="You are a helpful and clean AI chatbot assistant.",
    temperature=0.7,
)
chat = client.chats.create(model="gemini-2.5-flash", config=config)

# Define the structure for incoming user messages
class ChatMessage(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "Chatbot is running perfectly online!"}

@app.post("/chat")
def ask_chatbot(data: ChatMessage):
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Send message to the persistent chat session
        response = chat.send_message(data.message)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
