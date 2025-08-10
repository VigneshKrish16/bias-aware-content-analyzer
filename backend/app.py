from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import requests
import json

# Initialize FastAPI app
app = FastAPI(title="🧠 DeepSeek Code Companion with Mistral")

# Configuration
MODELS = {
    "deepseek-coder:1.3b": {
        "url": "http://localhost:11434",
        "description": "Smaller model for faster responses"
    },
    "deepseek-coder:6.7b": {
        "url": "http://localhost:11434",
        "description": "Larger model for more accurate responses"
    },
    "mistral": {
        "url": "http://localhost:11434/api/generate",
        "description": "General purpose model for diverse responses"
    }
}

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

class Message(BaseModel):
    text: str
    model: str = "deepseek-coder:6.7b"
    conversation_id: Optional[str] = None

def generate_with_mistral(prompt: str) -> str:
    """Generate response using Ollama Mistral"""
    try:
        response = requests.post(
            MODELS["mistral"]["url"],
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            }
        )
        response.raise_for_status()
        return response.json().get("response", "I couldn't generate a response.")
    except Exception as e:
        print(f"Error calling Ollama: {str(e)}")
        return "I encountered an error generating a response."

def generate_with_deepseek(prompt_chain):
    """Generate response using DeepSeek Coder models"""
    processing_pipeline = prompt_chain | ChatOllama(
        model=prompt_chain.model,
        base_url=MODELS[prompt_chain.model]["url"],
        temperature=0.3
    ) | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain(message_log, model):
    """Build the prompt chain from message history"""
    prompt_sequence = [system_prompt]
    for msg in message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    
    # Create the chain with model info
    chain = ChatPromptTemplate.from_messages(prompt_sequence)
    chain.model = model
    return chain

@app.post("/chat")
async def chat(message: Message):
    try:
        # Initialize or continue conversation
        if not hasattr(app, "conversations"):
            app.conversations = {}
        
        conv_id = message.conversation_id or "default"
        if conv_id not in app.conversations:
            app.conversations[conv_id] = {
                "message_log": [{"role": "ai", "content": "Hi! I'm your coding assistant. How can I help you today? 💻"}],
                "model": message.model
            }
        
        # Add user message to log
        app.conversations[conv_id]["message_log"].append({"role": "user", "content": message.text})
        
        # Generate response based on selected model
        if message.model == "mistral":
            # For Mistral, we use the direct API call
            ai_response = generate_with_mistral(message.text)
        else:
            # For DeepSeek models, we use LangChain
            prompt_chain = build_prompt_chain(
                app.conversations[conv_id]["message_log"],
                message.model
            )
            ai_response = generate_with_deepseek(prompt_chain)
        
        # Add AI response to log
        app.conversations[conv_id]["message_log"].append({"role": "ai", "content": ai_response})
        
        return {
            "response": ai_response,
            "model_used": message.model,
            "conversation_id": conv_id,
            "message_history": app.conversations[conv_id]["message_log"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models with descriptions"""
    return {
        "available_models": [
            {
                "name": name,
                "description": info["description"],
                "type": "coding" if "coder" in name else "general"
            }
            for name, info in MODELS.items()
        ]
    }

@app.get("/conversations")
async def list_conversations():
    """List active conversations"""
    if not hasattr(app, "conversations"):
        return {"active_conversations": []}
    return {
        "active_conversations": [
            {
                "id": conv_id,
                "message_count": len(conv_data["message_log"]),
                "current_model": conv_data["model"]
            }
            for conv_id, conv_data in app.conversations.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)