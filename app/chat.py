from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import requests

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose a model (e.g., Mistral-7B)
model_name = "distilgpt2"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Download and Load the Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)  # Auto-detect GPU/CPU
model.to(device)

router = APIRouter()

# Replace with your actual LLM API endpoint
LLM_API_URL = "https://your-llm-endpoint.com/generate"


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def chat(request_message: ChatRequest):
    try:
        # Send the user message to the LLM
        # response = requests.post(LLM_API_URL, json={"input": request.message})
        # response.raise_for_status()
        # llm_response = response.json().get("output", "Sorry, I couldn't process that.")
        #
        # return {"response": llm_response}
        input_text = request_message.message
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=20, pad_token_id=model.config.eos_token_id)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"response": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
