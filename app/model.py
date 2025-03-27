from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Choose a model (e.g., Mistral-7B)
model_name = "distilgpt2"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Download and Load the Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)  # Auto-detect GPU/CPU
model.to(device)

input_text = "What is the meaning of life?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

output = model.generate(**inputs, max_length=10, pad_token_id=model.config.eos_token_id)

print(tokenizer.decode(output[0], skip_special_tokens=True))



