from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


valid_action = []
test_prompt = ""


tokenizer = AutoTokenizer.from_pretrained(model, token = HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model, token = HF_TOKEN)

# construct the initial input text 
messages = "\n".join(msg['content'] for msg in messages)
inputs = tokenizer(messages, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]

# Get constrained logits
constrained_logits = logits.clone().fill_(-float('inf'))  # set all the logits to -inf
# Recover the logits for the valid action
idx = [tokenizer.convert_tokens_to_ids(action) for action in current_valid_actions if tokenizer.convert_tokens_to_ids(action) is not None]
mask = torch.zeros_like(logits, dtype=torch.bool)
mask[:, idx] = True
constrained_logits[mask] = logits[mask]

# sample from the constrained logits
probs = torch.nn.functional.softmax(constrained_logits, dim=-1)

# test if probs containing inf/nan/element<0
if torch.any(torch.isinf(probs)) or torch.any(torch.isnan(probs)) or torch.any(probs < 0):
    print("warning: probs contains inf/nan/element < 0")

    inf_nan_indices = torch.nonzero(torch.isinf(probs) | torch.isnan(probs) | (probs < 0), as_tuple=False)

    problematic_values = logits[inf_nan_indices[:, 0], inf_nan_indices[:, 1]]
    
    print("problematic_values", problematic_values)
    
next_token_id = torch.multinomial(probs / temperature, num_samples=1)

generated_text = tokenizer.decode(next_token_id.item(), skip_special_tokens=True)

