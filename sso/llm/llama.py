from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
import os
from functools import lru_cache 
import json

# HF_TOKEN = os.environ["HF_TOKEN"]
HF_TOKEN = "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"


def softmax(x):
    exps = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    softmax = exps / torch.sum(exps, dim = -1, keepdim = True)
    return softmax


def get_response(model: str, messages, valid_action_list, temperature = 1.0, **kwargs) -> str:
    
    tokenizer = AutoTokenizer.from_pretrained(model, token = HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model, token = HF_TOKEN)
    
    # construct the initial input text 
    messages = "\n".join(msg['content'] for msg in messages)
    print("The current prompt is")
    print(messages)

    # get max length of the instructions in valid_action_list
    max_action_length = max(len(action.split()) for action in valid_action_list)


    print("start generating")
    model.eval()
    final_output = ""
    iter_count = 0

    print("max_action_length", max_action_length)
    action_flag = [True for action in valid_action_list]

    for iter_count in range(max_action_length):
   
        inputs = tokenizer(messages, return_tensors="pt")
   
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]


        print("iter", iter_count)

        # update current_valid_actions
        current_valid_actions = set()  
        for i in range(len(valid_action_list)):
            word_list = valid_action_list[i].split()
            if iter_count < len(word_list) and action_flag[i]:
                word = word_list[iter_count].strip()
                current_valid_actions.add(word)  
        
        current_valid_actions = list(current_valid_actions)
        if len(current_valid_actions) == 0:
            return final_output
            
        print("current_valid_actions[:10]",current_valid_actions[:10])
        
        # Get constrained logits
        constrained_logits = logits.clone().fill_(-float('inf'))  # set all the logits to -inf
        # Recover the logits for the valid action
        idx = [tokenizer.convert_tokens_to_ids(action) for action in current_valid_actions if tokenizer.convert_tokens_to_ids(action) is not None]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, idx] = True
        constrained_logits[mask] = logits[mask]
        
        
        generated_text = tokenizer.eos_token

        token_probabilities = []

        while True:
            # sample from the constrained logits
            probs = torch.nn.functional.softmax(constrained_logits, dim=-1)
            
            # test if probs containing inf/nan/element<0
            if torch.any(torch.isinf(probs)) or torch.any(torch.isnan(probs)) or torch.any(probs < 0):
                print("warning: probs contains inf/nan/element < 0")
                inf_nan_indices = torch.nonzero(torch.isinf(probs) | torch.isnan(probs) | (probs < 0), as_tuple=False)
                problematic_values = logits[inf_nan_indices[:, 0], inf_nan_indices[:, 1]]
                # 打印出probs是负数的token是什么
                print("problematic_values", problematic_values)
                
            next_token_id = torch.multinomial(probs / temperature, num_samples=1)

            token_probability = probs[next_token_id.item()].item()
            token_probabilities.append(token_probability)

            generated_text = tokenizer.decode(next_token_id.item(), skip_special_tokens=True) # 试用generate代替decode
            print("generated_text:", repr(generated_text))
            print("check if generated_text in current_valid_actions:", generated_text in current_valid_actions)
            
            if generated_text in current_valid_actions:
                for i in range(len(valid_action_list)):
                    word_list = valid_action_list[i].split()
                    if iter_count < len(word_list) and word_list[iter_count] != generated_text:
                        action_flag[i] = False
                break

        # update final_output
        final_output += generated_text + " "

        print("the generated text is ", generated_text)

        # calculate the average probability of the generated_text
        print("the probability of each token in the generated text is ", token_probabilities)
        average_probs = sum(token_probabilities) / len(token_probabilities)
        print("the average probability is", average_probs)
    
        # update input_ids
        messages += " " + generated_text

        iter_count += 1

        if generated_text == tokenizer.eos_token:
            print("break")
            break
            
    
    return final_output



@lru_cache(maxsize=1000)
def get_embedding(model: str, content: str, max_tries=50) -> List[float]:
    
    tokenizer = AutoTokenizer.from_pretrained(model, token = HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model, token = HF_TOKEN)

    inputs = tokenizer(content, return_tensors='pt')
    # input_ids

    outputs = model(**inputs, output_hidden_states=True)
    # hugggingface 
    # model.hidden_states
    # (batch_size, seq_len, embedd_dim) 

    last_hidden_state = outputs.hidden_states[-1] 
    
    return last_hidden_state
    

