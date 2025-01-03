import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import Fore
import colorama
from tqdm import tqdm

colorama.init(autoreset=True)

auth_token = "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"
base_model = "google/gemma-2-2b-it"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model,
                                          use_fast=True,
                                          token=auth_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(base_model,
                                             device_map="cuda",
                                             torch_dtype=torch.bfloat16,
                                             token=auth_token)
print(f"{Fore.GREEN}Model loaded successfully !\n")


print("Starting inference...")
questions = [{"role": "user", "content": "give me just the number nothing else. 1 + 1 = ?" }]

token_inputs = tokenizer.apply_chat_template(questions,
                                            tokenize=True,
                                            token=auth_token,
                                            return_tensors="pt",
                                            use_fast=True,
                                            add_generation_prompt=True).to(model.device)

token_outputs = model.generate(input_ids=token_inputs,
                               do_sample=True,
                               max_new_tokens=500,
                               temperature=.5)

new_tokens = token_outputs[0][token_inputs.shape[-1]:]
answer = tokenizer.decode(new_tokens,
                          skip_special_tokens=True)
print(f"{Fore.GREEN}Inference completed !\n")

print(answer)


