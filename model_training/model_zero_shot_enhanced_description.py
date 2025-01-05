import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from colorama import Fore
import colorama
from tqdm import tqdm
import json
from utils import change_input_instruction, get_input_gemma_template, open_json_file

colorama.init(autoreset=True)

dev_data_path = "../data_processing/data/formated_dev_data_enhanced.json"
auth_token = "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"
base_model = "google/gemma-2-2b-it"
input_instruction = "Convert the following question into an SQL query using the provided database schema. The output should contain only the SQL statement."


print("Loading model...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(base_model,
                                          use_fast=True,
                                          token=auth_token,
                                          dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(base_model,
                                             device_map="cuda",
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=bnb_config,
                                             token=auth_token)
print(f"{Fore.GREEN}Model loaded successfully !\n")


print("Starting inference...")
change_input_instruction(dev_data_path, input_instruction)
data = open_json_file(dev_data_path)

output_data = []

for format_data in tqdm(data):
    input = get_input_gemma_template(format_data)
    token_inputs = tokenizer.apply_chat_template(input,
                                                tokenize=True,
                                                token=auth_token,
                                                return_tensors="pt",
                                                use_fast=True,
                                                add_generation_prompt=False).to(model.device)
    token_outputs = model.generate(input_ids=token_inputs,
                                   do_sample=True,
                                   max_new_tokens=256,
                                   temperature=.1).to(model.device)
    new_tokens = token_outputs[0][token_inputs.shape[-1]:]
    answer = tokenizer.decode(new_tokens,
                              skip_special_tokens=True)
    format_data["output"] = answer
    output_data.append(format_data)
    #print(f"{Fore.YELLOW}Input:", {format_data['input']})
    #print(answer)

print(f"{Fore.GREEN}Inference completed !\n")

print("Saving results...\n")
with open("outputs/zero_shot_enhanced_results.json", "w") as f:
    json.dump(output_data, f)
print(f"{Fore.GREEN}Results saved successfully !\n")




