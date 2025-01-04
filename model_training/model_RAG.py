import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from colorama import Fore
import colorama
from tqdm import tqdm
import json
import faiss
import numpy as np
from utils import change_input_instruction, get_input_gemma_template_RAG, open_json_file

colorama.init(autoreset=True)

# Initialize variables
dev_data_path = "../data_processing/data/formated_dev_data.json"
auth_token = "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"
base_model = "google/gemma-2-2b-it"
input_instruction = "Convert the following question into an SQL query using the provided database schema. The output should contain only the SQL statement."
rag_json_path = "../data_processing/data/RAG_data.json"

# Load model
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
print(f"{Fore.GREEN}Model loaded successfully!\n")


def load_rag_data(rag_json_path):
    with open(rag_json_path, 'r') as f:
        rag_data = json.load(f)
    documents = []
    for db in rag_data:
        for doc in db['documents']:
            documents.append(doc['text'])
    return documents


def initialize_faiss_index(documents, model, tokenizer):
    embeddings = []
    for doc in documents:
        inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.base_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
    return embeddings


def retrieve_relevant_docs(query, index, documents):
    tokenized_query = tokenizer(query, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = tokenized_query['input_ids'].to(model.device)
    with torch.no_grad():
        query_embedding = model.transformer(input_ids).last_hidden_state.mean(dim=1).cpu().numpy()

    D, I = index.search(query_embedding, k=3)
    relevant_docs = [documents[i] for i in I[0]]
    return relevant_docs


# Process data
print("Starting inference...")
change_input_instruction(dev_data_path, input_instruction)
data = open_json_file(dev_data_path)

documents = load_rag_data(rag_json_path)
index = initialize_faiss_index(documents, model, tokenizer)

output_data = []

for format_data in tqdm(data):
    relevant_docs = retrieve_relevant_docs(format_data['input'], index, documents)
    input = get_input_gemma_template_RAG(format_data, relevant_docs)
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
    print(f"{Fore.YELLOW}Input:", {format_data['input']})
    print(answer)

print(f"{Fore.GREEN}Inference completed !\n")

print("Saving results...\n")
with open("outputs/zero_shot_results.json", "w") as f:
    json.dump(output_data, f)
print(f"{Fore.GREEN}Results saved successfully !\n")
