import argparse
import os
import sys
import json
import torch

from transformers import BartTokenizer, BartForConditionalGeneration, BitsAndBytesConfig 

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from finetuning.ft import fine_tune


parser = argparse.ArgumentParser(description="Fine tune a model")

parser.add_argument(
    "-dir", "--data_dir", default="models/basic", type=str, help="data path"
)
parser.add_argument(
    "-c", "--config", default="finetuning/config.json", type=str, help="config path"
)
parser.add_argument(
    "--quantize", default=False, type=bool, help="quantize the model or not for faster exec."
)

args = parser.parse_args()

def main():

    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False, load_in_4bit=True
        )
    else:
        quantization_config = None
        
    name = 'facebook/bart-base' 
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained(
        name, 
        quantization_config=quantization_config, 
        device_map={"": 0}, # load all the model layers on GPU 0
        torch_dtype=torch.bfloat16, 
    )
    device = model.device

    with open(args.config, 'r') as file:
        config = json.load(file)

    print("let's traiiin!!")
        
    fine_tune(model, tokenizer, config, args.data_dir)


if __name__ == "__main__":
    main()
    
    