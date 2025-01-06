import json
import re


def open_json_file(file_path, mode='r'):
    with open(file_path, mode) as file:
        data = json.load(file)
    return data

def change_input_instruction(path, instruction):
    data = open_json_file(path)
    for input in data:
        input["instruction"] = instruction

    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def get_input_llama_template_fewshot(data):
    chat_template = (
        "You are an assistant that helps with SQL query generation.\n"
        "Example: {example}\n"
        "Instruction: {instruction}\n"
        "Tables schema: {schema}\n"
        "Question: {input}"
    )
    # Format the string with actual data
    formatted_input = chat_template.format(
        example=data["example"],
        instruction=data["instruction"],
        schema=data["schema"],
        input=data["input"]
    )
    return formatted_input

def get_input_llama_template(data):
    chat_template = (
        "You are an assistant that helps with SQL query generation.\n"
        "Instruction: {instruction}\n"
        "Tables schema: {schema}\n"
        "Question: {input}"
    )
    # Format the string with actual data
    formatted_input = chat_template.format(
        instruction=data["instruction"],
        schema=data["schema"],
        input=data["input"]
    )
    return formatted_input

def get_input_llama_template_RAG(data, RAG):
    # Combine the relevant documents into one string
    RAG = "".join(RAG)
    # Return a chat template as a string
    chat_template = (
        "You are an assistant that helps with SQL query generation.\n"
        "Instruction: {instruction}\n"
        "Tables schema: {schema}\n"
        "Relevant information: {RAG}\n"
        "Question: {input}"
    )
    # Format the string with actual data
    formatted_input = chat_template.format(
        instruction=data["instruction"],
        schema=data["schema"],
        RAG=RAG,
        input=data["input"]
    )
    return formatted_input


def balance_parentheses(query):
    open_count = 0
    for char in query:
        if char == '(':
            open_count += 1
        elif char == ')':
            open_count -= 1
    if open_count > 0:
        query = re.sub(r';', ')' * open_count + ';', query)

    return query


def clean_output(output):
    tmp = output.upper()
    start_index = tmp.find("SELECT")
    end_index = output.rfind(";")
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return output[start_index:end_index + 1]
    return ""







