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


def get_input_gemma_template(data):
    return [{"role": "user",
             "content": data["instruction"] + "\n\n" + "Example : " + data["question"] + "\n\n" + "tables schema : " + data["schema"] + "\n\n" + "Question : " + data["input"]}]

def get_input_gemma_template_fewshot(data):
    return [{"role": "user",
             "content": data["instruction"] + "\n\n" + "tables schema : " + data["schema"] + "\n\n" + "Question : " + data["input"]}]

def get_input_gemma_template_RAG(data, RAG):
    return [{"role": "user",
             "content": data["instruction"] + "\n\n" + "tables schema : " + data["schema"] + "Relevant information : " + RAG + "\n\n" + "Question : " + data["input"]}]

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







