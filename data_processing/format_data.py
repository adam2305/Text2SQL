import json

def get_json(path):
    data = None
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def format_data(data_json, table_description):
    new_data = []
    for i in range(len(data_json["db_id"])):
        db_id = data_json["db_id"][str(i)]
        for table in table_description:
            if table["db_id"] == db_id and table["schema"] != "":
                schema = table["schema"]
                instruction = ""
                output = ""
                input = data_json["question"][str(i)]
                gold_output = data_json["query"][str(i)]
                new_data.append({"db_id": db_id,
                                 "schema": schema,
                                 "instruction": instruction,
                                 "input": input,
                                 "output": output,
                                 "gold_output": gold_output})
    return new_data

def format_data_few_shot(data_json, table_description, few_shot):
    new_data = []
    for i in range(len(data_json["db_id"])):
        db_id = data_json["db_id"][str(i)]
        for index, table in enumerate(table_description):
            if table["db_id"] == db_id and table["schema"] != "":
                schema = table["schema"]
                instruction = ""
                output = ""
                input = data_json["question"][str(i)]
                example = few_shot[index]["few_shot"]
                gold_output = data_json["query"][str(i)]
                new_data.append({"db_id": db_id,
                                 "schema": schema,
                                 "instruction": instruction,
                                 "example" : example,
                                 "input": input,
                                 "output": output,
                                 "gold_output": gold_output})
    return new_data

def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def convert_to_rag_format(input_json):
    rag_data = []

    for db in input_json:
        db_id = db['db_id']
        schema = db['schema']
        documents = []
        documents.append({"text": f"Schema for {db_id} database: {schema.split('\n')[0]}"})
        tables_info = schema.split("\n\n")[1:]
        for table_info in tables_info:
            lines = table_info.split("\n")
            table_name = lines[0].strip()
            columns_info = "\n".join(lines[1:]).strip()
            documents.append({"text": f"{table_name}: {columns_info}"})
        relationships_info = schema.split("### Key Relationships:")[1:]
        if relationships_info:
            documents.append({"text": f"Relationships: {relationships_info[0].strip()}"})
        rag_data.append({
            "db_id": db_id,
            "documents": documents
        })

    return rag_data


table_description = get_json('data/tables_description.json')
table_description_enhanced = get_json('data/tables_description_enhanced.json')
raw_train_data = get_json('data/raw_train_data.json')
raw_dev_data = get_json('data/raw_dev_data.json')
few_shot = get_json('data/tables_few_shot.json')

formated_train_data = format_data(raw_train_data, table_description)
formated_dev_data = format_data(raw_dev_data, table_description)

formated_dev_data_enhanced = format_data(raw_dev_data, table_description_enhanced)

formated_train_data_few_shot = format_data_few_shot(raw_dev_data, table_description_enhanced, few_shot)

RAG_data = convert_to_rag_format(formated_dev_data_enhanced)

save_json(formated_train_data, 'data/formated_train_data.json')
save_json(formated_dev_data, 'data/formated_dev_data.json')
save_json(formated_dev_data_enhanced, 'data/formated_dev_data_enhanced.json')
save_json(formated_train_data_few_shot, 'data/formated_dev_data_few_shot.json')
save_json(RAG_data, 'data/RAG_data.json')




