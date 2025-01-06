import json
import os
import glob

def get_all_folders(directory):
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return folders

def remove_insert_commands(data: str) -> str:
    # Filter out lines that start with "INSERT INTO"
    filtered_lines = [line for line in data if not line.strip().upper().startswith("INSERT INTO")]

    # Join the filtered lines back into a single string
    return "\n".join(filtered_lines)


directory_path = 'spider/database'
folders = get_all_folders(directory_path)

json_data = []
for folder in folders:
    folder_path = os.path.join(directory_path, folder)
    sql_files = glob.glob(os.path.join(folder_path, '*.sql'))
    if len(sql_files) > 0:
        with open(sql_files[0], 'r') as file:
            content = file.readlines()
            filtered_content = []
            skip = False
        content = remove_insert_commands(content)
        for line in content:
            if line.strip().upper().startswith('INSERT INTO'):
                skip = True
            elif skip and line.strip().endswith(');'):
                skip = False
            elif not skip:
                filtered_content.append(line)
        merged_content = ''.join(filtered_content)
        merged_content = merged_content.replace("\n\n", "\n")
        data = {"db_id": folder, "schema": merged_content}
    else:
        data = {"db_id": folder, "schema": ""}

    json_data.append(data)


with open('data/tables_description.json', 'w') as file:
    json.dump(json_data, file, indent=4)
















