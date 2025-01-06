from datasets import Dataset
from typing import Optional

import os
import sys

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from data.databases import DBInfos
from data.prompts import PromptGenerator

def adapt_dataset(dataset, format: Optional[dict] = None):

    db_ids = dataset["db_id"]
    questions = dataset["question"]
    queries = dataset["query"]

    if not format:
        inputs = questions

    else:
        if format['infos']:
            db_infos = DBInfos("data/tables.json")
            infos = [db_infos.get(db_id) for db_id in db_ids]
        else:
            infos = [None for _ in questions]

        if not isinstance(format["shots"], int):
            raise TypeError('"shots" should be an integer')
        else:
            shots = [format['shots'] for _ in questions]
        
        inputs = [None for _ in questions]
        for i, question in enumerate(questions):    
            prompt_generator = PromptGenerator(infos = infos[i], shots = shots[i], train_set = dataset)
            inputs[i] = prompt_generator(question)
    
    data_dict = {
        "input": inputs,
        "target": queries,
    }
    return Dataset.from_dict(data_dict)
