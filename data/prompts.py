import os
import sys
import random

from typing import Optional
from datasets import Dataset

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from data.databases import format_schema

class PromptGenerator:
    def __init__(self, infos: Optional[dict] = None, shots: int = 0, train_set: Optional[Dataset] = None):
        self.infos = infos
        self.shots = shots
        self.train_set = train_set

    def __call__(self, question: str) -> str:
        if self.shots == 0:
            if self.infos is None:
                return self._generate_zero_shot_prompt(question)
            else:
                schemas, p_keys, f_keys = format_schema(self.infos)
                return self._generate_zero_shot_prompt_with_infos(question, schemas, p_keys, f_keys)
        else:
            if self.train_set is None:
                raise ValueError('You have to specify a train_set to perform few-shot generation')
    
            demos = format_demos(get_random_demos(self.shots, self.train_set))
            if self.infos is None:
                return self._generate_few_shot_prompt(question, demos)
            else:
                schemas, p_keys, f_keys = format_schema(self.infos)
                return self._generate_few_shot_prompt_with_infos(question, demos, schemas, p_keys, f_keys)

    def _generate_zero_shot_prompt(self, question: str) -> str:
        prompt = """
You are an expert in databases. Your task is to convert the question after <<<>>> into a valid SQL Query.

You will only answer with the Query.

<<<
Question: {question}
SQL Query:
>>>
        """.strip()
        return prompt.format(question=question)

    def _generate_zero_shot_prompt_with_infos(self, question: str, schemas: str, p_keys: str, f_keys: str) -> str:
        prompt = """
You are an expert in databases. Your task is to convert the question after <<<>>> into a valid SQL Query.

You can help you with the following informations:

Schemas:
{schemas}

Primary Keys:
{p_keys}

Foreign Keys:
{f_keys}

You will only answer with the Query.

<<<
Question: {question}
SQL Query:
>>>
        """.strip()
        return prompt.format(schemas=schemas, p_keys=p_keys, f_keys=f_keys, question=question)

    def _generate_few_shot_prompt(self, question: str, demos: str) -> str:
        prompt = """
You are an expert in databases. Your task is to convert the question after <<<>>> into a valid SQL Query.

You will only answer with the Query.

####
Here are some examples:

{examples}
####

<<<
Question: {question}
SQL Query:
>>>
        """.strip()
        return prompt.format(examples=demos, question=question)

    def _generate_few_shot_prompt_with_infos(self, question: str, demos: str, schemas: str, p_keys: str, f_keys: str) -> str:
        prompt = """
You are an expert in databases. Your task is to convert the question after <<<>>> into a valid SQL Query.

You can help you with the following informations:

Schemas:
{schemas}

Primary Keys:
{p_keys}

Foreign Keys:
{f_keys}

You will only answer with the Query.

####
Here are some examples:

{examples}
####

<<<
Question: {question}
SQL Query:
>>>
        """.strip()
        return prompt.format(schemas=schemas, p_keys=p_keys, f_keys=f_keys, examples=demos, question=question)
            

def format_demo(demo):
    return f"Question: {demo['question']}\nSQL Query: {demo['query']}."
    
def format_demos(demos):
    formatted_str = []
    for demo in demos:
        formatted_str.append(format_demo(demo))
    return '\n\n'.join(formatted_str)
        
def get_random_demos(k, train):
    examples = []
    for i in range(k):
        examples.append(random.choice(train))
    return examples

        
            

