import sys
import re
import sqlite3
import os
import glob
from statistics import mean
from nltk.translate.bleu_score import sentence_bleu

sys.path.append(os.path.dirname(os.getcwd()))
from model_training_llama.utils import open_json_file, clean_output, balance_parentheses


def get_void_output(output_path):
    results = open_json_file(output_path)
    void_output = 0
    for result in results:
        if not ("SELECT" in result["output"].upper() and "FROM" in result["output"].upper()):
            void_output += 1
    return [void_output, void_output / len(results)]

def get_EMA(output_path):
    results = open_json_file(output_path)
    exact_match = 0
    for result in results:
        match = re.search(r"```sql\s*(.*?)\s*```", result["output"], re.IGNORECASE | re.DOTALL)
        if match:
            output =  match.group(0).strip()
            output = output.replace("\n", " ").replace("\t", "").replace(";","").replace("'", "\"")
            output = re.sub(r'\s+', ' ', output).strip()
            output = output.replace("```sql", "").replace("```", "").replace(";", "").replace(",", "")
            if "," in result["gold_output"]:
                result["gold_output"] = result["gold_output"].replace(",", "").replace(";", "").replace("'", "\"")
            result["gold_output"] = result["gold_output"].replace(";", "").replace("'", "\"")
            if output.lower().split() == result["gold_output"].lower().strip().split():
                exact_match += 1
        else:
            output = clean_output(result["output"])
            if output != "":
                output = output.replace("\n", " ").replace("\t", "").replace(";", "").replace("'", "\"")
                output = re.sub(r'\s+', ' ', output).strip()
                output = output.replace("```sql", "").replace("```", "").replace(";", "").replace(",", "")
                if "," in result["gold_output"]:
                    result["gold_output"] = result["gold_output"].replace(",", "").replace(";", "").replace("'", "\"")
                result["gold_output"] = result["gold_output"].replace(";", "").replace("'", "\"")
                if output.lower().split() == result["gold_output"].lower().strip().split():
                    exact_match += 1
    return exact_match


def Initialize_databases(path):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    for folder in folders:
        db_file = os.path.join(path, folder, f"{folder}.sqlite")
        sql_file = glob.glob(os.path.join(path, folder, '*.sql'))
        if len(sql_file) > 0:
            try:
                connection = sqlite3.connect(db_file)
                cursor = connection.cursor()
                with open(sql_file[0], 'r') as file:
                    sql_script = file.read()
                cursor.executescript(sql_script)
                connection.commit()
                print("SQL script executed successfully.")
            except sqlite3.Error as e:
                print(f"An error occurred: {e}")
                continue
            finally:
                if connection:
                    connection.close()


def test_sql_command(data, db_path):
    sql_command = clean_output(data["output"])
    sql_command = sql_command.replace("```sql", "").replace("```", "")
    db_file = os.path.join(db_path, f"{data['db_id']}/{data['db_id']}.sqlite")
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()
    try:
        cursor.execute(sql_command)
        execute = True
    except:
        execute = False
    finally:
        if connection:
            connection.close()
    return execute


def get_syntax_validity(output_path, db_path):
    results = open_json_file(output_path)
    syntax_validity = 0
    for result in results:
        if "SELECT" in result["output"].upper():
            if test_sql_command(result, db_path):
                syntax_validity += 1
    return [syntax_validity, syntax_validity / len(results)]


def string_f1_score(str1, str2):
    words1 = set(str1.split())
    words2 = set(str2.split())
    common_words = words1.intersection(words2)
    precision = len(common_words) / len(words2) if len(words2) > 0 else 0
    recall = len(common_words) / len(words1) if len(words1) > 0 else 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return f1, precision, recall


def get_f1_score(output_path):
    results = open_json_file(output_path)
    f1_value = []
    recall_value = []
    precision_value = []
    for result in results:
        match = re.search(r"```sql\s*(.*?)\s*```", result["output"], re.IGNORECASE | re.DOTALL)
        output = " "
        if match:
            output =  match.group(0).strip()
            output = output.replace("\n", " ").replace("\t", " ").replace(";","").replace("'", "\"")
            output = re.sub(r'\s+', ' ', output).strip()

    f1, precision, recall = string_f1_score(output, result["gold_output"])
    f1_value.append(f1)
    recall_value.append(recall)
    precision_value.append(precision)
    return mean(f1_value), mean(recall_value), mean(precision_value)


def calculate_bleu_from_strings(reference_str, candidate_str):
    reference = [reference_str.split()]
    candidate = candidate_str.split()
    return sentence_bleu(reference, candidate)

def get_BLEU_score(output_path):
    results = open_json_file(output_path)
    bleu_value = []
    for result in results:
        match = re.search(r"```sql\s*(.*?)\s*```", result["output"], re.IGNORECASE | re.DOTALL)
        output = " "
        if match:
            output =  match.group(0).strip()
        output = output.replace("\n", " ").replace("\t", " ").replace(";","").replace("'", "\"")
        output = re.sub(r'\s+', ' ', output).strip()

        bleu_value.append(calculate_bleu_from_strings(result["gold_output"], output))
    return mean(bleu_value)


def get_execution_accuracy(output_path, db_path):
    results = open_json_file(output_path)
    execution_match = 0
    for result in results:
        db_file = os.path.join(db_path, f"{result['db_id']}/{result['db_id']}.sqlite")
        sql_command = clean_output(result["output"])
        sql_command = sql_command.replace("```sql", "").replace("```", "")
        db_file = os.path.join(db_path, f"{result['db_id']}/{result['db_id']}.sqlite")
        connection = sqlite3.connect(db_file)
        cursor = connection.cursor()
        try:
            cursor.execute(sql_command)
            output_result = cursor.fetchall()
            cursor.execute(result["gold_output"])
            gold_result = cursor.fetchall()
            if set(output_result) == set(gold_result):
                execution_match += 1
        except:
            continue
        finally:
            if connection:
                connection.close()
    return execution_match

error_categories = {
    "syntax error": ["syntax error"],
    "near": ["near"],
    "no such table": ["no such table"],
    "no such column": ["no such column"],
    "unable to open database file": ["unable to open database file"],
    "timeout": ["timeout"],
    "szmre": ["szmre"]
}

def categorize_error(error_message):
    for category, keywords in error_categories.items():
        if any(keyword in error_message.lower() for keyword in keywords):
            return category
    return "Other Error"

def test_sql_commands_and_categorize_errors(output_path, db_path):
    results = open_json_file(output_path)
    sql_errors = []
    error_counts = {}

    for result in results:
        sql_command = clean_output(result["output"])
        sql_command = sql_command.replace("```sql", "").replace("```", "")
        db_file = os.path.join(db_path, f"{result['db_id']}/{result['db_id']}.sqlite")
        connection = sqlite3.connect(db_file)
        cursor = connection.cursor()
        try:
            cursor.execute(sql_command)
        except sqlite3.Error as e:
            error_message = str(e)
            category = categorize_error(error_message)
            sql_errors.append({"db_id": result["db_id"], "error": error_message, "category": category, "sql_command": sql_command})
            if category in error_counts:
                error_counts[category] += 1
            else:
                error_counts[category] = 1
        finally:
            if connection:
                connection.close()

    return sql_errors, error_counts

# List of common SQL keywords
sql_keywords = [
    "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN",
    "FULL JOIN", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "OFFSET", "UNION", "DISTINCT"
]

def find_keywords_in_failed_queries(output_path, db_path):
    results = open_json_file(output_path)
    failed_queries = []

    for result in results:
        sql_command = clean_output(result["output"])
        sql_command = sql_command.replace("```sql", "").replace("```", "")
        db_file = os.path.join(db_path, f"{result['db_id']}/{result['db_id']}.sqlite")
        connection = sqlite3.connect(db_file)
        cursor = connection.cursor()
        try:
            cursor.execute(sql_command)
        except sqlite3.Error as e:
            error_message = str(e)
            found_keywords = [keyword for keyword in sql_keywords if keyword in sql_command.upper()]
            failed_queries.append({"db_id": result["db_id"], "error": error_message, "sql_command": sql_command, "keywords": found_keywords})
        finally:
            if connection:
                connection.close()

    return failed_queries










