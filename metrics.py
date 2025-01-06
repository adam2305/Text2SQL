import numpy as np


#exact match accuracy
def EMA(predicts: list, gold_answers: list) -> float:
    sum = np.sum([y_true == y_pred for y_true, y_pred in zip(gold_answers, predicts)])
    return sum / len(gold_answers)

#execution accuracy
def EA(predicts: list, gold_answers: list) -> float:
    """
    Check the ratio of predictions that gave the same execution result as the gold answer
    """
    sum = np.sum([y_true == y_pred for y_true, y_pred in zip(gold_answers, predicts)])
    return sum / len(gold_answers)