
#exact match accuracy
def EMA(y_true, y_pred):
    return np.mean(y_true == y_pred)