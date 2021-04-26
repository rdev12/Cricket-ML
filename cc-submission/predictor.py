### Custom definitions and classes if any ###
import numpy as np
import pandas as pd
from basic_regression import get_predictions
from trainer import train_data

def predictRuns(testInput):
    input = pd.read_csv(testInput)
    X, Y, test_case = train_data(input)
    runs_pred = get_predictions(test_case, X, Y)
    return runs_pred
