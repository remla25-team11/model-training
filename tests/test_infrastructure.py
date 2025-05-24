import os
import pandas as pd
import joblib
import numpy as np


def test_model_file_loads():
    assert os.path.exists("models/c2_Classifier_Sentiment_Model.pkl"), "Model c2_Classifier_Sentiment_Model file missing"

    try:
        model = joblib.load("models/c2_Classifier_Sentiment_Model.pkl")
    except Exception as e:
        assert False, f"Model c2_Classifier_Sentiment_Model failed to load: {e}"


def test_model_prediction_structure():
    model = joblib.load("models/c2_Classifier_Sentiment_Model.pkl")
    X_test = pd.read_csv("data/X_test.csv").values
    y_pred = model.predict(X_test)

    assert isinstance(y_pred, np.ndarray), "Predictions must be np array"
    assert y_pred.shape[0] == X_test.shape[0], "Mismatch in prediction and input size"


# Maybe has to be changed in the future if we change the y-labels
def test_model_prediction_binary():
    model = joblib.load("models/c2_Classifier_Sentiment_Model.pkl")
    X_test = pd.read_csv("data/X_test.csv").values

    y_pred = model.predict(X_test)
    unique = set(y_pred)
    assert unique.issubset({0, 1}), f"Predictions contain unexpected values: {unique}"