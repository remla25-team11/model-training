import os
import pandas as pd
import joblib
import glob


def test_any_model_exists():
    model_files = glob.glob("models/*_Model*.pkl")
    assert len(model_files) > 0, "No model files found in models/ folder"


def test_model_file_exists1():
    assert os.path.exists("models/c1_BoW_Sentiment_Model.pkl"), "Missing model c1_BoW_Sentiment_Model.pkl file"


def test_model_file_exists2():
    assert os.path.exists("models/c2_Classifier_Sentiment_Model.pkl"), "Missing model c2_Classifier_Sentiment_Model file"


def test_vectorizer():
    vectorizer = joblib.load("models/c1_BoW_Sentiment_Model.pkl")
    sample = ["the food was great"]
    vector = vectorizer.transform(sample).toarray()

    assert vector.shape[0] == 1, "Should be one row for one input"
    assert vector.shape[1] == vectorizer.max_features, "Number of columns should match the max number of features"
    assert vector.sum() > 0, "Vector should not be all zeros for valid input"


def test_model_predict_non_empty():
    model = joblib.load("models/c2_Classifier_Sentiment_Model.pkl")
    X_test = pd.read_csv("data/X_test.csv").values

    preds = model.predict(X_test)
    assert len(preds) == X_test.shape[0], "Mismatch between number of predictions and input samples when using c2_Classifier_Sentiment_Model"


def test_saved_correctly():
    assert os.path.exists("data/X_test.csv"), "X_test.csv missing"
    assert os.path.exists("data/y_test.csv"), "y_test.csv missing"

    X_test = pd.read_csv("data/X_test.csv").values
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    assert X_test.shape[0] == y_test.shape[0], "MIsmatch between X_test and Y_test rows count"
    assert len(y_test.shape) == 1, "Y_test should be in 1D"