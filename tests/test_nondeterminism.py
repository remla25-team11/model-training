import joblib
import numpy as np
import pandas as pd

vectorizer = joblib.load("models/c1_BoW_Sentiment_Model.pkl")
sample = ["good food"]
vector = vectorizer.transform(sample).toarray()
np.savetxt("data/sample_input_vector.csv", vector[0], delimiter=",")

def test_model_deterministic_output():
    model = joblib.load("models/c2_Classifier_Sentiment_Model.pkl")
    sample = np.loadtxt("data/sample_input_vector.csv", delimiter=",")

    outputs = [model.predict(sample.reshape(1, -1))[0] for _ in range(5)]
    assert all(o == outputs[0] for o in outputs)