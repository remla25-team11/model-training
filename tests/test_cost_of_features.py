import joblib
import numpy as np
import os


# Check feature dimensionality
def test_bow_vector_size():
    vectorizer = joblib.load("models/c1_BoW_Sentiment_Model.pkl")
    sample = ["average food and ok service"]

    vector = vectorizer.transform(sample).toarray()
    assert vector.shape[1] < 2000, "Bag of Words vector size should be less than 2000 features"
