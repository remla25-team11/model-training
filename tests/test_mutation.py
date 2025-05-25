import joblib
import numpy as np
import os

def test_synonyms():
    vectorizer_path = "models/c1_BoW_Sentiment_Model.pkl"
    model_path = "models/c2_Classifier_Sentiment_Model.pkl"

    assert os.path.exists(vectorizer_path), "Missing c1_BoW_Sentiment_Model model"
    assert os.path.exists(model_path), "Missing c2_Classifier_Sentiment_Model model"

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    input_1 = ["okay"]
    input_2 = ["fine"]

    X_a = vectorizer.transform(input_1).toarray()
    X_b = vectorizer.transform(input_2).toarray()

    pred_a = model.predict(X_a)[0]
    pred_b = model.predict(X_b)[0]

    assert pred_a != pred_b, (
        f"Expected equal, received: {pred_a} vs {pred_b}"
    )
