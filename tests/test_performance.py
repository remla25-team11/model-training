import joblib
import time
import pandas as pd
import tracemalloc

def test_model_inference_time():
    model = joblib.load("models/c2_Classifier_Sentiment_Model.pkl")
    X_test = pd.read_csv("data/X_test.csv").values[:10]

    start_time = time.time()
    model.predict(X_test)
    elapsed = time.time() - start_time

    assert elapsed < 0.5

def test_model_memory_usage():
    model = joblib.load("models/c2_Classifier_Sentiment_Model.pkl")
    X_test = pd.read_csv("data/X_test.csv").values[:10]

    tracemalloc.start()
    model.predict(X_test)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 50_000_000