import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import json

def run():
    # Load model and test data
    classifier = joblib.load('c2_Classifier_Sentiment_Model')
    X_test = pd.read_csv('data/X_test.csv').values
    y_test = pd.read_csv('data/y_test.csv').values.ravel()

    # Predict and evaluate
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)

    print(f'Confusion matrix: {cm}')
    print(f'Accuracy score: {acc_score:.4f}')

    # Save metrics for DVC tracking
    with open('metrics.json', 'w') as f:
        json.dump({"accuracy": float(acc_score)}, f)

if __name__ == "__main__":
    run()
