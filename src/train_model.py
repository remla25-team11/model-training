import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# pylint: disable=invalid-name

def run():
    # Load features
    X = pd.read_csv('data/X.csv').values
    y = pd.read_csv('data/y.csv').values.ravel()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train classifier
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Save model
    joblib.dump(classifier, 'c2_Classifier_Sentiment_Model')

    # Save test data
    pd.DataFrame(X_test).to_csv('data/X_test.csv', index=False)
    pd.DataFrame(y_test).to_csv('data/y_test.csv', index=False)

if __name__ == "__main__":
    run()
