import joblib

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from lib_ml import preprocess_text  # make sure this exists or define it here

def run():

    dataset = pd.read_csv(
        'data/a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3
    )

    corpus = [preprocess_text(review) for review in dataset['Review'][:900]]

    cv = CountVectorizer(max_features=1420)
    x = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:900, -1].values

    # Save BoW model
    joblib.dump(cv, 'models/c1_BoW_Sentiment_Model.pkl')

    pd.DataFrame(x).to_csv('data/X.csv', index=False)
    pd.DataFrame(y).to_csv('data/y.csv', index=False)


if __name__ == "__main__":
    run()
