import joblib

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# from lib_ml import preprocess_text  # make sure this exists or define it here

# defining here for now to be removed
#### START HERE
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
if 'not' in all_stopwords:
    all_stopwords.remove('not')

def preprocess_text(text):
    """
    Cleans the text for sentiment analysis using:
    - Regex cleanup
    - Lowercasing
    - Tokenizing
    - Stopword removal (excluding 'not')
    - Stemming
    """
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    return ' '.join(review)
#### REMOVE UNTIL HERE


def run():

    dataset = pd.read_csv(
        'data/a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3
    )

    corpus = [preprocess_text(review) for review in dataset['Review'][:900]]

    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:900, -1].values

    # Save BoW model
    joblib.dump(cv, 'models/c1_BoW_Sentiment_Model.pkl')

    pd.DataFrame(X).to_csv('data/X.csv', index=False)
    pd.DataFrame(y).to_csv('data/y.csv', index=False)


if __name__ == "__main__":
    run()
