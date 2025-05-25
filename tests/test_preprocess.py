# from lib_ml import preprocess_text
import os
import pandas as pd
import glob

# defining here for now to be removed
#### START HERE
import re
import nltk
# nltk.download('stopwords')
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

def test_preprocess_text():
    raw = "I loved the FOOD!!! 😍😍"
    processed = preprocess_text(raw)
    assert isinstance(processed, str)
    assert "love" in processed  # assuming stemming reduces "loved" to "love"
    assert "food" in processed
    assert "😍" not in processed


# Ai has been used to generate this line of code
def test_any_training_data_exists():
    training_files = glob.glob("data/a*_RestaurantReviews_*.tsv")
    assert len(training_files) > 0, "No training data files found in data/ folder"
# end of AI generated code


def test_data_file_exists1():
    assert os.path.exists("data/a1_RestaurantReviews_HistoricDump.tsv"), "Missing a1_RestaurantReviews_HistoricDump.tsv file in data folder"


def test_data_file_exists2():
    assert os.path.exists("data/a2_RestaurantReviews_FreshDump.tsv"), "Missing a2_RestaurantReviews_FreshDump.tsv file in data folder"


def test_data_file_exists3():
    assert os.path.exists("data/c3_Predicted_Sentiments_Fresh_Dump.tsv"), "Missing c3_Predicted_Sentiments_Fresh_Dump.tsv file in data folder"


def test_processed():
    X = pd.read_csv("data/X.csv")
    y = pd.read_csv("data/y.csv")
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 0, "There should be at least one feature"


# Maybe has to be changed in the future if we change the y-labels
def test_label_binary_val():
    y = pd.read_csv("data/y.csv").values.ravel()
    assert set(y).issubset({0, 1})


def test_not_all_empty():
    X = pd.read_csv("data/X.csv").values
    non_empty_rows = sum(row.sum() > 0 for row in X)
    assert non_empty_rows > 0, "All rows in X are empty"
