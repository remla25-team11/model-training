import pickle
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

# - Import dataset
dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)
dataset.shape
dataset.head()

# - Data preprocessing
#TODO fetch corpus from lib-ml repository
corpus = []

# - Data transformation
cv = CountVectorizer(max_features = 1420)

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Saving BoW dictionary to later use in prediction
bow_path = 'c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, "wb"))

# - Divide dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# - Model fitting (Naive Bayes)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# - Export NB Classifier to later use in prediction
joblib.dump(classifier, 'c2_Classifier_Sentiment_Model')

# - Model performance
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(f'Confusion matrix: {cm}')

acc_score = accuracy_score(y_test, y_pred)
print(f'Accuracy score: {acc_score}')
