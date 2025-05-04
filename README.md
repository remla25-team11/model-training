# Sentiment Analysis

This project is based on [Skillcate AI "Sentiment Analysis Project — with traditional ML & NLP"](https://medium.com/@skillcate/sentiment-analysis-project-with-traditional-ml-nlp-349185bf98dd). The linked Medium article gives further insights into the workflow and the reasoning of the authors.

In this repository, we have simplified the setup and integrated the data to avoid the external Google Drive dependency. This project is used as a runnning example in the course [Release Engineering for Machine Learning Applications (REMLA)](https://studyguide.tudelft.nl/a101_displayCourse.do?course_id=68893) taught at the Delft University of Technology. The copyright remains with the original authors.

The project illustrates how to train a model that performs sentiment analysis on restaurant reviews:

| File | Purpose |
| --- | --- |
| Training pipeline | `b1_Sentiment_Analysis_Model.ipynb` |
| Inference pipeline | `b2_Sentiment_Predictor.ipynb` |
| Training data | `a1_RestaurantReviews_HistoricDump.tsv` |
| New, unlabeled data | `a2_RestaurantReviews_FreshDump` |

The project is known to work with Python 3.10, using the following dependencies:

- `notebook`
- `pandas`
- `nltk`
- `scikit-learn`

