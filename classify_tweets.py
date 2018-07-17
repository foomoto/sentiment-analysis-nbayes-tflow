from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path
from nltk.corpus import stopwords
data = []
data_labels = []

TWITTER_DATA_FILE_NAME = './emotion_data_twitter.txt'
MODEL_FILE_NAME = '../bayes_tweets.pkl'
VECTOR_FILE_NAME = '../vector_tweets.pkl'

emotions = set()
with open(TWITTER_DATA_FILE_NAME) as f:
    for line in f:
        emotions.add(line.split(',')[1])
sentiments_arr = list(emotions)

def model():
    with open(TWITTER_DATA_FILE_NAME) as f:
        for line in f:
            line_split = line.split(',')
            sentiment = sentiments_arr.index(line_split[1])
            text = line_split[3]
            data.append(text)
            data_labels.append(sentiment)

    vec_tool = CountVectorizer(
        analyzer='word',
        lowercase=True
    )

    raw_features = vec_tool.fit_transform(
        data
    )

    features = raw_features.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        data_labels,
        train_size=0.70,
        random_state=1234)

    log_model = MultinomialNB()
    log_model = log_model.fit(X=X_train, y=y_train)
    joblib.dump(log_model, MODEL_FILE_NAME)
    joblib.dump(vec_tool, VECTOR_FILE_NAME)
    y_pred = log_model.predict(X_test)
    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))


def predict(text):
    model_path = Path(MODEL_FILE_NAME)
    vector_path = Path(VECTOR_FILE_NAME)
    if not model_path.exists() or not vector_path.exists():
        model()
    log_model = joblib.load(MODEL_FILE_NAME)
    vec_tool = joblib.load(VECTOR_FILE_NAME)
    future = vec_tool.transform([text])
    prediction = log_model.predict(future)
    return sentiments_arr[prediction[0]]

print(predict('Funeral'))

