from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import words
data = []
data_labels = []

MODEL_FILE_NAME = '../bayes.pkl'
VECTOR_FILE_NAME = '../vector.pkl'
DATA_FILE_NAME = '../train_data.txt'

def synonimize(text):
    equivalents = list()
    new_text = list()
    for word in text.split(' '):
        if word in words and word not in stopwords:
            new_text = new_text + word + ' '
    # generate word synonyms: use word embedding in the future
    for i in range(0, len(new_text)):
        synonyms = get_synonyms(new_text[i])
        for syn in synonyms:
            equivalent = new_text[i]


def get_synonyms (word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

def model():
    with open(DATA_FILE_NAME) as f:
        for line in f:
            rating_arr = line.split(' ', 1)
            rating = rating_arr[0]
            text = rating_arr[1]
            data.append(text)
            data_labels.append(rating)

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
        train_size=0.80,
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
    return prediction

