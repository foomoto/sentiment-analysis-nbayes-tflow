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
import nltk
from fuzzywuzzy import fuzz as fsearch

data = []
data_labels = []

MODEL_FILE_NAME = './bayes.pkl'
VECTOR_FILE_NAME = './vector.pkl'
DATA_FILE_NAME = './train_data.txt'

# Needs more machine power
def check_similar_word_dic (text):
    dic = words.words()
    for word in dic:
        if fsearch.ratio(str(text).lower(), str(word).lower()) > 95:
            return word
    return None

def get_sentence_equivalence (text):
    equivalents = list()
    new_text = list()
    for word in text.split(' '):
        word = word.replace('n\'t', ' not').lower()
        if word not in stopwords.words('english') or word is 'not' or word is 'non':
            new_text.append(word)

        # elif word.lower() in words.words():
        #     new_text.append(word)
        # elif word not in words.words():
        #     similar = check_similar_word_dic(word)
        #     if similar is not None:
        #         new_text.append(word)

    # generate word synonyms: use word/character embedding in the future
    for i in range(0, len(new_text)):
        word = new_text[i]
        # if word is in dictionary look for equivalents
        if word.lower() in words.words():
            synonyms = get_synonyms(word)
            for syn in synonyms:
                equivalent = new_text.copy()
                equivalent[i] = syn
                equivalents.append(' '.join(equivalent))
    return equivalents


def get_synonyms (word):
    count = 1
    synonyms = set()
    for syn in wordnet.synsets(word):
        if count <= 0:
            break
        for l in syn.lemmas():
            synonyms.add(l.name())
            count = count - 1
    return synonyms

def model():
    nltk.download('words')
    nltk.download('stopwords')
    nltk.download('wordnet')
    with open(DATA_FILE_NAME) as f:
        for line in f:
            rating_arr = line.split(' ', 1)
            rating = rating_arr[0]
            text = rating_arr[1]
            data.append(text)
            data_labels.append(rating)
            for equivalent in get_sentence_equivalence(text):
                data.append(equivalent)
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
    print(prediction)
    return prediction

predict("NOT RECOMMENDED")