from nltk.corpus import stopwords


def prepare_text(text):
    text = str(text).lower()
    text.replace('.', '').replace('.', '').replace('n\'t', ' not')


def remove_stop_words(text):
    filtered = [word for word in prepare_text(text) if word not in stopwords.words('english')]
    return filtered


def synonymize_words():
    return