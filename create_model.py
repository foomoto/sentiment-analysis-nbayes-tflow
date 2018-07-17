import os
import pandas as pd
import tensorflow as tf
import re
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.INFO)


MODEL_NAME = 'dnn_estimator_hub'
MODE_DIR = os.path.join(os.getcwd(), MODEL_NAME)


def rework_label(label):

    if label == "empty": return "neutral" # neutral
    elif label == "sadness": return "sad"
    elif label == "enthusiasm": return "happy"
    elif label == "neutral": return "neutral"
    elif label == "worry": return "sad"
    elif label == "surprise": return "happy"
    elif label == "love": return "happy"
    elif label == "fun": return "happy"
    elif label == "hate": return "anger"
    elif label == "happiness": return "happy"
    elif label == "boredom": return "neutral"
    elif label == "relief": return "happy"
    elif label == "anger": return "anger"


def vectorize_label(label):
    if label == "neutral":
        return 0
    elif label == "sad":
        return 1
    elif label == "happy":
        return 2
    elif label == "anger":
        return 3


def load_directory_data(directory):
    data = pd.read_csv(directory)

    # print(data.head(5))
    # print(set(data['sentiment']))

    data['rework_sentiment'] = data['sentiment'].apply(lambda x: rework_label(x))
    data['label'] = data['rework_sentiment'].apply(lambda x: vectorize_label(x))
    # print(set(data['rework_sentiment']))

    text_train, text_test, label_train, label_test = train_test_split(data['content'].tolist(),
                                                                      data['label'].tolist(),
                                                                      test_size=0.3)

    train_df = pd.DataFrame({'sentence': text_train, 'label': label_train})
    test_df = pd.DataFrame({'sentence': text_test, 'label': label_test})

    return train_df, test_df


def build_model(directory="text_emotion.csv"):
    train_df, test_df = load_directory_data(directory)
    # Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["label"], num_epochs=None, shuffle=True)

    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["label"], shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df["label"], shuffle=False)

    tf.logging.info("loading embeddings..")
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    tf.logging.info("finished loading embeddings...")

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 500],
        feature_columns=[embedded_text_feature_column],
        n_classes=4,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.003),
        model_dir=MODE_DIR)

    estimator.train(input_fn=train_input_fn, steps=250000)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    tf.logging.info("Training set accuracy: {accuracy}".format(**train_eval_result))
    tf.logging.info("Test set accuracy: {accuracy}".format(**test_eval_result))
