import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.INFO)


MODEL_NAME = 'dnn_estimator_hub_v_2'
MODE_DIR = os.path.join(os.getcwd(), MODEL_NAME)


def vectorize_label_(label):
    if label == "1.0":
        return 0
    elif label == "2.0":
        return 1
    elif label == "3.0":
        return 2
    elif label == "4.0":
        return 3
    elif label == "5.0":
        return 4


def remove_encoding(row):
    return "".join(i for i in row if ord(i) < 128)


def load_directory_data_(directory):
    data = pd.read_csv(directory)
    data.columns = ['sentiment', 'sentence']
    data['sentiment'] = data['sentiment'].astype(str)

    data['label'] = data['sentiment'].apply(lambda x: vectorize_label_(x))
    data['sentence'] = data['sentence'].apply(remove_encoding)

    text_train, text_test, label_train, label_test = train_test_split(data['sentence'].tolist(),
                                                                      data['label'].tolist(),
                                                                      test_size=0.3)

    train_df = pd.DataFrame({'sentence': text_train, 'label': label_train})
    test_df = pd.DataFrame({'sentence': text_test, 'label': label_test})

    return train_df, test_df


def build_model(directory="/Users/oladeletosin/Downloads/train_data_sentiment.csv"):
    train_df, test_df = load_directory_data_(directory)
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
        n_classes=5,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.003),
        model_dir=MODE_DIR)

    estimator.train(input_fn=train_input_fn, steps=250000)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    tf.logging.info("Training set accuracy: {accuracy}".format(**train_eval_result))
    tf.logging.info("Test set accuracy: {accuracy}".format(**test_eval_result))

build_model()
