import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import create_sentiment_model as model_builder

MODEL_NAME = 'dnn_estimator_hub'
MODE_DIR = os.path.join(os.getcwd(), MODEL_NAME)

PREDICTION_DICT = {b'0': '1.0', b'1': '2.0', b'2': '3.0', b'3': '4.0', b'4': '5.0'}

embedded_text_feature_column = hub.text_embedding_column(
    key="sentence",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

estimator = tf.estimator.DNNClassifier(hidden_units=[500, 500],
                                       feature_columns=[embedded_text_feature_column],
                                       n_classes=4,
                                       optimizer=tf.train.AdamOptimizer(learning_rate=0.003),
                                       model_dir=MODE_DIR)


def predict(text):
    model_path = Path(MODE_DIR)
    if not model_path.exists():
        model_builder.build_model()
    single_text_df = pd.DataFrame({'sentence': [text]})
    predict_input_fn = tf.estimator.inputs.pandas_input_fn(single_text_df, shuffle=False)
    prediction = list(estimator.predict(input_fn=predict_input_fn))[0]['classes'][0]
    return PREDICTION_DICT[prediction]
