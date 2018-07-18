from flask import Flask, render_template, redirect, url_for
from flask import request
from flask import jsonify
from flask import make_response
import time
from jinja2 import Environment, FileSystemLoader
from urllib import parse
import os, sys
import logging
import classify_ratings_bayes_with_word_embedding as bayes
import predict_sentence as tensorflow


from sklearn.externals import joblib

app = Flask(__name__)
app.config.update(
    DEBUG=True,
    PORT='5000',
    SEND_FILE_MAX_AGE_DEFAULT=0
)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')

log = logging.getLogger(__name__)

emotions = {'1.0': 'Angry or very unhappy',
                '2.0' : 'Sad or moody',
            '3.0' : 'Indifferent or neutral',
            '4.0' : 'Somewhat happy',
            '5.0' : 'Super excited'
            }

@app.route('/')
def home():
    q = request.args.get('q')
    model = request.args.get('bayes')
    image = 'logo.png'
    message = 'Hello, I can predict how you feel'
    if q is not None and model is not None:
        prediction = bayes.predict(q)[0]
        message = emotions[prediction]
        image = prediction.split('.')[0] + '.png'
    elif q is not None:
        prediction = tensorflow.predict(q)
        message = emotions[prediction]
        image = prediction.split('.')[0] + '.png'
    return render_template("index.html", route='home', image_name=image, message=message)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    if 'OCTOPUS_ENV' in os.environ and os.environ['OCTOPUS_ENV'] == 'local':
        app.run(host="0.0.0.0", port=5000)
    else:
        print("Running in development environment")
        app.run(host="0.0.0.0", port=5000)
