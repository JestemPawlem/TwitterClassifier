from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

from vectorizer import vect


app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'tweets.sqlite')


def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    return label[y]

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO tweets_db (tweet, sentiment, date) VALUES (?, ?, DATETIME('now'))",
              (document, y))
    conn.commit()
    conn.close()


class TweetForm(Form):
    tweet = TextAreaField('',
                          [validators.DataRequired(), validators.length(min=1, max=500)])


@app.route('/')
def index():
    form = TweetForm(request.form)
    return render_template('tweetform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = TweetForm(request.form)
    if form.validate():
        tweet =request.form['tweet']
        y = classify(tweet)
        return render_template('results.html',
                               content=tweet,
                               prediction=y)
    return render_template('tweetform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    tweet = request.form['tweet']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(tweet, y)
    sqlite_entry(db, tweet, y)
    return render_template('thanks.html')


def init_db(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute('''
    CREATE TABLE IF NOT EXISTS tweets_db (
        tweet TEXT,
        sentiment INTEGER,
        date TEXT
    )
    ''')
    conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db(db)
    app.run(debug=True)