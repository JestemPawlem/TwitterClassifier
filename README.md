# Twitter Sentiment Analysis Web App
A simple web application that classifies the sentiment of text (Positive/Negative) using machine learning.
It features an online learning system where the model can be updated based on user feedback.

## Features
 - **Real-time Classification**: Uses an SGD Classifier and Hashing Vectorizer.
 - **SQLite Integration**: Automatically logs tweets, predictions and timestamps.
 - **Feedback Loop**: Users can correct the model, triggering a `partial_fit` to improve accuracy.

## Tech Stack
 - **Python 3.13**
 - **Flask**
 - **Skicit-learn**
 - **SQLite3**
 - **WTForms**

## How to run
 1. **Clone the repository**:
    git clone https://github.com/JestemPawlem/TwitterClassifier.git
    cd TwitterClassifier
 2. **Setup and activate virtual environment**:
    python -m venv .venv
    .\venv\Scripts\activate
 3. **Install dependencies**:
    pip install -r requirements.txt
 4. **Start the application**:
    python twitter_classifier/app.py

## Project structure
 - app.py: Core application logic and database management.
 - vectorizer.py: Text preprocessing and vectorization.
 - pkl_objects/: Serialized model and stopwords.
 - templates/: HTML templates for the UI.
 - static/: CSS stylesheets for the application.