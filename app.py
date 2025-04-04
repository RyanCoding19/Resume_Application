from flask import Flask, request, jsonify
import joblib
import pandas as pd
from preprocessing import preprocess_data

app = Flask(__name__)

# Load pre-trained models and vectorizer
model = joblib.load('models/resume_classifier_lr.pkl')
tfidf = joblib.load('models/tfidf_vectorizer_lr.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Check if 'resume' key exists in the request
    if not data or 'resume' not in data:
        return jsonify({'error': 'Missing "resume" in request'}), 400

    resume_text = data['resume']

    # Preprocess the input resume text
    cleaned_resume = preprocess_data([resume_text])[0]

    # Vectorize the cleaned resume text using the pre-loaded TF-IDF vectorizer
    resume_vector = tfidf.transform(cleaned_resume)

    # Make prediction using the pre-trained model
    prediction = model.predict(resume_vector)

    # Return the predicted category as a JSON response
    return jsonify({'category': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
