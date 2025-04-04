from flask import Flask, request, jsonify
import joblib
import pandas as pd
from preprocessing import preprocess_data

app = Flask(__name__)

# Load pre-trained models and vectorizers
model_lr = joblib.load('models/resume_classifier_lr.pkl')
tfidf_lr = joblib.load('models/tfidf_vectorizer_lr.pkl')

model_rf = joblib.load('models/resume_classifier_rf.pkl')
tfidf_rf = joblib.load('models/tfidf_vectorizer_rf.pkl')

@app.route('/')
def home():
    return "Resume Classifier API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Check if 'resume' key exists in the request
    if not data or 'resume' not in data:
        return jsonify({'error': 'Missing "resume" in request'}), 400

    resume_text = data['resume']

    try:
        # Preprocess the input resume text
        cleaned_resume = preprocess_data([resume_text])[0]

        # Vectorize the cleaned resume text using the pre-loaded TF-IDF vectorizers for both models
        resume_vector_lr = tfidf_lr.transform(cleaned_resume)
        resume_vector_rf = tfidf_rf.transform(cleaned_resume)

        # Make predictions using the pre-trained models
        prediction_lr = model_lr.predict(resume_vector_lr)
        prediction_rf = model_rf.predict(resume_vector_rf)

        # Return the predicted category for both models as a JSON response
        return jsonify({
            'Logistic Regression Category': prediction_lr[0],
            'Random Forest Category': prediction_rf[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)