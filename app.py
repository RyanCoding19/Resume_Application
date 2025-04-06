from flask import Flask, request, jsonify, render_template
import joblib
from PyPDF2 import PdfReader  # library to extract text from PDF files
import os
from preprocessing import preprocess_data 

app = Flask(__name__)

# Load pre-trained models and vectorizers
model_lr = joblib.load('models/resume_classifier_lr.pkl')
tfidf_lr = joblib.load('models/tfidf_vectorizer_lr.pkl')

model_rf = joblib.load('models/resume_classifier_rf.pkl')
tfidf_rf = joblib.load('models/tfidf_vectorizer_rf.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        try:
            # Step 1: Read PDF file
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            # Step 2: Preprocess the extracted text using the preprocess_data function
            cleaned_text, _ = preprocess_data([text])  # Only preprocess the text (no vectorization yet)

            # Step 3: Vectorize the cleaned text using the pre-trained vectorizers
            resume_vector_lr = tfidf_lr.transform(cleaned_text)  # Vectorize for Logistic Regression
            resume_vector_rf = tfidf_rf.transform(cleaned_text)  # Vectorize for Random Forest

            # Step 4: Make predictions using the models
            prediction_lr = model_lr.predict(resume_vector_lr)
            prediction_rf = model_rf.predict(resume_vector_rf)

            # Step 5: Return the predicted category for both models
            return jsonify({
                'Logistic Regression Category': prediction_lr[0],
                'Random Forest Category': prediction_rf[0]
            })
        
        except Exception as e:
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500
    
    else:
        return jsonify({'error': 'Invalid file format. Only PDF is supported.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
