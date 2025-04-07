Resume Selector

This project is a Flask-based web application that classifies resumes into different job categories using machine learning. Users can upload a PDF resume, and the app will predict the most suitable job category based on its content.


Project Structure
Resume_Selector/

├── Resume_Application/

│   ├── app.py                  # Main Flask application

│   ├── preprocessing.py        # Text cleaning and feature extraction

│   ├── train_model.py          # Model training script

│   ├── Templates/

│   │   └── index.html          # Frontend template

│   ├── models/                 # Pretrained models and TF-IDF vectorizers

│   ├── data/

│   │   └── Resume.csv          # Dataset for training

│   └── requirements.txt        # Dependencies

Features

Resume text extraction from PDFs

TF-IDF vectorization

Classification using Logistic Regression and Random Forest

Named Entity Recognition with spaCy

Clean and modular code

Setup Instructions

Clone the repository

$ git clone https://github.com/RyanCoding19/Resume_Application

$ cd Resume_Selector/Resume_Application

Install dependencies

$ pip install -r requirements.txt 

$ python -m spacy download en_core_web_sm

Train the model (optional)

$ python train_model.py

# activate environment to support MLflow and pydantic v2

$ cd ..

$ python -m venv mlflow-env

$ mlflow-env\Scripts\activate

$ pip install mlflow

# Navigate to the original project folder

$ cd Resume_Application

$ mlflow ui

http://127.0.0.1:5000

Run the application

$ flask run

Then open your browser at http://127.0.0.1:5000

Upload Format

Only .pdf files are supported


Deployed Endpoint

The model is live at: https://resume-selector-zzew.onrender.com/

Example HTTP request (using `curl`)

bash
curl -X POST -F "file=@resume.pdf" https://resume-selector-zzew.onrender.com/

Make sure the resume contains readable text (not images only)

Testing

To run unit tests:

$ pytest tests/

Dependencies

Flask

scikit-learn

joblib

PyPDF2

spaCy

pandas

Team Contribution

Each team member helped contribute to the project and assisted in developing and training the code, Ryan created the repository and created the baseline for the project, Kenneth developed the application further and enhanced the API, Endpoints and UI and Nico helped develop the UI and Presentation for Phase-2.

