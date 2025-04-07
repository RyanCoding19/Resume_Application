# train_model.py (Model Training with MLflow)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_data
from sklearn.preprocessing import MaxAbsScaler
import joblib
import pandas as pd
import os
import mlflow
import mlflow.sklearn

# Load data
dataset = pd.read_csv("data/resume.csv")

# Preprocess text data
X, tfidf = preprocess_data(dataset['Resume_str'])
y = dataset['Category']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale for Logistic Regression
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure models directory exists
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# --- Logistic Regression + MLflow ---
param_grid_lr = {
    'C': [0.01, 0.1, 1, 5, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter': [500, 1000, 1500]
}

with mlflow.start_run(run_name="Logistic Regression"):
    grid_search_lr = GridSearchCV(LogisticRegression(), param_grid=param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_lr.fit(X_train_scaled, y_train)
    best_lr_model = grid_search_lr.best_estimator_

    y_pred_test_lr = best_lr_model.predict(X_test_scaled)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_params(grid_search_lr.best_params_)
    mlflow.log_metric("train_accuracy", best_lr_model.score(X_train_scaled, y_train))
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test_lr))
    mlflow.sklearn.log_model(best_lr_model, "logistic_regression_model")

    joblib.dump(best_lr_model, os.path.join(models_dir, 'resume_classifier_lr.pkl'))
    joblib.dump(tfidf, os.path.join(models_dir, 'tfidf_vectorizer_lr.pkl'))

# --- Random Forest + MLflow ---
param_grid_rf = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [2, 4, 6],
    'class_weight': ['balanced', None]
}

with mlflow.start_run(run_name="Random Forest"):
    grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_

    y_pred_test_rf = best_rf_model.predict(X_test)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_params(grid_search_rf.best_params_)
    mlflow.log_metric("train_accuracy", best_rf_model.score(X_train, y_train))
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test_rf))
    mlflow.sklearn.log_model(best_rf_model, "random_forest_model")

    joblib.dump(best_rf_model, os.path.join(models_dir, 'resume_classifier_rf.pkl'))
    joblib.dump(tfidf, os.path.join(models_dir, 'tfidf_vectorizer_rf.pkl'))

# --- BERT (Optional placeholder for fine-tuning later) ---
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(dataset['Category'].unique()))
inputs = tokenizer(dataset['Resume_str'].tolist(), padding=True, truncation=True, return_tensors="pt")
