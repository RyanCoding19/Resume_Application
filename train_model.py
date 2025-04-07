# model_training.py (Model Training)
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

# Load data (for example from CSV)
dataset = pd.read_csv("data/resume.csv")

# Preprocess text data
X, tfidf = preprocess_data(dataset['Resume_str'])
y = dataset['Category']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale the data to prevent overfitting in Logistic Regression ---
scaler = MaxAbsScaler()  # Using MaxAbsScaler to handle sparse matrices
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test_scaled = scaler.transform(X_test)  # Only transform on test data

# --- Logistic Regression Hyperparameter Tuning ---
param_grid_lr = {
    'C': [0.01, 0.1, 1, 5, 10],  # Regularization strength (larger C means weaker regularization)
    'solver': ['liblinear', 'saga'],  # Trying different solvers
    'max_iter': [500, 1000, 1500]  # Max iterations for convergence
}

grid_search_lr = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr.fit(X_train_scaled, y_train)  # Use scaled data

# Get the best Logistic Regression model
best_lr_model = grid_search_lr.best_estimator_

# --- Evaluate on Training Data ---
train_accuracy_lr = best_lr_model.score(X_train_scaled, y_train)
print(f"Training Accuracy for Logistic Regression: {train_accuracy_lr}")

# --- Evaluate on Test Data ---
test_accuracy_lr = best_lr_model.score(X_test_scaled, y_test)
print(f"Test Accuracy for Logistic Regression: {test_accuracy_lr}")

# --- Classification Report for Training Data ---
y_pred_train_lr = best_lr_model.predict(X_train_scaled)
print(f"Classification Report for Logistic Regression (Training Data):\n {classification_report(y_train, y_pred_train_lr)}")

# --- Classification Report for Test Data ---
y_pred_test_lr = best_lr_model.predict(X_test_scaled)
print(f"Classification Report for Logistic Regression (Test Data):\n {classification_report(y_test, y_pred_test_lr)}")


# --- Random Forest Hyperparameter Tuning ---
param_grid_rf = {
    'n_estimators': [100, 150, 200],  # Number of trees
    'max_depth': [10, 20, 30, None],  # Tree depths
    'min_samples_split': [10, 20],  # Increased to avoid splitting nodes too easily and overfitting
    'min_samples_leaf': [2, 4, 6],  # Increased to avoid creating leaf nodes with too few samples
    'class_weight': ['balanced', None]  # Adding class weight to address class imbalance
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)  # No need to scale for Random Forest

# Get the best Random Forest model
best_rf_model = grid_search_rf.best_estimator_

# --- Evaluate on Training Data ---
train_accuracy_rf = best_rf_model.score(X_train, y_train)
print(f"Training Accuracy for Random Forest: {train_accuracy_rf}")

# --- Evaluate on Test Data ---
test_accuracy_rf = best_rf_model.score(X_test, y_test)
print(f"Test Accuracy for Random Forest: {test_accuracy_rf}")

# --- Classification Report for Training Data ---
y_pred_train_rf = best_rf_model.predict(X_train)
print(f"Classification Report for Random Forest (Training Data):\n {classification_report(y_train, y_pred_train_rf)}")

# --- Classification Report for Test Data ---
y_pred_test_rf = best_rf_model.predict(X_test)
print(f"Classification Report for Random Forest (Test Data):\n {classification_report(y_test, y_pred_test_rf)}")

# --- Save the tuned models ---
# Ensure the 'models' directory exists
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)  # Create the 'models' directory if it doesn't exist (I keep getting an error for this)

# Save models to the 'models' directory
joblib.dump(best_lr_model, os.path.join(models_dir, 'resume_classifier_lr.pkl'))
joblib.dump(tfidf, os.path.join(models_dir, 'tfidf_vectorizer_lr.pkl'))
joblib.dump(best_rf_model, os.path.join(models_dir, 'resume_classifier_rf.pkl'))
joblib.dump(tfidf, os.path.join(models_dir, 'tfidf_vectorizer_rf.pkl'))

# BERT (Advanced model) - Optional, to be done later for fine-tuning
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(dataset['Category'].unique()))
inputs = tokenizer(dataset['Resume_str'].tolist(), padding=True, truncation=True, return_tensors="pt")
