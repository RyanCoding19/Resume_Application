# model_training.py (Model Training)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_data
import joblib
import pandas as pd

# Load data (for example from CSV)
dataset = pd.read_csv("data/resume.csv")

# Preprocess text data
X, tfidf = preprocess_data(dataset['Resume_str'])
y = dataset['Category']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"Logistic Regression Accuracy: {lr.score(X_test, y_test)}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {rf.score(X_test, y_test)}")

# Save the models
joblib.dump(lr, 'models/resume_classifier_lr.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer_lr.pkl')

# Optionally, save Random Forest model as well
joblib.dump(rf, 'models/resume_classifier_rf.pkl')

# BERT (Advanced model)
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(dataset['Category'].unique()))
inputs = tokenizer(dataset['Resume_str'].tolist(), padding=True, truncation=True, return_tensors="pt")

