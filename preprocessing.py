import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy NER model for extracting named entities (skills, experience, etc.)
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Preprocess resume text by cleaning and normalizing."""
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'\W', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    return text

def extract_entities(text):
    """Extract named entities such as skills, experience, etc. using spaCy."""
    doc = nlp(text)
    # Extract named entities related to organizations, skills, and time
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "SKILL", "TIME", "DATE"]]
    return entities

def preprocess_data(resumes):
    """Preprocess and vectorize resumes using TF-IDF."""
    cleaned_resumes = [clean_text(resume) for resume in resumes]
    # Extract features using TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(cleaned_resumes)
    return X, tfidf
