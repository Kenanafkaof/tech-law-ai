import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load and preprocess data
def preprocess_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df_model = df[['full_text', 'court', 'judge', 'citing_cases', 'cited_by', 'tech_relevance_score']].dropna()
    
    # Ensure text data is string type
    df_model['full_text'] = df_model['full_text'].astype(str)
    
    # Process citation counts
    df_model['citing_cases_count'] = df_model['citing_cases'].apply(
        lambda x: len(eval(x)) if isinstance(x, str) else 0
    )
    df_model['cited_by_count'] = df_model['cited_by'].apply(
        lambda x: len(eval(x)) if isinstance(x, str) else 0
    )
    
    return df_model

# Train model
def train_model(df_model):
    # Encode categorical variables
    label_enc_court = LabelEncoder()
    label_enc_judge = LabelEncoder()
    
    df_model['court_encoded'] = label_enc_court.fit_transform(df_model['court'].astype(str))
    df_model['judge_encoded'] = label_enc_judge.fit_transform(df_model['judge'].astype(str))
    
    # Prepare features and target
    X_text = df_model['full_text']
    X_other = df_model[['court_encoded', 'judge_encoded', 'citing_cases_count', 'cited_by_count']]
    y = df_model['tech_relevance_score'].astype(int)
    
    # Split data
    X_train_text, X_test_text, X_train_other, X_test_other, y_train, y_test = train_test_split(
        X_text, X_other, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and fit TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    
    # Combine with other features
    X_train_combined = np.hstack((
        X_train_tfidf.toarray(),
        X_train_other.values
    ))
    X_test_combined = np.hstack((
        X_test_tfidf.toarray(),
        X_test_other.values
    ))
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_combined, y_train)
    
    return model, tfidf_vectorizer, label_enc_court, label_enc_judge

if __name__ == "__main__":
    # Process data
    df_model = preprocess_data("../../datasets/test_all_tech_cases_fixed.csv")
    
    # Train model
    model, tfidf_vectorizer, label_enc_court, label_enc_judge = train_model(df_model)
    
    # Save artifacts
    joblib.dump(model, "legal_case_model.pkl")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(label_enc_court, "court_label_encoder.pkl")
    joblib.dump(label_enc_judge, "judge_label_encoder.pkl")
    
    print("Model and preprocessing artifacts saved successfully.")