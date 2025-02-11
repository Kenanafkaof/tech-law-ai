import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "../datasets/test_all_tech_cases_5year.csv"
df = pd.read_csv(file_path)

# Selecting relevant columns
df_filtered = df[['full_text', 'precedential_status']].dropna()

# Encoding the target variable (precedential_status)
df_filtered['precedential_status'] = df_filtered['precedential_status'].astype('category').cat.codes

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_filtered['full_text'], 
                                                    df_filtered['precedential_status'], 
                                                    test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_tfidf, y_train)

# Predictions
y_pred = clf.predict(X_test_tfidf)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', report)

# Function to classify a new case
def classify_case(case_text):
    case_tfidf = vectorizer.transform([case_text])
    prediction = clf.predict(case_tfidf)
    return prediction[0]

# Example usage
case_input = "I have a case regarding intellectual property in digital assets"
prediction_result = classify_case(case_input)
print(f'The predicted classification for the case is: {prediction_result}')
