import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import ast
from typing import Tuple
import re

class TechLawDataPrep:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.text_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=10,
            max_df=0.9
        )
        self.label_encoders = {}

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the vectorizer on training data and transform it."""
        # Fit and transform text features
        text = df['text_excerpt'].fillna('') + ' ' + df['case_name'].fillna('')
        text = text.apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x.lower()))
        text_features = self.text_vectorizer.fit_transform(text)
        
        # Process categorical features (if any fitting is needed, do it here)
        categorical_features = pd.DataFrame()
        categorical_features['court_level'] = df['court_id'].map(
            lambda x: {
                'cafc': 2,
                'ca1': 2, 'ca2': 2, 'ca3': 2, 'ca4': 2, 'ca5': 2,
                'ca6': 2, 'ca7': 2, 'ca8': 2, 'ca9': 2, 'ca10': 2,
                'ca11': 2, 'cadc': 2,
                'scotus': 3,
                'dcd': 1, 'nysd': 1, 'cand': 1, 'txed': 1
            }.get(x, 1)
        )
        
        # Process numerical features
        numerical_features = pd.DataFrame()
        numerical_features['text_length'] = df['text_excerpt'].str.len()
        numerical_features['title_length'] = df['case_name'].str.len()
        numerical_features['citation_count'] = pd.to_numeric(
            df['citation_count'], errors='coerce'
        ).fillna(0)
        numerical_features['year'] = pd.to_datetime(
            df['date_filed']
        ).dt.year.fillna(2020)
        
        # Combine features
        feature_matrix = np.hstack([
            text_features.toarray(),
            categorical_features.values,
            numerical_features.values
        ])
        
        # Create target variable
        target = (df['tech_relevance_score'] >= 0.2).astype(int)
        return feature_matrix, target

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform new data using the already-fitted vectorizer."""
        text = df['text_excerpt'].fillna('') + ' ' + df['case_name'].fillna('')
        text = text.apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x.lower()))
        text_features = self.text_vectorizer.transform(text)
        
        categorical_features = pd.DataFrame()
        categorical_features['court_level'] = df['court_id'].map(
            lambda x: {
                'cafc': 2,
                'ca1': 2, 'ca2': 2, 'ca3': 2, 'ca4': 2, 'ca5': 2,
                'ca6': 2, 'ca7': 2, 'ca8': 2, 'ca9': 2, 'ca10': 2,
                'ca11': 2, 'cadc': 2,
                'scotus': 3,
                'dcd': 1, 'nysd': 1, 'cand': 1, 'txed': 1
            }.get(x, 1)
        )
        
        numerical_features = pd.DataFrame()
        numerical_features['text_length'] = df['text_excerpt'].str.len()
        numerical_features['title_length'] = df['case_name'].str.len()
        numerical_features['citation_count'] = pd.to_numeric(
            df['citation_count'], errors='coerce'
        ).fillna(0)
        numerical_features['year'] = pd.to_datetime(
            df['date_filed']
        ).dt.year.fillna(2020)
        
        feature_matrix = np.hstack([
            text_features.toarray(),
            categorical_features.values,
            numerical_features.values
        ])
        
        target = (df['tech_relevance_score'] >= 0.2).astype(int)
        return feature_matrix, target
    
    def infer_precedential_status(self) -> pd.Series:
        """Infer precedential status from available metadata."""
        def infer_status(row):
            # Federal Circuit cases are usually precedential
            if row['court_id'] == 'cafc':
                return 'Precedential'
            # Published status from the data
            elif row['precedential_status'] == 'Published':
                return 'Precedential'
            # Cases with high citation counts tend to be precedential
            elif row['citation_count'] > 5:
                return 'Precedential'
            # Default to non-precedential if unsure
            else:
                return 'Non-Precedential'
        
        return self.df.apply(infer_status, axis=1)
    
    def extract_tech_keywords(self) -> pd.Series:
        """Convert string representation of keywords to actual list."""
        def parse_keywords(kw_str):
            try:
                # Safely evaluate string representation of list
                if pd.isna(kw_str):
                    return []
                kw_list = ast.literal_eval(kw_str)
                # Extract just the keywords without primary/secondary prefix
                return [k.split(':')[1] for k in kw_list]
            except:
                return []
        
        return self.df['tech_keywords_found'].apply(parse_keywords)
    
    def prepare_text_features(self) -> np.ndarray:
        """Create text features focusing on document structure."""
        # Combine case name and excerpt
        text = self.df['text_excerpt'].fillna('') + ' ' + self.df['case_name'].fillna('')
        
        # Clean text
        text = text.apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x.lower()))
        
        # Transform to TF-IDF features
        return self.text_vectorizer.fit_transform(text)

    
    def prepare_categorical_features(self) -> pd.DataFrame:
        """Encode categorical features properly."""
        categorical_features = pd.DataFrame()
        
        # Encode court type
        court_levels = {
            'cafc': 2,  # Appeal
            'ca1': 2, 'ca2': 2, 'ca3': 2, 'ca4': 2, 'ca5': 2,
            'ca6': 2, 'ca7': 2, 'ca8': 2, 'ca9': 2, 'ca10': 2,
            'ca11': 2, 'cadc': 2,
            'scotus': 3,  # Supreme
            'dcd': 1, 'nysd': 1, 'cand': 1, 'txed': 1  # District
        }
        
        # Convert court IDs to numerical levels
        categorical_features['court_level'] = self.df['court_id'].map(
            lambda x: court_levels.get(x, 1)  # Default to district court level
        )
        
        return categorical_features

    
    def prepare_numerical_features(self) -> pd.DataFrame:
        """Prepare numerical features."""
        numerical_features = pd.DataFrame()
        
        # Text length features
        numerical_features['text_length'] = self.df['text_excerpt'].str.len()
        numerical_features['title_length'] = self.df['case_name'].str.len()
        
        # Citation count
        numerical_features['citation_count'] = pd.to_numeric(
            self.df['citation_count'], errors='coerce'
        ).fillna(0)
        
        # Year features
        numerical_features['year'] = pd.to_datetime(
            self.df['date_filed']
        ).dt.year.fillna(2020)
        
        return numerical_features

    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare full feature matrix and target variable."""
        print("Starting data preparation...")
        
        print("Preparing text features...")
        text_features = self.prepare_text_features()
        
        print("Preparing categorical features...")
        categorical_features = self.prepare_categorical_features()
        
        print("Preparing numerical features...")
        numerical_features = self.prepare_numerical_features()
        
        # Combine features
        print("Combining features...")
        feature_matrix = np.hstack([
            text_features.toarray(),
            categorical_features.values,
            numerical_features.values
        ])
        
        # Convert target to numerical
        target = (self.df['tech_relevance_score'] >= 0.2).astype(int)
        
        print(f"Final feature matrix shape: {feature_matrix.shape}")
        print(f"Number of positive cases: {sum(target)}")
        print(f"Number of negative cases: {len(target) - sum(target)}")
        
        return feature_matrix, target


    
    def get_feature_names(self) -> list:
        text_features = self.text_vectorizer.get_feature_names_out().tolist()
        categorical_features = ['court_level']
        numerical_features = ['text_length', 'title_length', 'citation_count', 'year']
        return text_features + categorical_features + numerical_features


