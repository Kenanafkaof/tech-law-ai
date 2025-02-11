from patent_analysis import PatentNLPInterface
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Dict

def add_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add network features to the training data."""
    # Initialize network features with zeros
    network_features = {
        'citation_network_in_degree': 0,
        'citation_network_out_degree': 0,
        'citation_network_pagerank': 0,
        'citation_network_betweenness': 0
    }
    
    # Add network features to each row
    for feature, value in network_features.items():
        df[feature] = value
        
    return df


class ConsistentFeaturePatentNLPInterface(PatentNLPInterface):
    """Enhanced PatentNLPInterface with consistent feature handling"""
    
    def _convert_to_structured_format(self, parsed_info: Dict) -> Dict:
        """Convert parsed information to format expected by PatentClaimAnalyzer with all required features."""
        # Create passage location text based on identified rejection types
        passage_locations = []
        for rejection in parsed_info['rejection_types']:
            passage_locations.append(f"c. {rejection}")
        
        # Create complete feature set including network features
        structured_data = {
            'examinerCitedReferenceIndicator': 'FALSE',
            'applicantCitedExaminerReferenceIndicator': 'FALSE',
            'nplIndicator': 'FALSE',
            'techCenter': self._map_technical_field(parsed_info['technical_field']),
            'passageLocationText': str(passage_locations),
            'publicationNumber': None,
            'citedDocumentIdentifier': None,
            # Add required network features
            'citation_network_in_degree': 0,
            'citation_network_out_degree': 0,
            'citation_network_pagerank': 0,
            'citation_network_betweenness': 0
        }
        
        return structured_data

def train_and_save_model():
    """Train and save the patent analysis model with proper feature initialization."""
    try:
        # Load data
        df = pd.read_csv('../datasets/patent_data/patent_data_20250205_231903_combined_patents_training.csv')
        
        # Create target variable
        def has_rejection_type(text, rejection_type):
            if pd.isna(text):
                return False
            return f'c. {rejection_type}' in text or f'c.{rejection_type}' in text
        
        y = df['passageLocationText'].apply(lambda x: 
            has_rejection_type(x, '103') or has_rejection_type(x, '102')
        )
        
        # Select and prepare features
        feature_columns = [
            'publicationNumber', 'citedDocumentIdentifier',
            'examinerCitedReferenceIndicator', 'applicantCitedExaminerReferenceIndicator',
            'nplIndicator', 'techCenter', 'passageLocationText'
        ]
        X = df[feature_columns].copy()
        
        # Initialize network features
        network_features = {
            'citation_network_in_degree': np.zeros(len(X)),
            'citation_network_out_degree': np.zeros(len(X)),
            'citation_network_pagerank': np.zeros(len(X)),
            'citation_network_betweenness': np.zeros(len(X))
        }
        
        for feature, values in network_features.items():
            X[feature] = values
        
        # Initialize and train with custom interface
        nlp_interface = ConsistentFeaturePatentNLPInterface()
        nlp_interface.patent_analyzer.fit(X, y)
        
        # Save model
        model_path = "patent_nlp_model.joblib"
        nlp_interface.save(model_path)
        print(f"Model saved to: {model_path}")
        
        return model_path
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise




if __name__ == "__main__":
    train_and_save_model()