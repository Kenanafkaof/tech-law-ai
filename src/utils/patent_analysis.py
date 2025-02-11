from transformers import AutoTokenizer, AutoModel
import torch
import spacy
import re
from typing import Dict, List, Optional, Union
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import joblib
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import networkx as nx
import pandas as pd
from tqdm import tqdm
import logging
import sys
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('patent_analyzer.log')
    ]
)
logger = logging.getLogger(__name__)

class PatentFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer for extracting patent features."""
    
    def __init__(self):
        self.citation_graph = nx.DiGraph()
        self.network_metrics_cache = {}
        
    def fit(self, X, y=None):
        """Build citation network during fit."""
        logger.info("Building citation network for feature extraction...")
        for _, row in tqdm(X.iterrows(), total=len(X), desc="Building citation network"):
            if pd.notna(row['publicationNumber']) and pd.notna(row['citedDocumentIdentifier']):
                self.citation_graph.add_edge(row['publicationNumber'], row['citedDocumentIdentifier'])
        
        # Precompute network metrics
        logger.info("Precomputing network metrics...")
        self._precompute_network_metrics()
        return self
        
    def _precompute_network_metrics(self):
        """Precompute all network metrics for efficiency."""
        pagerank = nx.pagerank(self.citation_graph)
        betweenness = nx.betweenness_centrality(self.citation_graph)
        in_degrees = dict(self.citation_graph.in_degree())
        out_degrees = dict(self.citation_graph.out_degree())
        
        for node in self.citation_graph.nodes():
            self.network_metrics_cache[node] = {
                'in_degree': in_degrees.get(node, 0),
                'out_degree': out_degrees.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'betweenness': betweenness.get(node, 0)
            }
    
    def _extract_passage_features(self, locations: str) -> Dict:
        """Extract detailed features from passage locations."""
        if pd.isna(locations):
            return {
                'num_passages': 0,
                'num_claims_cited': 0,
                'num_paragraphs_cited': 0,
                'num_figures_cited': 0,
                'has_spec_support': False
            }
            
        locations = str(locations).strip("[]'")
        passages = [loc.strip() for loc in locations.split('|')]
        
        return {
            'num_passages': len(passages),
            'num_claims_cited': sum(1 for loc in passages if 'claim' in loc.lower()),
            'num_paragraphs_cited': sum(1 for loc in passages if 'paragraph' in loc.lower() or 'par.' in loc.lower()),
            'num_figures_cited': sum(1 for loc in passages if 'figure' in loc.lower() or 'fig.' in loc.lower()),
            'has_spec_support': any('paragraph' in loc.lower() or 'par.' in loc.lower() for loc in passages)
        }
    
    def _get_network_features(self, pub_num: str) -> Dict:
        """Get network features for a publication."""
        if pd.isna(pub_num) or pub_num not in self.network_metrics_cache:
            return {
                'citation_network_in_degree': 0,
                'citation_network_out_degree': 0,
                'citation_network_pagerank': 0,
                'citation_network_betweenness': 0
            }
            
        metrics = self.network_metrics_cache[pub_num]
        return {
            'citation_network_in_degree': metrics['in_degree'],
            'citation_network_out_degree': metrics['out_degree'],
            'citation_network_pagerank': metrics['pagerank'],
            'citation_network_betweenness': metrics['betweenness']
        }
        
    def transform(self, X):
        """Transform patent data into feature matrix."""
        logger.info("Extracting features from patent data...")
        features_list = []
        
        for _, row in tqdm(X.iterrows(), total=len(X), desc="Extracting features"):
            # Basic indicators
            features = {
                'is_examiner_cited': row['examinerCitedReferenceIndicator'] == 'TRUE',
                'is_applicant_cited': row['applicantCitedExaminerReferenceIndicator'] == 'TRUE',
                'is_npl': row['nplIndicator'] == 'TRUE',
                'tech_center': int(row['techCenter']) if pd.notna(row['techCenter']) else 0,
            }
            
            # Passage-based features
            passage_features = self._extract_passage_features(row['passageLocationText'])
            features.update(passage_features)
            
            # Network-based features
            if pd.notna(row['publicationNumber']):
                network_features = self._get_network_features(row['publicationNumber'])
                features.update(network_features)
            
            features_list.append(features)
            
        return pd.DataFrame(features_list)


class PatentClaimAnalyzer:
    """
    A model for analyzing patent claims and predicting their likelihood of approval
    based on historical patent data, citation patterns, and office actions.
    """
    
    def __init__(self, use_xgboost=False):
        logger.info("Initializing PatentClaimAnalyzer...")
        
        # Initialize feature extractor and preprocessors
        self.feature_extractor = PatentFeatureExtractor()
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.scaler = StandardScaler()
        
        # Choose classifier
        try:
            if use_xgboost:
                from xgboost import XGBClassifier
                base_classifier = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                logger.info("Using XGBoost classifier")
            else:
                from sklearn.ensemble import RandomForestClassifier
                base_classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=4,
                    class_weight='balanced',
                    random_state=42
                )
                logger.info("Using Random Forest classifier")
        except ImportError:
            logger.warning("XGBoost not available, falling back to Random Forest")
            from sklearn.ensemble import RandomForestClassifier
            base_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42
            )
        
        # Create full pipeline with imputation before SMOTE
        self.pipeline = ImbPipeline([
            ('extractor', self.feature_extractor),
            ('imputer', self.imputer),
            ('scaler', self.scaler),
            ('sampler', SMOTE(random_state=42)),
            ('classifier', CalibratedClassifierCV(base_classifier, cv=5))
        ])
        
        logger.info("PatentClaimAnalyzer initialized successfully")


    
    def build_citation_network(self, df: pd.DataFrame) -> None:
        """Build a directed graph of patent citations."""
        logger.info("Building citation network...")
        edge_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building citation network"):
            if pd.notna(row['publicationNumber']) and pd.notna(row['citedDocumentIdentifier']):
                self.citation_graph.add_edge(
                    row['publicationNumber'],
                    row['citedDocumentIdentifier']
                )
                edge_count += 1
        
        logger.info(f"Citation network built with {edge_count} edges and "
                   f"{len(self.citation_graph.nodes)} nodes")

    def precompute_network_metrics(self) -> Dict[str, Dict]:
        """Precompute network metrics for all nodes."""
        logger.info("Precomputing network metrics...")
        
        # Precompute metrics in batch
        pagerank = nx.pagerank(self.citation_graph)
        betweenness = nx.betweenness_centrality(self.citation_graph)
        in_degrees = dict(self.citation_graph.in_degree())
        out_degrees = dict(self.citation_graph.out_degree())
        
        # Combine all metrics
        metrics = {}
        for node in self.citation_graph.nodes():
            metrics[node] = {
                'in_degree': in_degrees.get(node, 0),
                'out_degree': out_degrees.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'betweenness': betweenness.get(node, 0)
            }
            
        logger.info("Network metrics precomputed")
        return metrics
    
    def preprocess_passage_locations(self, locations: str) -> List[str]:
        """Extract and clean passage locations from the raw string."""
        if pd.isna(locations):
            return []
        # Remove brackets and split by pipe
        locations = locations.strip("[]'")
        return [loc.strip() for loc in locations.split('|')]

    def extract_claim_features(self, row: pd.Series) -> Dict:
        """Extract relevant features from a patent claim row."""
        features = {
            'num_citations': 1,  # Base count for this citation
            'num_passage_locations': 0,
            'has_examiner_citation': row['examinerCitedReferenceIndicator'] == 'TRUE',
            'has_applicant_citation': row['applicantCitedExaminerReferenceIndicator'] == 'TRUE',
            'is_npl': row['nplIndicator'] == 'TRUE',
            'tech_center': int(row['techCenter']) if pd.notna(row['techCenter']) else 0,
        }
        
        # Process passage locations
        passage_locs = self.preprocess_passage_locations(row['passageLocationText'])
        features['num_passage_locations'] = len(passage_locs)
        
        # Extract specific types of citations
        features.update({
            'has_103_rejection': any('103' in loc for loc in passage_locs),
            'has_102_rejection': any('102' in loc for loc in passage_locs),
            'has_112_rejection': any('112' in loc for loc in passage_locs),
            'num_claim_citations': sum(1 for loc in passage_locs if 'claim' in loc.lower()),
            'num_paragraph_citations': sum(1 for loc in passage_locs if 'paragraph' in loc.lower() or 'par.' in loc.lower()),
            'num_figure_citations': sum(1 for loc in passage_locs if 'figure' in loc.lower() or 'fig.' in loc.lower())
        })
        
        return features

    def calculate_network_metrics(self, publication_number: str) -> Dict:
        """Calculate network centrality metrics for a given patent."""
        if publication_number not in self.citation_graph:
            return {
                'in_degree': 0,
                'out_degree': 0,
                'pagerank': 0,
                'betweenness': 0
            }
            
        return {
            'in_degree': self.citation_graph.in_degree(publication_number),
            'out_degree': self.citation_graph.out_degree(publication_number),
            'pagerank': nx.pagerank(self.citation_graph).get(publication_number, 0),
            'betweenness': nx.betweenness_centrality(self.citation_graph).get(publication_number, 0)
        }
    
    def generate_recommendations(self, features: Dict, rejection_prob: float) -> List[str]:
        """Generate recommendations based on claim analysis."""
        recommendations = []
        
        if rejection_prob > 0.7:
            recommendations.append("High risk of rejection. Consider reviewing prior art thoroughly.")
            
            if features['has_103_rejection']:
                recommendations.append("Potential obviousness issues detected. Consider strengthening non-obviousness arguments.")
            
            if features['has_102_rejection']:
                recommendations.append("Potential novelty issues detected. Review cited prior art carefully.")
                
        if features['num_claim_citations'] > 5:
            recommendations.append("Multiple claim citations detected. Consider narrowing claim scope.")
            
        if features['num_paragraph_citations'] == 0:
            recommendations.append("No paragraph citations found. Consider adding more detailed support from specification.")
            
        return recommendations

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """Train the model with cross-validation."""
        logger.info(f"Starting model training with {len(X)} samples")
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='f1')
        
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Fit final model
        logger.info("Training final model...")
        self.pipeline.fit(X, y)
        logger.info("Model training completed successfully")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get calibrated probabilities for predictions."""
        return self.pipeline.predict_proba(X)

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        return obj

    def analyze_claim(self, claim_data: Dict) -> Dict:
        """Analyze a patent claim with confidence calibration."""
        logger.info("Starting claim analysis...")
        
        required_features = {
            'citation_network_in_degree': 0,
            'citation_network_out_degree': 0,
            'citation_network_pagerank': 0,
            'citation_network_betweenness': 0
        }
        
        # Update claim_data with any missing required features
        for key, default_value in required_features.items():
            if key not in claim_data:
                claim_data[key] = default_value
        
        # Convert single claim to DataFrame
        claim_df = pd.DataFrame([claim_data])
        
        # Get calibrated probability
        proba = self.predict_proba(claim_df)[0, 1]
        logger.info(f"Calibrated rejection probability: {proba:.3f}")

        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            importance = self.pipeline.named_steps['classifier'].feature_importances_
            feature_names = self.pipeline.named_steps['features'].get_feature_names_out()
            feature_importance = dict(zip(feature_names, importance))
        
        result = {
            'rejection_probability': float(proba),
            'confidence_level': self._get_confidence_level(proba),
            'feature_importance': feature_importance,
            'recommendations': self._generate_recommendations(claim_data, proba)
        }
        
        logger.info("Claim analysis completed")
        return result
        
    def _precompute_network_metrics(self):
        """Precompute all network metrics for efficiency."""
        pagerank = nx.pagerank(self.citation_graph)
        betweenness = nx.betweenness_centrality(self.citation_graph)
        in_degrees = dict(self.citation_graph.in_degree())
        out_degrees = dict(self.citation_graph.out_degree())
        
        for node in self.citation_graph.nodes():
            self.network_metrics_cache[node] = {
                'in_degree': in_degrees.get(node, 0),
                'out_degree': out_degrees.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'betweenness': betweenness.get(node, 0)
            }
    
    def _extract_passage_features(self, locations: str) -> Dict:
        """Extract detailed features from passage locations."""
        if pd.isna(locations):
            return {
                'num_passages': 0,
                'num_claims_cited': 0,
                'num_paragraphs_cited': 0,
                'num_figures_cited': 0,
                'has_spec_support': False
            }
            
        locations = str(locations).strip("[]'")
        passages = [loc.strip() for loc in locations.split('|')]
        
        return {
            'num_passages': len(passages),
            'num_claims_cited': sum(1 for loc in passages if 'claim' in loc.lower()),
            'num_paragraphs_cited': sum(1 for loc in passages if 'paragraph' in loc.lower() or 'par.' in loc.lower()),
            'num_figures_cited': sum(1 for loc in passages if 'figure' in loc.lower() or 'fig.' in loc.lower()),
            'has_spec_support': any('paragraph' in loc.lower() or 'par.' in loc.lower() for loc in passages)
        }
    
    def _get_network_features(self, pub_num: str) -> Dict:
        """Get network features for a publication."""
        if pd.isna(pub_num) or pub_num not in self.network_metrics_cache:
            return {
                'citation_network_in_degree': 0,
                'citation_network_out_degree': 0,
                'citation_network_pagerank': 0,
                'citation_network_betweenness': 0
            }
            
        metrics = self.network_metrics_cache[pub_num]
        return {
            'citation_network_in_degree': metrics['in_degree'],
            'citation_network_out_degree': metrics['out_degree'],
            'citation_network_pagerank': metrics['pagerank'],
            'citation_network_betweenness': metrics['betweenness']
        }
        
    def transform(self, X):
        """Transform patent data into feature matrix."""
        logger.info("Extracting features from patent data...")
        features_list = []
        
        for _, row in tqdm(X.iterrows(), total=len(X), desc="Extracting features"):
            # Basic indicators
            features = {
                'is_examiner_cited': row['examinerCitedReferenceIndicator'] == 'TRUE',
                'is_applicant_cited': row['applicantCitedExaminerReferenceIndicator'] == 'TRUE',
                'is_npl': row['nplIndicator'] == 'TRUE',
                'tech_center': int(row['techCenter']) if pd.notna(row['techCenter']) else 0,
            }
            
            # Passage-based features
            passage_features = self._extract_passage_features(row['passageLocationText'])
            features.update(passage_features)
            
            # Network-based features
            if pd.notna(row['publicationNumber']):
                network_features = self._get_network_features(row['publicationNumber'])
                features.update(network_features)
            
            features_list.append(features)
            
        return pd.DataFrame(features_list)

    def _get_confidence_level(self, probability: float) -> str:
        """Convert probability to confidence level."""
        if probability > 0.8:
            return "High"
        elif probability > 0.6:
            return "Medium"
        return "Low"
    
    def _generate_recommendations(self, claim_data: Dict, prob: float) -> List[str]:
        """Generate detailed recommendations based on analysis."""
        recommendations = []
        
        # Base recommendations on probability
        if prob > 0.7:
            recommendations.append("High risk of rejection - detailed review recommended")
        elif prob > 0.5:
            recommendations.append("Moderate risk - consider strengthening claim language")
        
        # Add specific recommendations based on features
        passage_features = self.pipeline.named_steps['features'].named_steps['patent_features']._extract_passage_features(
            claim_data.get('passageLocationText', '')
        )
        
        if not passage_features['has_spec_support']:
            recommendations.append("Add more detailed support from specification")
        
        if passage_features['num_claims_cited'] > 5:
            recommendations.append("High number of claim citations - consider narrowing scope")
        
        return recommendations
    
    def save_model(self, filepath=None):
        """Save the trained model to disk."""
        if filepath is None:
            # Create default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"patent_analyzer_model_{timestamp}.joblib"
            
        logger.info(f"Saving model to {filepath}")
        joblib.dump(self.pipeline, filepath)
        logger.info("Model saved successfully")
        
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from disk."""
        logger.info(f"Loading model from {filepath}")
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        logger.info("Model loaded successfully")
        return instance



class PatentNLPInterface:
    """
    Natural language interface for patent claim analysis.
    Handles user queries and converts them to structured patent analysis.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the NLP interface.
        
        Args:
            model_path: Optional path to saved PatentClaimAnalyzer model
        """
        try:
            # Load SpaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Load BERT model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            logger.info("Falling back to basic NLP processing...")


        self.section_patterns = {
            "103": ["obvious", "obviousness", "combine", "modification"],
            "102": ["novelty", "anticipated", "prior art", "identical"],
            "112": ["indefinite", "written description", "enablement", "clarity"]
        }
        # Load or create patent analyzer
        if model_path:
            self.patent_analyzer = PatentClaimAnalyzer.load_model(model_path)
        else:
            self.patent_analyzer = PatentClaimAnalyzer()

    
    def parse_user_query(self, query: str) -> Dict:
        """
        Parse natural language query into structured patent information.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary containing structured patent information
        """
        doc = self.nlp(query.lower())
        
        # Extract key information
        info = {
            'query_type': self._determine_query_type(doc),
            'technical_field': self._extract_technical_field(doc),
            'claim_text': self._extract_claim_text(doc),
            'cited_references': self._extract_references(doc),
            'rejection_types': self._identify_rejection_types(doc)
        }
        
        return info
    
    def _determine_query_type(self, doc) -> str:
        """Determine the type of analysis the user is requesting."""
        query_types = {
            'patentability': ['patentable', 'patent', 'protect'],
            'validity': ['valid', 'invalid', 'challenge'],
            'infringement': ['infringe', 'copying', 'similar'],
            'general': ['analyze', 'review', 'check']
        }
        
        for query_type, keywords in query_types.items():
            if any(keyword in doc.text.lower() for keyword in keywords):
                return query_type
        
        return 'general'
    
    def _extract_technical_field(self, doc) -> str:
        """Extract the technical field from the query."""
        # Look for technical field indicators
        tech_indicators = ['technology', 'field', 'industry', 'domain', 'area']
        
        for token in doc:
            if token.text in tech_indicators:
                # Look for the next noun phrase
                for chunk in doc.noun_chunks:
                    if token.i < chunk.start:
                        return chunk.text
        
        return None
    
    def _extract_claim_text(self, doc) -> str:
        """Extract potential claim text from the query."""
        # Look for claim indicators
        claim_indicators = ['claim', 'invention', 'system', 'method', 'device']
        
        for token in doc:
            if token.text in claim_indicators:
                # Get the surrounding context
                start_idx = max(0, token.i - 2)
                end_idx = min(len(doc), token.i + 15)
                return doc[start_idx:end_idx].text
        
        return None
    
    def _extract_references(self, doc) -> List[str]:
        """Extract mentioned references or prior art."""
        references = []
        
        # Look for patent numbers
        patent_pattern = re.compile(r'US\d{7,8}|US\d{4}/\d+|\d{7,8}')
        matches = patent_pattern.findall(doc.text)
        references.extend(matches)
        
        # Look for company or product names
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                references.append(ent.text)
                
        return references
    
    def _convert_to_structured_format(self, parsed_info: Dict) -> Dict:
        """Convert parsed information to format expected by PatentClaimAnalyzer."""
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
            # Add default network metrics
            'citation_network_in_degree': 0,
            'citation_network_out_degree': 0,
            'citation_network_pagerank': 0,
            'citation_network_betweenness': 0
        }
        
        return structured_data

    def _identify_rejection_types(self, doc) -> List[str]:
        """Identify potential rejection types based on the query."""
        rejection_types = []
        
        text = doc.text.lower()
        for section, patterns in self.section_patterns.items():
            if any(pattern in text for pattern in patterns):
                rejection_types.append(section)
                
        return rejection_types
    
    def process_query(self, query: str) -> Dict:
        """
        Process a natural language query and return analysis results.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary containing analysis results
        """
        # Parse the query
        parsed_info = self.parse_user_query(query)
        
        # Convert to structured format for PatentClaimAnalyzer
        structured_data = self._convert_to_structured_format(parsed_info)
        
        # Get analysis results
        analysis_results = self.patent_analyzer.analyze_claim(structured_data)
        
        # Enhance results with NLP insights
        enhanced_results = self._enhance_results(analysis_results, parsed_info)
        
        return enhanced_results
    
    def _convert_to_structured_format(self, parsed_info: Dict) -> Dict:
        """Convert parsed information to format expected by PatentClaimAnalyzer."""
        # Create passage location text based on identified rejection types
        passage_locations = []
        for rejection in parsed_info['rejection_types']:
            passage_locations.append(f"c. {rejection}")
        
        return {
            'examinerCitedReferenceIndicator': 'FALSE',  # Default values
            'applicantCitedExaminerReferenceIndicator': 'FALSE',
            'nplIndicator': 'FALSE',
            'techCenter': self._map_technical_field(parsed_info['technical_field']),
            'passageLocationText': str(passage_locations),
            'publicationNumber': None  # New application
        }
    
    def _map_technical_field(self, field: Optional[str]) -> str:
        """Map technical field to USPTO tech center."""
        if not field:
            return '2100'  # Default to computer tech
            
        field = field.lower()
        tech_center_mapping = {
            'computer': '2100',
            'software': '2100',
            'ai': '2100',
            'network': '2100',
            'electronic': '2800',
            'hardware': '2800',
            'biotech': '1600',
            'chemical': '1700',
            'mechanical': '3700'
        }
        
        for key, value in tech_center_mapping.items():
            if key in field:
                return value
                
        return '2100'  # Default
    
    def _enhance_results(self, analysis_results: Dict, parsed_info: Dict) -> Dict:
        """Enhance analysis results with NLP insights."""
        enhanced = analysis_results.copy()
        
        # Add query-specific information
        enhanced['query_analysis'] = {
            'query_type': parsed_info['query_type'],
            'technical_field': parsed_info['technical_field'],
            'identified_claim': parsed_info['claim_text'],
            'cited_references': parsed_info['cited_references']
        }
        
        # Add natural language explanations
        enhanced['explanations'] = self._generate_explanations(
            analysis_results,
            parsed_info
        )
        
        return enhanced
    
    def _generate_explanations(self, analysis_results: Dict, parsed_info: Dict) -> Dict:
        """Generate natural language explanations of the analysis."""
        explanations = {
            'summary': self._generate_summary(analysis_results, parsed_info),
            'technical_analysis': self._generate_technical_analysis(analysis_results),
            'recommendations': self._enhance_recommendations(
                analysis_results['recommendations'],
                parsed_info
            )
        }
        
        return explanations
    
    def _generate_summary(self, results: Dict, parsed_info: Dict) -> str:
        """Generate a natural language summary of the analysis."""
        rejection_prob = results['rejection_probability']
        query_type = parsed_info['query_type']
        
        if query_type == 'patentability':
            if rejection_prob > 0.7:
                return "Based on the analysis, this invention may face significant patentability challenges."
            elif rejection_prob > 0.4:
                return "The invention shows promise but may need refinement to improve patentability."
            else:
                return "Initial analysis suggests favorable patentability prospects."
        elif query_type == 'validity':
            # Generate validity-specific summary
            pass
        # Add other query types...
        
        return "Analysis complete. See detailed results below."
    
    def _generate_technical_analysis(self, results: Dict) -> str:
        """Generate technical analysis explanation."""
        key_features = results['key_features']
        analysis = []
        
        if key_features['has_103_rejection']:
            analysis.append("Potential obviousness concerns identified.")
        if key_features['has_102_rejection']:
            analysis.append("Novelty issues may need to be addressed.")
        if key_features['has_112_rejection']:
            analysis.append("Clarity or written description issues detected.")
            
        return " ".join(analysis) if analysis else "No significant technical issues identified."
    
    def _enhance_recommendations(self, recommendations: List[str], parsed_info: Dict) -> List[str]:
        """Enhance recommendations based on query context."""
        enhanced = recommendations.copy()
        
        # Add query-specific recommendations
        if parsed_info['query_type'] == 'patentability':
            if not parsed_info['cited_references']:
                enhanced.append("Consider conducting a thorough prior art search.")
                
        if parsed_info['technical_field']:
            enhanced.append(
                f"Review recent patents in {parsed_info['technical_field']} "
                "for potential prior art."
            )
            
        return enhanced
    
    def save(self, filepath=None):
        """Save both NLP models and patent analyzer."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"patent_nlp_model_{timestamp}"
        
        # Create a dictionary with all necessary components
        model_data = {
            'patent_analyzer': self.patent_analyzer,
            'nlp': self.nlp,
            'section_patterns': self.section_patterns
        }
        
        # Save everything in one file
        logger.info(f"Saving model to {filepath}")
        joblib.dump(model_data, filepath)
        logger.info("Model saved successfully")

    @classmethod
    def load(cls, filepath):
        """Load a saved NLP interface."""
        logger.info(f"Loading model from {filepath}")
        
        # Create instance
        instance = cls(model_path=None)  # Create empty instance
        
        try:
            # Load the saved data
            model_data = joblib.load(filepath)
            
            # Restore components
            instance.patent_analyzer = model_data['patent_analyzer']
            instance.nlp = model_data['nlp']
            instance.section_patterns = model_data['section_patterns']
            
            logger.info("Model loaded successfully")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


# Example Flask route
def create_patent_routes(app: Flask, nlp_interface: PatentNLPInterface):
    @app.route("/analyze_patent", methods=["POST"])
    def analyze_patent():
        try:
            data = request.get_json()
            if not data or "query" not in data:
                return jsonify({"error": "Please provide a query."}), 400
            
            results = nlp_interface.process_query(data["query"])
            
            return jsonify({
                "results": results,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_version": "patent_nlp_v1"
                }
            })
            
        except Exception as e:
            return jsonify({
                "error": "Internal server error",
                "details": str(e)
            }), 500

# Example usage:
if __name__ == "__main__":
    
    patent_analyzer = PatentClaimAnalyzer()
    
    # Helper function to detect rejection types
    def has_rejection_type(text, rejection_type):
        if pd.isna(text):
            return False
        return f'c. {rejection_type}' in text or f'c.{rejection_type}' in text
    
    df = pd.read_csv('../datasets/patent_data/patent_data_20250205_231903_combined_patents_training.csv')
    
    # Create target variable
    y = df['passageLocationText'].apply(lambda x: 
        has_rejection_type(x, '103') or has_rejection_type(x, '102')
    )
    
    # Select features
    feature_columns = [
        'publicationNumber', 'citedDocumentIdentifier',
        'examinerCitedReferenceIndicator', 'applicantCitedExaminerReferenceIndicator',
        'nplIndicator', 'techCenter', 'passageLocationText'
    ]
    X = df[feature_columns]
    
    # Initialize and train
    nlp_interface = PatentNLPInterface()
    nlp_interface.patent_analyzer.fit(X, y)
    
    # Save the trained models
    nlp_interface.save("patent_nlp_model")


if __name__ == "__main__":
    # Load the saved model
    nlp_interface = PatentNLPInterface.load("patent_nlp_model")
    
    # Create a Flask app
    app = Flask(__name__)
    
    # Create routes for patent analysis
    create_patent_routes(app, nlp_interface)
    
    # Run the app
    app.run(port=5000, debug=True)