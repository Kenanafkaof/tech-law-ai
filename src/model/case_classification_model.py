import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import nltk
import re
import logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from datetime import datetime
from sklearn.pipeline import Pipeline

class TemporalLegalClassifier:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.mlb = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.time_period_encoder = None
        
        # Add time periods for evolving legal terminology
        self.time_periods = {
            'pre_2000': (None, '2000-01-01'),
            '2000_2010': ('2000-01-01', '2010-01-01'),
            '2010_2020': ('2010-01-01', '2020-01-01'),
            'post_2020': ('2020-01-01', None)
        }
        
        # Enhanced legal patterns with temporal variations
        self.base_patterns = {
            'infringement': [
                r'\b(infring|infringement|infringing|infringes)\b',
                r'\b(direct infringement|indirect infringement)\b',
                r'\b(literal infringement|doctrine of equivalents)\b',
                r'\b(induced|contributory)\s+infringement\b'
            ],
            'validity': [
                r'\b(valid|invalid|validity|invalidate|unenforceable)\b',
                r'\b(enforce|enforceability)\b',
                r'\b(patent eligible|patent ineligible)\b'
            ],
            'patentability': [
                r'\b(101|subject matter|abstract idea)\b',
                r'\b(natural phenomenon|laws of nature)\b',
                r'\b(alice test|mayo test)\b',  # Post-2014 terminology
                r'\b(machine-or-transformation)\b'  # Pre-2014 terminology
            ]
            # Add other patterns...
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def evaluate(self, test_cases: List[Dict]):
        """Evaluate model with temporal awareness."""
        if not self.classifier:
            raise ValueError("Model must be trained before evaluation")
        
        processed_cases = []
        true_issues = []
        case_periods = []
        
        for case in test_cases:
            try:
                features = self.extract_features(case)
                if features['text']:
                    processed_cases.append(features)
                    true_issues.append(features['legal_issues'])
                    case_periods.append(features['time_period'])
            except Exception as e:
                self.logger.error(f"Error processing test case: {str(e)}")
                continue
        
        if not processed_cases:
            raise ValueError("No valid test cases found")
            
        # Extract features
        text_features = self.tfidf_vectorizer.transform([case['text'] for case in processed_cases])
        
        # Extract numeric features
        numeric_features = np.array([
            [
                case['citation_count'],
                case['avg_citation_age'],
                case['citation_span']
            ] for case in processed_cases
        ])
        numeric_features_scaled = self.scaler.transform(numeric_features)
        
        # Extract temporal features
        time_period_features = self.time_period_encoder.transform([[case['time_period']] for case in processed_cases])
        
        # Combine all features
        X = np.hstack([
            text_features.toarray(),
            numeric_features_scaled,
            time_period_features
        ])
        
        # Get predictions
        y_true = self.mlb.transform(true_issues)
        y_pred = self.classifier.predict(X)
        
        # Generate overall report
        report = classification_report(
            y_true, y_pred,
            target_names=self.mlb.classes_,
            zero_division=0,
            output_dict=True
        )
        
        # Print overall metrics
        self.logger.info("\nOverall Classification Report:")
        for issue in self.mlb.classes_:
            metrics = report[issue]
            self.logger.info(f"\n{issue}:")
            self.logger.info(f"  Precision: {metrics['precision']:.3f}")
            self.logger.info(f"  Recall: {metrics['recall']:.3f}")
            self.logger.info(f"  F1-score: {metrics['f1-score']:.3f}")
            self.logger.info(f"  Support: {metrics['support']}")
        
        # Generate temporal analysis
        self.logger.info("\nTemporal Performance Analysis:")
        for period in set(case_periods):
            period_indices = [i for i, p in enumerate(case_periods) if p == period]
            if not period_indices:
                continue
                
            y_true_period = y_true[period_indices]
            y_pred_period = y_pred[period_indices]
            
            period_report = classification_report(
                y_true_period, y_pred_period,
                target_names=self.mlb.classes_,
                zero_division=0,
                output_dict=True
            )
            
            self.logger.info(f"\n{period} Performance:")
            for issue in self.mlb.classes_:
                metrics = period_report[issue]
                if metrics['support'] > 0:
                    self.logger.info(f"  {issue}:")
                    self.logger.info(f"    Precision: {metrics['precision']:.3f}")
                    self.logger.info(f"    Recall: {metrics['recall']:.3f}")
                    self.logger.info(f"    F1-score: {metrics['f1-score']:.3f}")
                    self.logger.info(f"    Support: {metrics['support']}")
        
        return report

    def get_time_period(self, date_str):
        """Determine which time period a case belongs to."""
        if not date_str:
            return '2010_2020'  # Default to most common period
        
        try:
            date = pd.to_datetime(date_str)
            for period, (start, end) in self.time_periods.items():
                start_date = pd.to_datetime(start) if start else pd.Timestamp.min
                end_date = pd.to_datetime(end) if end else pd.Timestamp.max
                if start_date <= date < end_date:
                    return period
            # If date doesn't fall in any period, return most recent applicable period
            if date >= pd.to_datetime('2020-01-01'):
                return 'post_2020'
            elif date >= pd.to_datetime('2010-01-01'):
                return '2010_2020'
            elif date >= pd.to_datetime('2000-01-01'):
                return '2000_2010'
            else:
                return 'pre_2000'
        except:
            return '2010_2020'  

    def extract_features(self, case: Dict) -> Dict:
        """Extract features with temporal awareness."""
        features = {}
        
        # Basic text features
        text_fields = {
            'full_text': case.get('full_text', ''),
            'syllabus': case.get('syllabus', ''),
            'procedural_history': case.get('procedural_history', ''),
            'text_excerpt': case.get('text_excerpt', '')
        }
        
        features['text'] = ' '.join(filter(None, text_fields.values()))
        
        # Add temporal features
        date_filed = case.get('date_filed', '')
        time_period = self.get_time_period(date_filed)
        features['time_period'] = time_period
        
        # Extract citations and analyze their dates
        citations = case.get('citations', [])
        if isinstance(citations, str):
            try:
                citations = eval(citations) if citations.strip() else []
            except:
                citations = []
                
        # Calculate temporal citation features
        citation_years = []
        for cite in citations:
            try:
                cite_date = pd.to_datetime(cite.get('date', ''))
                if pd.notna(cite_date):
                    citation_years.append(cite_date.year)
            except:
                continue
                
        features['citation_count'] = len(citations)
        features['avg_citation_age'] = (pd.to_datetime(date_filed).year - np.mean(citation_years)) if citation_years else 0
        features['citation_span'] = (max(citation_years) - min(citation_years)) if len(citation_years) > 1 else 0
        
        # Extract legal issues with temporal context
        features['legal_issues'] = self.extract_temporal_issues(features['text'], time_period)
        
        return features

    # In the TemporalLegalClassifier class, modify the augment_minority_cases method:
    def augment_minority_cases(self, cases: List[Dict], min_cases_per_period: int = 50) -> List[Dict]:
        """Augment cases from underrepresented time periods."""
        from nltk.tokenize import sent_tokenize
        import random
        
        augmented_cases = cases.copy()
        period_cases = defaultdict(list)
        
        # Group cases by time period
        for case in cases:
            period = case.get('time_period', '2010_2020')
            period_cases[period].append(case)
        
        for period, cases_in_period in period_cases.items():
            if len(cases_in_period) < min_cases_per_period:
                # Number of cases to generate
                num_to_generate = min_cases_per_period - len(cases_in_period)
                
                for _ in range(num_to_generate):
                    # Randomly select a base case
                    base_case = random.choice(cases_in_period)
                    
                    # Create augmented version
                    augmented_case = base_case.copy()
                    
                    # Sentence-level augmentation
                    text = base_case.get('full_text', '')
                    sentences = sent_tokenize(text)
                    
                    if len(sentences) > 5:  # Only augment if enough content
                        # Randomly select 80% of sentences
                        num_sentences = int(len(sentences) * 0.8)
                        selected_sentences = random.sample(sentences, num_sentences)
                        augmented_case['full_text'] = ' '.join(selected_sentences)
                        
                        # Add slight variation to citation features
                        for feature in ['citation_count', 'avg_citation_age', 'citation_span']:
                            if feature in augmented_case:
                                variation = random.uniform(0.9, 1.1)
                                augmented_case[feature] *= variation
                        
                        augmented_cases.append(augmented_case)
        
        return augmented_cases

    def implement_dynamic_weighting(self, classifier):
        """Modify classifier to implement dynamic instance weighting."""
        
        def weighted_fit(self, cases: List[Dict]):
            processed_cases = []
            case_weights = []
            
            # Calculate base weights
            current_year = datetime.now().year
            period_counts = defaultdict(int)
            for case in cases:
                period = case.get('time_period', '2010_2020')
                period_counts[period] += 1
            
            max_count = max(period_counts.values())
            period_weights = {
                period: max_count / count 
                for period, count in period_counts.items()
            }
            
            # Process cases with weights
            for case in cases:
                try:
                    features = self.extract_features(case)
                    if features['text']:
                        processed_cases.append(features)
                        
                        # Calculate case weight
                        period = case.get('time_period', '2010_2020')
                        base_weight = period_weights[period]
                        
                        # Adjust weight based on number of issues
                        num_issues = len(features['legal_issues'])
                        issue_modifier = 1.0 if num_issues > 0 else 0.8
                        
                        case_weights.append(base_weight * issue_modifier)
                        
                except Exception as e:
                    self.logger.error(f"Error processing case: {str(e)}")
                    continue
            
            # Update fit method to use weights
            self._fit_with_weights(processed_cases, case_weights)
            
        # Add method to classifier
        setattr(TemporalLegalClassifier, 'weighted_fit', weighted_fit)

    # Add this method to the TemporalLegalClassifier class
    def _fit_with_weights(self, processed_cases: List[Dict], case_weights: List[float]):
        """Internal method to fit the model with case weights."""
        if not processed_cases:
            raise ValueError("No valid cases for training")
        
        # Create temporal features
        self.time_period_encoder = OneHotEncoder(sparse_output=False)
        time_periods = [[case['time_period']] for case in processed_cases]
        time_period_features = self.time_period_encoder.fit_transform(time_periods)
        
        # Initialize and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            ngram_range=(1, 3),
            stop_words='english',
            use_idf=True,
            smooth_idf=True
        )
        
        # Extract and combine all features
        text_features = self.tfidf_vectorizer.fit_transform([case['text'] for case in processed_cases])
        numeric_features = np.array([
            [
                case['citation_count'],
                case['avg_citation_age'],
                case['citation_span']
            ] for case in processed_cases
        ])
        
        # Scale numeric features
        numeric_features_scaled = self.scaler.fit_transform(numeric_features)
        
        # Combine all features
        X = np.hstack([
            text_features.toarray(),
            numeric_features_scaled,
            time_period_features
        ])
        
        # Prepare labels
        self.mlb = MultiLabelBinarizer()
        y = self.mlb.fit_transform([case['legal_issues'] for case in processed_cases])
        
        # Train separate classifier for each issue with weights
        estimators = []
        for i, issue in enumerate(self.mlb.classes_):
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            # Apply weights to current issue
            issue_weights = np.array(case_weights) * (1 + y[:, i])
            rf.fit(X, y[:, i], sample_weight=issue_weights)
            estimators.append(rf)
        
        self.classifier = MultiOutputClassifier(None)
        self.classifier.estimators_ = estimators
        self.classifier.classes_ = None
        
        # Log training summary
        self.log_temporal_summary(processed_cases)

    def enhance_temporal_features(self, text: str, time_period: str) -> Dict[str, float]:
        """Extract enhanced temporal features from text."""
        features = {}
        
        # Add citation evolution features
        features['recent_citation_ratio'] = self._calculate_citation_recency(text, time_period)
        features['precedent_influence'] = self._calculate_precedent_influence(text)
        
        # Add legal term evolution features
        modern_terms = {
            'post_2020': ['subject matter eligibility', 'alice test', 'mayo framework'],
            '2010_2020': ['abstract idea', 'significantly more', 'routine and conventional'],
            '2000_2010': ['machine-or-transformation', 'business method patents'],
            'pre_2000': ['useful arts', 'concrete steps']
        }
        
        # Calculate term usage ratios
        period_terms = modern_terms.get(time_period, [])
        if period_terms:
            term_matches = sum(1 for term in period_terms if term in text.lower())
            features['period_term_ratio'] = term_matches / len(period_terms)
        
        # Add complexity metrics
        features['legal_complexity'] = self._calculate_legal_complexity(text)
        
        return features

    def _calculate_precedent_influence(self, text: str) -> float:
        """Calculate precedent influence score based on citation patterns."""
        # Define patterns for precedential indicators
        precedent_patterns = [
            r'controlling precedent',
            r'binding authority',
            r'stare decisis',
            r'overrul(ed|ing)',
            r'followed by',
            r'citing',
            r'see also',
            r'cf\.',
            r'but see',
            r'accord',
            r'relying on'
        ]
        
        # Count matches for each pattern
        total_matches = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in precedent_patterns
        )
        
        # Normalize by text length (per 1000 words)
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
            
        influence_score = (total_matches * 1000) / word_count
        
        # Cap and normalize score to 0-1 range
        return min(1.0, influence_score / 10)

    def _calculate_citation_recency(self, text: str, time_period: str) -> float:
        """Calculate ratio of recent to old citations."""
        # Extract years from citation patterns
        citation_years = re.findall(r'\b(19|20)\d{2}\b', text)
        if not citation_years:
            return 0.0
            
        citation_years = [int(year) for year in citation_years]
        period_start = {
            'post_2020': 2020,
            '2010_2020': 2010,
            '2000_2010': 2000,
            'pre_2000': 1990
        }.get(time_period, 2010)
        
        recent_citations = sum(1 for year in citation_years if year >= period_start)
        return recent_citations / len(citation_years) if citation_years else 0.0

    def _calculate_legal_complexity(self, text: str) -> float:
        """Calculate legal complexity score based on various metrics."""
        # Count legal phrases
        legal_phrases = [
            'whereas', 'hereby', 'pursuant to', 'notwithstanding',
            'hereinafter', 'aforementioned', 'subject matter'
        ]
        phrase_count = sum(text.lower().count(phrase) for phrase in legal_phrases)
        
        # Calculate average sentence length
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Combine metrics
        complexity_score = (phrase_count / len(text.split()) * 100) + (avg_sentence_length / 50)
        return min(1.0, complexity_score / 10)  # Normalize to 0-1


    def extract_temporal_issues(self, text: str, time_period: str) -> Set[str]:
        """Extract legal issues considering temporal context."""
        issues = set()
        
        # Apply base patterns with temporal context
        for issue, patterns in self.base_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    issues.add(issue)
                    break
                    
        # Add period-specific patterns
        if time_period in ['2010_2020', 'post_2020']:
            # Modern terminology
            if re.search(r'\b(alice step one|alice step two)\b', text, re.IGNORECASE):
                issues.add('patentability')
        elif time_period == 'pre_2000':
            # Historical terminology
            if re.search(r'\b(useful|machine-or-transformation)\b', text, re.IGNORECASE):
                issues.add('patentability')
                
        return issues


    def fit(self, cases: List[Dict]):
        """Train the model with temporal awareness."""
        processed_cases = []
        
        # Process cases with temporal information
        for case in cases:
            try:
                features = self.extract_features(case)
                if features['text']:
                    processed_cases.append(features)
            except Exception as e:
                self.logger.error(f"Error processing case: {str(e)}")
                continue
                
        if not processed_cases:
            raise ValueError("No valid cases found for training")
            
        # Create temporal features
        self.time_period_encoder = OneHotEncoder(sparse_output=False)
        time_periods = [[case['time_period']] for case in processed_cases]
        time_period_features = self.time_period_encoder.fit_transform(time_periods)
        
        # Initialize and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Increased for larger dataset
            min_df=2,
            ngram_range=(1, 3),
            stop_words='english',
            use_idf=True,
            smooth_idf=True
        )
        
        # Extract and combine all features
        text_features = self.tfidf_vectorizer.fit_transform([case['text'] for case in processed_cases])
        numeric_features = np.array([
            [
                case['citation_count'],
                case['avg_citation_age'],
                case['citation_span']
            ] for case in processed_cases
        ])
        
        # Scale numeric features
        numeric_features_scaled = self.scaler.fit_transform(numeric_features)
        
        # Combine all features
        X = np.hstack([
            text_features.toarray(),
            numeric_features_scaled,
            time_period_features
        ])
        
        # Prepare labels
        self.mlb = MultiLabelBinarizer()
        y = self.mlb.fit_transform([case['legal_issues'] for case in processed_cases])
        
        # Train model with temporal weighting
        self.train_temporal_model(X, y, [case['time_period'] for case in processed_cases])
        
        # Log training summary with temporal analysis
        self.log_temporal_summary(processed_cases)

    def train_temporal_model(self, X, y, time_periods):
        """Train model with temporal weighting."""
        # Calculate temporal weights
        current_year = datetime.now().year
        temporal_weights = np.array([
            2.0 if period == 'post_2020' else
            1.5 if period == '2010_2020' else
            1.0 if period == '2000_2010' else
            0.5 # for pre_2000
            for period in time_periods
        ])
        
        # Train separate classifier for each issue
        estimators = []
        for i, issue in enumerate(self.mlb.classes_):
            # Create sample weights combining class balance and temporal importance
            sample_weights = temporal_weights * (1 + y[:, i])  # Give more weight to positive examples
            
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y[:, i], sample_weight=sample_weights)
            estimators.append(rf)
            
        self.classifier = MultiOutputClassifier(None)
        self.classifier.estimators_ = estimators
        self.classifier.classes_ = None

    def log_temporal_summary(self, processed_cases):
        """Log training summary with temporal analysis."""
        # Group cases by time period
        period_groups = defaultdict(list)
        for case in processed_cases:
            period_groups[case['time_period']].append(case)
            
        # Print temporal distribution
        self.logger.info("\nTemporal Distribution of Training Data:")
        for period, cases in sorted(period_groups.items()):
            self.logger.info(f"{period}: {len(cases)} cases")
            
            # Calculate issue distribution for this period
            issue_counts = defaultdict(int)
            for case in cases:
                for issue in case['legal_issues']:
                    issue_counts[issue] += 1
                    
            # Print issue distribution for this period
            for issue, count in sorted(issue_counts.items()):
                percentage = (count / len(cases)) * 100
                self.logger.info(f"  - {issue}: {count} cases ({percentage:.1f}%)")

    def predict(self, case_text: str, case_date: str = None) -> Tuple[List[str], Dict[str, float]]:
        """Make predictions with temporal context."""
        if not self.classifier:
            raise ValueError("Model must be trained before making predictions")
            
        # Process input with temporal context
        features = self.extract_features({
            'full_text': case_text,
            'date_filed': case_date
        })
        
        # Extract features in same format as training
        text_features = self.tfidf_vectorizer.transform([features['text']])
        time_period_features = self.time_period_encoder.transform([[features['time_period']]])
        numeric_features = np.array([[
            features['citation_count'],
            features['avg_citation_age'],
            features['citation_span']
        ]])
        numeric_features_scaled = self.scaler.transform(numeric_features)
        
        # Combine features
        X = np.hstack([
            text_features.toarray(),
            numeric_features_scaled,
            time_period_features
        ])
        
        # Make predictions with confidence threshold
        confidence_threshold = 0.4  # Adjusted based on temporal proximity
        if features['time_period'] in ['post_2020', '2010_2020']:
            confidence_threshold = 0.35  # Lower threshold for recent cases
            
        confidence_scores = {}
        predictions = []
        
        for i, (estimator, issue) in enumerate(zip(self.classifier.estimators_, self.mlb.classes_)):
            probas = estimator.predict_proba(X)[0]
            confidence = float(probas[1]) if len(probas) > 1 else float(probas[0])
            confidence_scores[issue] = confidence
            if confidence >= confidence_threshold:
                predictions.append(issue)
                
        return predictions, confidence_scores

class EnhancedTemporalClassifier(TemporalLegalClassifier):
    def __init__(self):
        super().__init__()
        self.period_specific_models = {}
        self.confidence_calibration = {}
    
    def fit(self, cases: List[Dict]):
        """Enhanced training with period-specific models."""
        # First, train the base model
        super().fit(cases)
        
        # Group cases by time period
        period_cases = defaultdict(list)
        for case in cases:
            period = case.get('time_period', '2010_2020')
            period_cases[period].append(case)
        
        # Train period-specific models if enough data
        for period, period_specific_cases in period_cases.items():
            if len(period_specific_cases) >= 20:  # Minimum cases threshold
                period_model = TemporalLegalClassifier()
                period_model.fit(period_specific_cases)
                self.period_specific_models[period] = period_model
                
                # Calculate confidence calibration
                self.confidence_calibration[period] = self._calibrate_confidence(
                    period_model, period_specific_cases
                )
    
    def predict(self, case_text: str, case_date: str = None) -> Tuple[List[str], Dict[str, float]]:
        """Enhanced prediction using ensemble of models."""
        time_period = self.get_time_period(case_date)
        
        # Get base model predictions
        base_predictions, base_confidence = super().predict(case_text, case_date)
        
        # Get period-specific predictions if available
        if time_period in self.period_specific_models:
            period_predictions, period_confidence = self.period_specific_models[time_period].predict(
                case_text, case_date
            )
            
            # Combine predictions with calibrated weights
            calibration = self.confidence_calibration[time_period]
            combined_confidence = {}
            
            for issue in set(base_predictions + period_predictions):
                base_conf = base_confidence.get(issue, 0.0) * (1 - calibration)
                period_conf = period_confidence.get(issue, 0.0) * calibration
                combined_confidence[issue] = base_conf + period_conf
            
            # Filter predictions based on combined confidence
            threshold = 0.4 if time_period in ['post_2020', '2010_2020'] else 0.5
            predictions = [
                issue for issue, conf in combined_confidence.items()
                if conf >= threshold
            ]
            
            return predictions, combined_confidence
        
        return base_predictions, base_confidence
    
    def _calibrate_confidence(self, model, cases: List[Dict]) -> float:
        """Calculate optimal weight for period-specific model."""
        correct_predictions = 0
        total_predictions = 0
        
        for case in cases:
            predictions, _ = model.predict(case['full_text'], case.get('date_filed'))
            actual_issues = case.get('legal_issues', set())
            
            correct_predictions += len(set(predictions) & actual_issues)
            total_predictions += len(predictions)
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.5

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate cases while preserving the most complete entries."""
    # Sort by completeness (number of non-null values)
    df['completeness'] = df.notna().sum(axis=1)
    df = df.sort_values('completeness', ascending=False)
    
    # Drop duplicates keeping first (most complete) entry
    df = df.drop_duplicates(subset=['full_text'], keep='first')
    
    # Drop the completeness column
    df = df.drop('completeness', axis=1)
    
    return df



def main():
    # Initialize enhanced classifier
    classifier = EnhancedTemporalClassifier()
    
    try:
        print("Loading and cleaning case data...")
        df = pd.read_csv('../datasets/test_all_tech_cases_fixed.csv', encoding='utf-8')
        print(f"Initially loaded {len(df)} cases")
        
        # Remove duplicates
        df = remove_duplicates(df)
        print(f"After removing duplicates: {len(df)} cases")
        
        # Clean data
        df = df.fillna('')
        
        # Convert to list of dictionaries
        cases = df.to_dict(orient='records')
        
        if len(cases) == 0:
            print("Error: No valid cases found in the dataset")
            return
        
        # Augment minority cases before splitting
        print("\nAugmenting minority cases...")
        augmented_cases = classifier.augment_minority_cases(cases, min_cases_per_period=50)
        print(f"After augmentation: {len(augmented_cases)} cases")
            
        # Split into train/test with stratification
        train_cases, test_cases = train_test_split(
            augmented_cases, 
            test_size=0.2, 
            random_state=42
        )
        print(f"Training on {len(train_cases)} cases, testing on {len(test_cases)} cases")
        
        # Apply dynamic weighting
        print("\nImplementing dynamic weighting...")
        classifier.implement_dynamic_weighting(classifier)
        
        # Train model with weighted fit
        print("\nTraining enhanced model...")
        classifier.weighted_fit(train_cases)
        
        # Evaluate
        print("\nEvaluating model performance...")
        report = classifier.evaluate(test_cases)
        
        # Sample predictions with enhanced features
        print("\nSample Predictions on Test Cases:")
        for i, test_case in enumerate(test_cases[:3]):
            print(f"\nTest Case {i+1}:")
            
            # Extract enhanced temporal features
            time_period = classifier.get_time_period(test_case.get('date_filed', ''))
            enhanced_features = classifier.enhance_temporal_features(
                test_case['full_text'], 
                time_period
            )
            
            # Get predictions
            predicted_issues, confidence_scores = classifier.predict(
                test_case['full_text'],
                test_case.get('date_filed', '')
            )
            
            print("Predicted Issues:")
            for issue in predicted_issues:
                conf = confidence_scores[issue]
                print(f"  - {issue}: {conf:.2f} confidence")
            
            print("Enhanced Features:")
            for feature, value in enhanced_features.items():
                print(f"  - {feature}: {value:.3f}")
            
            actual_issues = classifier.extract_temporal_issues(
                test_case['full_text'],
                time_period
            )
            print("Actual Issues:")
            print(f"  - {actual_issues}")
            
        # Save evaluation metrics
        print("\nSaving evaluation report...")
        with open('evaluation_report.txt', 'w') as f:
            f.write("Overall Classification Report:\n")
            for issue in classifier.mlb.classes_:
                metrics = report[issue]
                f.write(f"\n{issue}:\n")
                f.write(f"  Precision: {metrics['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['recall']:.3f}\n")
                f.write(f"  F1-score: {metrics['f1-score']:.3f}\n")
                f.write(f"  Support: {metrics['support']}\n")
            
    except FileNotFoundError:
        print("Error: Could not find the CSV file")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    main()
