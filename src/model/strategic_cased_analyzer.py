# strategic_case_analyzer.py
from collections import defaultdict
from typing import Dict, List
import re
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from case_classification_model import EnhancedTemporalClassifier
import pandas as pd

nltk.download('punkt')

class StrategicCaseAnalyzer(EnhancedTemporalClassifier):
    def __init__(self):
        super().__init__()
        self.processed_cases = []
        self.outcome_patterns = {
            'winning_arguments': [
                r'court\s+(?:finds|concludes|holds|determines)\s+(?:that\s+)?([^.]{10,200})',
                r'(?:motion|petition)\s+(?:is|was)\s+granted\s+(?:because|as|since)\s+([^.]{10,200})',
                r'successfully\s+(?:demonstrated|proved|established)\s+(?:that\s+)?([^.]{10,200})',
                r'evidence\s+(?:shows|demonstrates|establishes)\s+(?:that\s+)?([^.]{10,200})',
                r'plaintiff\s+(?:prevails|succeeded)\s+(?:on|in)\s+([^.]{10,200})'
            ],
            'key_reasoning': [
                r'(?:because|since|as)\s+([^.]{10,200})',
                r'(?:therefore|thus|accordingly)\s*,\s*([^.]{10,200})',
                r'(?:based\s+on|in\s+light\s+of)\s+([^.]{10,200})'
            ]
        }

    def process_case(self, case: Dict) -> Dict:
        """Process a case with enhanced feature extraction."""
        processed = case.copy()
        
        # Extract text fields
        text_fields = [
            case.get('full_text', ''),
            case.get('syllabus', ''),
            case.get('text_excerpt', ''),
            case.get('procedural_history', '')
        ]
        processed['text'] = ' '.join(filter(None, text_fields))
        
        # Get time period
        date_filed = case.get('date_filed', '')
        processed['time_period'] = self.get_time_period(date_filed)
        
        # Extract citations
        citations = case.get('citations', [])
        if isinstance(citations, str):
            try:
                citations = eval(citations) if citations.strip() else []
            except:
                citations = []
        
        # Calculate citation features
        citation_years = []
        for cite in citations:
            try:
                cite_date = pd.to_datetime(cite.get('date', ''))
                if pd.notna(cite_date):
                    citation_years.append(cite_date.year)
            except:
                continue
        
        # Add citation metrics
        processed['citation_count'] = len(citations)
        processed['avg_citation_age'] = (
            pd.to_datetime(date_filed).year - np.mean(citation_years)
            if citation_years and date_filed else 0
        )
        processed['citation_span'] = (
            max(citation_years) - min(citation_years)
            if len(citation_years) > 1 else 0
        )
        
        # Extract legal issues
        processed['legal_issues'] = self.extract_temporal_issues(
            processed['text'], 
            processed['time_period']
        )
        
        return processed
    
    def extract_strategic_insights(self, case_text: str, time_period: str = None) -> Dict:
        """Extract winning arguments and reasoning with context."""
        insights = {
            'winning_arguments': [],
            'key_reasoning': [],
            'time_period': time_period
        }
        
        sentences = nltk.sent_tokenize(case_text)
        
        for i, sentence in enumerate(sentences):
            for category, patterns in self.outcome_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        text = match.group(1) if match.groups() else match.group(0)
                        context = ' '.join(sentences[max(0, i-1):min(len(sentences), i+2)])
                        
                        insight = {
                            'text': text.strip(),
                            'context': context,
                            'time_period': time_period
                        }
                        
                        if not any(self._is_similar_insight(insight, existing) 
                                 for existing in insights[category]):
                            insights[category].append(insight)
        
        return insights
    
    def _is_similar_insight(self, insight1: Dict, insight2: Dict, threshold: float = 0.8) -> bool:
        """Check if insights are too similar to include both."""
        text1 = insight1['text'].lower()
        text2 = insight2['text'].lower()
        
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        if not tokens1 or not tokens2:
            return False
            
        overlap = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        return overlap > threshold
    
    def recommend_strategy(self, case_text: str) -> Dict:
        """Enhanced recommendation system with better analysis."""
        similar_cases = []
        time_period_dist = defaultdict(int)
        
        # Process query case
        query_case = {'full_text': case_text, 'date_filed': None}
        query_features = self.process_case(query_case)
        query_vector = self.tfidf_vectorizer.transform([query_features['text']])
        
        # Extract query case issues
        query_issues = query_features['legal_issues']
        query_period = query_features['time_period']
        
        # Find similar cases with enhanced similarity metrics
        for case in self.processed_cases:
            case_vector = self.tfidf_vectorizer.transform([case['text']])
            
            # Calculate multiple similarity components
            text_similarity = float(cosine_similarity(query_vector, case_vector)[0][0])
            
            # Issue overlap score
            case_issues = case['legal_issues']
            issue_overlap = len(query_issues & case_issues) / len(query_issues | case_issues) if query_issues or case_issues else 0
            
            # Temporal proximity score
            period_weights = {
                'post_2020': 4,
                '2010_2020': 3,
                '2000_2010': 2,
                'pre_2000': 1
            }
            temporal_score = 1.0 - (abs(period_weights[query_period] - period_weights[case['time_period']]) / 3.0)
            
            # Citation overlap score
            query_citations = set(self._extract_citations(query_features['text']))
            case_citations = set(self._extract_citations(case['text']))
            citation_overlap = len(query_citations & case_citations) / len(query_citations | case_citations) if query_citations or case_citations else 0
            
            # Combined similarity score with weights
            similarity = (
                0.4 * text_similarity +
                0.3 * issue_overlap +
                0.2 * temporal_score +
                0.1 * citation_overlap
            )
            
            if similarity > 0.3:  # Increased threshold
                insights = self.extract_strategic_insights(case['text'], case['time_period'])
                time_period_dist[case['time_period']] += 1
                
                # Extract key points from the case
                key_points = self._extract_key_points(case['text'])
                
                similar_cases.append({
                    'similarity': similarity,
                    'similarity_breakdown': {
                        'text_similarity': text_similarity,
                        'issue_overlap': issue_overlap,
                        'temporal_score': temporal_score,
                        'citation_overlap': citation_overlap
                    },
                    'time_period': case['time_period'],
                    'key_points': key_points,
                    'winning_arguments': self._deduplicate_insights(insights['winning_arguments']),
                    'key_reasoning': self._deduplicate_insights(insights['key_reasoning'])
                })
        
        # Sort and deduplicate similar cases
        similar_cases = self._deduplicate_cases(
            sorted(similar_cases, key=lambda x: x['similarity'], reverse=True)
        )[:5]
        
        # Extract and deduplicate insights from query case
        query_insights = self.extract_strategic_insights(case_text, query_period)
        winning_args = self._deduplicate_insights(query_insights['winning_arguments'])
        key_reasoning = self._deduplicate_insights(query_insights['key_reasoning'])
        
        # Generate strategic analysis
        strategy_analysis = self._analyze_winning_strategies(similar_cases)
        
        return {
            'winning_arguments': winning_args,
            'key_reasoning': key_reasoning,
            'time_period_distribution': dict(time_period_dist),
            'similar_cases': similar_cases,
            'number_similar_cases': len(similar_cases),
            'strategy_analysis': strategy_analysis
        }

    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from case text."""
        citations = []
        
        # Common citation patterns
        patterns = [
            # Case citations (e.g., Smith v. Jones, 123 F.3d 456)
            r'(?:[A-Z][a-z]+\s+)+v\.?\s+(?:[A-Z][a-z]+\s+)+\d+\s+F\.(?:2d|3d|4th)?\s+\d+',
            
            # Supreme Court citations (e.g., 567 U.S. 123)
            r'\d+\s+U\.S\.\s+\d+',
            
            # Patent citations (e.g., U.S. Patent No. 7,123,456)
            r'U\.S\.\s+Patent\s+No\.\s+[\d,]+',
            
            # Federal Circuit citations (e.g., Fed. Cir. 2015)
            r'Fed\.\s+Cir\.\s+\d{4}',
            
            # Patent application citations
            r'Patent\s+Application\s+No\.\s+[\d/,]+',
            
            # Publication citations
            r'Publication\s+No\.\s+[\d\-,]+',
            
            # Year citations in legal context
            r'(?:19|20)\d{2}\s+WL\s+\d+'
        ]
        
        # Extract citations using patterns
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citation = match.group().strip()
                if citation not in citations:  # Avoid duplicates
                    citations.append(citation)
        
        # Extract citations from references list if present
        ref_section = re.search(r'References Cited[^\n]*\n(.*?)(?:\n\n|\Z)', 
                            text, re.DOTALL | re.IGNORECASE)
        if ref_section:
            ref_text = ref_section.group(1)
            # Extract structured references
            ref_patterns = [
                r'\[\d+\]\s*(.*?)(?=\[\d+\]|\Z)',  # Numbered references
                r'\d+\.\s*(.*?)(?=\d+\.\s*|\Z)',   # Numbered list
                r'(?:^|\n)\s*[A-Z].*?(?=\n\s*[A-Z]|\Z)'  # New line starting with capital
            ]
            for pattern in ref_patterns:
                matches = re.finditer(pattern, ref_text, re.MULTILINE | re.DOTALL)
                for match in matches:
                    citation = match.group(1).strip() if match.groups() else match.group().strip()
                    if citation and citation not in citations:
                        citations.append(citation)
        
        return citations

    def _deduplicate_insights(self, insights: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """Improved deduplication of insights."""
        unique_insights = []
        for insight in insights:
            if not any(self._is_similar_insight(insight, existing, threshold) 
                    for existing in unique_insights):
                unique_insights.append(insight)
        return unique_insights

    def _extract_key_points(self, text: str) -> List[Dict]:
        """Extract key points from case text."""
        key_points = []
        key_patterns = {
            'holding': r'court\s+(?:holds|concludes|finds)\s+that\s+([^.]+)',
            'evidence': r'(?:evidence|record)\s+(?:shows|demonstrates|proves)\s+that\s+([^.]+)',
            'reasoning': r'(?:because|since|as)\s+([^.]+)'
        }
        
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            for point_type, pattern in key_patterns.items():
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    point = match.group(1).strip()
                    if point and len(point) > 10:
                        key_points.append({
                            'type': point_type,
                            'text': point
                        })
        return key_points

    def _analyze_winning_strategies(self, similar_cases: List[Dict]) -> Dict:
        """Analyze winning strategies across similar cases."""
        strategy_frequency = defaultdict(int)
        reasoning_frequency = defaultdict(int)
        successful_patterns = defaultdict(list)
        
        for case in similar_cases:
            for arg in case['winning_arguments']:
                strategy_frequency[arg['text']] += 1
                successful_patterns[arg['text']].append({
                    'similarity': case['similarity'],
                    'time_period': case['time_period']
                })
                
            for reason in case['key_reasoning']:
                reasoning_frequency[reason['text']] += 1
        
        return {
            'top_strategies': [
                {
                    'strategy': strategy,
                    'frequency': freq,
                    'cases': patterns
                }
                for strategy, freq, patterns in sorted(
                    [(k, v, successful_patterns[k]) for k, v in strategy_frequency.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            ],
            'common_reasoning': [
                {
                    'pattern': reason,
                    'frequency': freq
                }
                for reason, freq in sorted(
                    reasoning_frequency.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            ]
        }


    def _deduplicate_cases(self, cases: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """Deduplicate similar cases with enhanced criteria."""
        unique_cases = []
        for case in cases:
            # Check if case is sufficiently different from existing cases
            is_unique = True
            for unique_case in unique_cases:
                # Check multiple criteria for similarity
                text_overlap = self._calculate_text_similarity(
                    str(case['winning_arguments']),
                    str(unique_case['winning_arguments'])
                )
                same_period = case['time_period'] == unique_case['time_period']
                similar_points = self._calculate_point_similarity(
                    case['key_points'],
                    unique_case['key_points']
                )
                
                if text_overlap > similarity_threshold and same_period and similar_points > 0.7:
                    is_unique = False
                    break
                    
            if is_unique:
                unique_cases.append(case)
                
        return unique_cases

    def _calculate_point_similarity(self, points1: List[Dict], points2: List[Dict]) -> float:
        """Calculate similarity between two sets of key points."""
        if not points1 or not points2:
            return 0.0
            
        similarities = []
        for p1 in points1:
            for p2 in points2:
                if p1['type'] == p2['type']:
                    text_sim = self._calculate_text_similarity(p1['text'], p2['text'])
                    similarities.append(text_sim)
                    
        return max(similarities) if similarities else 0.0

    def _extract_key_findings(self, text: str) -> List[Dict]:
        """Extract key findings from case text."""
        findings = []
        patterns = [
            r'court\s+(?:holds|concludes|finds)\s+that\s+([^.]+)',
            r'(?:granting|denying)\s+(?:motion|petition)\s+(?:because|as|since)\s+([^.]+)',
            r'(?:evidence|record)\s+(?:shows|demonstrates|establishes)\s+that\s+([^.]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                finding = match.group(1).strip()
                if finding and len(finding) > 10:
                    findings.append({
                        'text': finding,
                        'type': 'finding'
                    })
        
        return findings

    def _generate_analysis_summary(self, similar_cases: List[Dict]) -> Dict:
        """Generate a summary of analysis findings."""
        common_patterns = defaultdict(int)
        successful_strategies = defaultdict(int)
        
        for case in similar_cases:
            for arg in case['winning_arguments']:
                successful_strategies[arg['text']] += 1
            for reason in case['key_reasoning']:
                common_patterns[reason['text']] += 1
                
        return {
            'most_successful_strategies': [
                {'strategy': k, 'frequency': v}
                for k, v in sorted(successful_strategies.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            ],
            'common_reasoning_patterns': [
                {'pattern': k, 'frequency': v}
                for k, v in sorted(common_patterns.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            ]
        }

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0

    
    def fit(self, cases: List[Dict]):
        """Train the model and store processed cases."""
        self.processed_cases = []
        for case in cases:
            processed = self.process_case(case)
            if processed['text'].strip():  # Only keep non-empty cases
                self.processed_cases.append(processed)
        
        super().fit(self.processed_cases)
        return self
    
    def predict(self, case_text: str):
        """Wrapper for recommend_strategy to maintain compatibility."""
        return self.recommend_strategy(case_text)