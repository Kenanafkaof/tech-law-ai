import pandas as pd
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class EnhancedPatentAnalyzer:
    def __init__(self, file_path: str):
        """Initialize the analyzer with patent data."""
        try:
            if file_path:
                self.df = pd.read_csv(file_path)
                self.clean_data()
            else:
                self.df = None
        except Exception as e:
            raise Exception(f"Error loading patent data: {str(e)}")

    def clean_data(self):
        """Clean and prepare the data for analysis."""
        # Convert date columns
        date_columns = ['officeActionDate', 'createDateTime']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Add derived columns with error handling
        try:
            self.df['claim_numbers'] = self.df['relatedClaimNumberText'].apply(self.parse_claim_numbers)
            self.df['rejection_types'] = self.df['passageLocationText'].apply(self.extract_rejection_types)
        except Exception as e:
            print(f"Warning in data cleaning: {str(e)}")

    def parse_claim_numbers(self, claim_text: str) -> List[int]:
        """Parse claim numbers from text."""
        if pd.isna(claim_text):
            return []
        
        claims = set()
        try:
            parts = str(claim_text).split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    claims.update(range(start, end + 1))
                else:
                    try:
                        claims.add(int(part))
                    except ValueError:
                        continue
        except (ValueError, AttributeError):
            return []
            
        return sorted(list(claims))

    def extract_rejection_types(self, text: str) -> List[str]:
        """Extract rejection types from text."""
        if pd.isna(text):
            return []
        
        text = str(text).lower()
        found_rejections = set()
        
        # Look for rejection patterns
        patterns = [
            r'(?:section|§|\bc\.?\s*)(101|102|103|112)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_rejections.update(matches)
            
        return sorted(list(found_rejections))

    def analyze_tech_center(self, tech_center: str = None):
        if tech_center and int(tech_center) not in self.df['techCenter'].unique():
            raise ValueError(f"Tech center {tech_center} not found in data.")
        
        # Filter dynamically instead of hardcoding
        data = self.df if tech_center is None else self.df[self.df['techCenter'] == int(tech_center)]
        
        return {
            'rejection_analysis': self.analyze_rejections(data),
            'claim_analysis': self.analyze_claims(data),
            'temporal_analysis': self.analyze_temporal_patterns(data),
            'citation_analysis': self.analyze_citations(data),
        }


    def analyze_rejections(self, data: pd.DataFrame) -> Dict:
        """Analyze rejection patterns."""
        rejection_counts = defaultdict(int)
        
        # Count rejections
        for rejections in data['rejection_types']:
            for rejection in rejections:
                rejection_counts[rejection] += 1
        
        # Find most common rejection
        most_common = None
        if rejection_counts:
            most_common = max(rejection_counts.items(), key=lambda x: x[1])
        
        return {
            'counts': dict(rejection_counts),
            'most_common': most_common,
            'total_rejections': sum(rejection_counts.values())
        }

    def analyze_claims(self, data: pd.DataFrame) -> Dict:
        """Analyze claim patterns."""
        claim_counts = defaultdict(int)
        
        # Count claim frequencies
        for claims in data['claim_numbers']:
            for claim in claims:
                claim_counts[claim] += 1
        
        # Get most rejected claims
        most_rejected = []
        if claim_counts:
            most_rejected = sorted(claim_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'most_rejected_claims': most_rejected,
            'avg_claims_per_rejection': sum(len(claims) for claims in data['claim_numbers']) / len(data) if len(data) > 0 else 0,
            'claim_frequency': dict(claim_counts)
        }

    def analyze_temporal_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze patterns over time."""
        if 'officeActionDate' not in data.columns:
            return {
                'yearly_trends': {},
                'monthly_trends': {},
                'total_applications': len(data)
            }
        
        # Calculate yearly trends
        yearly_data = data['officeActionDate'].dt.year.value_counts().sort_index()
        yearly_trends = {str(k): int(v) for k, v in yearly_data.items()}
        
        # Calculate monthly trends
        monthly_data = data['officeActionDate'].dt.to_period('M').value_counts().sort_index()
        monthly_trends = {str(k): int(v) for k, v in monthly_data.items()}
        
        # Get last 12 months if available
        if len(monthly_trends) > 12:
            monthly_trends = dict(list(monthly_trends.items())[-12:])
        
        return {
            'yearly_trends': yearly_trends,
            'monthly_trends': monthly_trends,
            'total_applications': len(data)
        }

    def analyze_citations(self, data: pd.DataFrame) -> Dict:
        """Analyze citation patterns."""
        try:
            # Remove NaN values and get counts
            valid_citations = data['citedDocumentIdentifier'].dropna()
            citation_counts = valid_citations.value_counts()
            
            # Convert to list and get top 10
            top_citations = []
            if not citation_counts.empty:
                for idx, (doc_id, count) in enumerate(citation_counts.items()):
                    if idx >= 10:
                        break
                    top_citations.append((str(doc_id), int(count)))
            
            return {
                'top_citations': top_citations,
                'total_citations': len(valid_citations)
            }
        except Exception as e:
            print(f"Citation analysis error: {str(e)}")
            return {
                'top_citations': [],
                'total_citations': 0
            }

    def generate_summary(self, tech_center: str, rejection_analysis: Dict,
                        claim_analysis: Dict, temporal_analysis: Dict) -> str:
        """Generate a natural language summary of the analysis."""
        try:
            most_common_text = ""
            if rejection_analysis['most_common']:
                most_common_text = f"Most common: Section {rejection_analysis['most_common'][0]} ({rejection_analysis['most_common'][1]} instances)"
                
            most_rejected_claim_text = ""
            if claim_analysis['most_rejected_claims']:
                claim, count = claim_analysis['most_rejected_claims'][0]
                most_rejected_claim_text = f"Most frequently rejected claim: {claim} ({count} times)"
            
            return f"""
Technology Center {tech_center} Analysis Summary:

1. Rejection Patterns:
   - Total rejections: {rejection_analysis['total_rejections']}
   - {most_common_text}
   
2. Claim Analysis:
   - Average claims per rejection: {claim_analysis['avg_claims_per_rejection']:.2f}
   - {most_rejected_claim_text}
   
3. Temporal Trends:
   - Total applications: {temporal_analysis['total_applications']}
   - Recent trend: {'Increasing' if self._is_increasing(temporal_analysis['monthly_trends']) else 'Decreasing'}
"""
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def _is_increasing(self, trends: Dict[str, int]) -> bool:
        """Check if trend is increasing."""
        values = list(trends.values())
        if len(values) < 2:
            return False
        return values[-1] > values[0]

    def visualize_rejection_patterns(self, tech_center: str = None):
        """Create visualization of rejection patterns."""
        try:
            if tech_center:
                data = self.df[self.df['techCenter'] == int(tech_center)]
            else:
                data = self.df
                
            rejection_counts = defaultdict(int)
            for rejections in data['rejection_types']:
                for rejection in rejections:
                    rejection_counts[f'Section {rejection}'] += 1
                    
            plt.figure(figsize=(12, 6))
            
            # Create bar plot
            x = list(rejection_counts.keys())
            y = list(rejection_counts.values())
            sns.barplot(x=x, y=y, palette='viridis')
            
            plt.title(f'Rejection Patterns {"for Tech Center " + str(tech_center) if tech_center else "Overall"}')
            plt.xlabel('Rejection Type')
            plt.ylabel('Number of Rejections')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return plt
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            return None
        
    @classmethod
    def get_instance(cls, file_path: str = None):
        """Singleton pattern to ensure one instance."""
        if not hasattr(cls, 'instance'):
            cls.instance = cls(file_path)
        return cls.instance

    def load_data(self, file_path: str):
        """Load patent data from file."""
        try:
            self.df = pd.read_csv(file_path)
            self.clean_data()
        except Exception as e:
            raise Exception(f"Error loading patent data: {str(e)}")

    def clean_data(self):
        """Clean and prepare the data for analysis."""
        date_columns = ['officeActionDate', 'createDateTime']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        try:
            self.df['claim_numbers'] = self.df['relatedClaimNumberText'].apply(self.parse_claim_numbers)
            self.df['rejection_types'] = self.df['passageLocationText'].apply(self.extract_rejection_types)
        except Exception as e:
            print(f"Warning in data cleaning: {str(e)}")
    
    def get_health_status(self) -> Dict:
        """Get health status of the analyzer."""
        return {
            "status": "healthy" if self.df is not None else "not_initialized",
            "data_loaded": self.df is not None,
            "total_records": len(self.df) if self.df is not None else 0,
            "timestamp": datetime.now().isoformat()
        }

def main():
    try:
        # Initialize analyzer
        file_path = '../../datasets/patent_data/patent_data_20250205_231903_combined_patents_training.csv'
        print(f"\nLoading data from: {file_path}")
        analyzer = EnhancedPatentAnalyzer(file_path)
        
        # Analyze tech center
        tech_center = '2100'
        print(f"\nAnalyzing Technology Center {tech_center}...")
        
        try:
            analysis = analyzer.analyze_tech_center(tech_center)
            
            # Print analysis components
            print("\nRejection Analysis:")
            print(json.dumps(analysis['rejection_analysis'], indent=2))
            
            print("\nClaim Analysis:")
            print(json.dumps(analysis['claim_analysis'], indent=2))
            
            print("\nTemporal Analysis:")
            print(json.dumps(analysis['temporal_analysis'], indent=2))
            
            print("\nCitation Analysis:")
            print(json.dumps(analysis['citation_analysis'], indent=2))
            
            print("\nSummary:")
            print(analysis['summary'])
            
            # Create visualization
            print("\nGenerating visualization...")
            plt = analyzer.visualize_rejection_patterns(tech_center)
            if plt:
                plt.show()
            
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
    except Exception as e:
        print(f"Main error: {str(e)}")
        import traceback
        print(traceback.format_exc())

    

if __name__ == "__main__":
    main()