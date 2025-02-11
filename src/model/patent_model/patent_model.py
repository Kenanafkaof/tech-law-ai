import pandas as pd
import json
from collections import defaultdict
import re
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

class PatentDataAnalyzer:
    def __init__(self, file_path: str):
        """Initialize the analyzer with the patent data CSV file."""
        self.df = pd.read_csv(file_path)
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare the data for analysis."""
        # Convert date columns to datetime
        date_columns = ['officeActionDate', 'createDateTime']
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col])
            
        # Extract rejection types from passageLocationText
        self.df['rejection_types'] = self.df['passageLocationText'].apply(self.extract_rejection_types)
        
    def extract_rejection_types(self, text: str) -> List[str]:
        """Extract rejection types (101, 102, 103, 112) from text."""
        if pd.isna(text):
            return []
        
        rejection_patterns = [
            r'c\.\s*101',
            r'c\.\s*102',
            r'c\.\s*103',
            r'c\.\s*112'
        ]
        
        found_rejections = []
        for pattern in rejection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_rejections.append(pattern.split('.')[1].strip())
        return found_rejections

    def analyze_tech_center_rejections(self) -> Dict[str, Dict[str, int]]:
        """Analyze rejection patterns by technology center."""
        tech_center_rejections = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df.iterrows():
            tech_center = str(row['techCenter'])
            for rejection in row['rejection_types']:
                tech_center_rejections[tech_center][rejection] += 1
                
        # Convert defaultdict to regular dict and ensure all values are standard Python types
        return {k: dict(v) for k, v in tech_center_rejections.items()}

    def analyze_common_citations(self) -> List[Tuple[str, int]]:
        """Find most commonly cited patent publications."""
        citation_counts = self.df['publicationNumber'].value_counts()
        # Convert to list of tuples with standard Python types
        return [(str(k), int(v)) for k, v in citation_counts.items()][:10]

    def analyze_temporal_patterns(self) -> Dict[str, int]:
        """Analyze temporal patterns in patent applications."""
        year_counts = self.df['officeActionDate'].dt.year.value_counts().sort_index()
        # Convert to dict with standard Python types
        return {str(k): int(v) for k, v in year_counts.items()}

    def generate_tech_center_summary(self, tech_center: str) -> Dict:
        """Generate a summary for a specific tech center."""
        tech_center_data = self.df[self.df['techCenter'] == int(tech_center)]
        
        # Calculate trends and convert to standard Python types
        recent_trends = tech_center_data['officeActionDate'].dt.year.value_counts().sort_index().tail(3)
        recent_trends_dict = {str(k): int(v) for k, v in recent_trends.items()}
        
        return {
            'total_applications': int(len(tech_center_data)),
            'unique_examiners': int(tech_center_data['createUserIdentifier'].nunique()),
            'avg_claims_affected': float(tech_center_data['relatedClaimNumberText'].str.count(',').mean()),
            'common_rejection_types': {k: int(v) for k, v in 
                                    self.analyze_tech_center_rejections()[tech_center].items()},
            'recent_trends': recent_trends_dict
        }

    def query_by_description(self, query: str) -> Dict:
        """Process a natural language query about patent applications."""
        tech_center_mapping = {
            'software': '2100',
            'user interface': '2100',
            'computer': '2100',
            'network': '2100',
            'security': '2100',
            'database': '2100',
            'electrical': '2800',
            'semiconductor': '2800',
            'mechanical': '3700',
            'chemical': '1700',
            'biotechnology': '1600'
        }
        
        relevant_tech_center = None
        for term, tc in tech_center_mapping.items():
            if term in query.lower():
                relevant_tech_center = tc
                break
        
        if relevant_tech_center:
            return {
                'tech_center': relevant_tech_center,
                'summary': self.generate_tech_center_summary(relevant_tech_center)
            }
        else:
            return {'error': 'Could not determine relevant technology center from query'}

    def plot_rejection_patterns(self, tech_center: str = None):
        """Plot rejection patterns, optionally filtered by tech center."""
        if tech_center:
            data = self.df[self.df['techCenter'] == int(tech_center)]
        else:
            data = self.df
            
        rejection_counts = defaultdict(int)
        for rejections in data['rejection_types']:
            for rejection in rejections:
                rejection_counts[f'Section {rejection}'] += 1
                
        plt.figure(figsize=(10, 6))
        plt.bar(rejection_counts.keys(), rejection_counts.values())
        plt.title(f'Rejection Patterns {"for Tech Center " + tech_center if tech_center else "Overall"}')
        plt.ylabel('Number of Rejections')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt

def main():
    try:
        # Initialize analyzer
        analyzer = PatentDataAnalyzer('../../datasets/patent_data/patent_data_20250205_231903_combined_patents_training.csv')
        
        # Example query
        query_result = analyzer.query_by_description("software patent application about user interfaces")
        print("\nQuery Results:")
        print(json.dumps(query_result, indent=2))
        
        # Analyze patterns
        tech_center_patterns = analyzer.analyze_tech_center_rejections()
        print("\nRejection Patterns by Tech Center:")
        print(json.dumps(tech_center_patterns, indent=2))
        
        # Plot rejection patterns for software patents (Tech Center 2100)
        analyzer.plot_rejection_patterns('2100')
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

#../../datasets/patent_data/patent_data_20250205_231903_combined_patents_training.csv