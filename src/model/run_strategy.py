# run_strategy.py
import pandas as pd
import numpy as np
from strategic_cased_analyzer import StrategicCaseAnalyzer
import logging
from typing import Dict, List
import sys
from datetime import datetime

# Set up logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def prepare_case_data(df: pd.DataFrame) -> List[Dict]:
    """
    Prepare case data with proper handling of missing values and data types.
    """
    # Fill NaN values appropriately for different column types
    df = df.copy()
    
    # Date handling
    date_columns = ['date_filed', 'date_argued', 'date_reargued']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].fillna('')
    
    # Text field handling
    text_columns = ['full_text', 'text_excerpt', 'syllabus', 'procedural_history']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')
            df[col] = df[col].astype(str)
    
    # Convert citations to list format if it's a string representation
    if 'citations' in df.columns:
        df['citations'] = df['citations'].apply(lambda x: 
            eval(x) if isinstance(x, str) and x.strip() else []
        )
    
    # Convert DataFrame to records
    cases = df.to_dict(orient='records')
    
    # Additional validation
    valid_cases = []
    for case in cases:
        if any(case.get(field, '').strip() for field in text_columns):
            valid_cases.append(case)
    
    logger.info(f"Prepared {len(valid_cases)} valid cases out of {len(cases)} total cases")
    return valid_cases

def analyze_test_case(analyzer: StrategicCaseAnalyzer, test_case: str) -> Dict:
    """
    Analyze a test case and format the results with enhanced deduplication and validation.
    """
    try:
        analysis = analyzer.recommend_strategy(test_case)
        
        # Validate and clean analysis results
        if not isinstance(analysis, dict):
            raise ValueError("Analyzer returned invalid result type")
            
        required_keys = ['winning_arguments', 'key_reasoning', 'time_period_distribution', 
                        'similar_cases', 'number_similar_cases']
        for key in required_keys:
            if key not in analysis:
                logger.warning(f"Missing expected key in analysis results: {key}")
                analysis[key] = [] if key != 'number_similar_cases' else 0
                
        return analysis
        
    except Exception as e:
        logger.error(f"Error during case analysis: {str(e)}")
        raise

def print_analysis_results(analysis: Dict):
    """
    Print analysis results in a structured format.
    """
    print("\nAnalysis Results:\n")
    
    # Print winning arguments
    if analysis['winning_arguments']:
        print("Winning Arguments Found:\n")
        for arg in analysis['winning_arguments']:
            print(f"Argument: {arg.get('text', '')}")
            print(f"Context: {arg.get('context', '')}")
            print(f"Time Period: {arg.get('time_period', '')}\n")
    
    # Print key reasoning
    if analysis['key_reasoning']:
        print("Key Legal Reasoning:\n")
        for reason in analysis['key_reasoning']:
            print(f"Reasoning: {reason.get('text', '')}")
            print(f"Context: {reason.get('context', '')}\n")
    
    # Print temporal distribution
    print("Temporal Distribution:")
    for period, count in analysis['time_period_distribution'].items():
        print(f"- {period}: {count} cases")
    
    # Print similar cases
    print("\nSimilar Cases Analysis:\n")
    for i, case in enumerate(analysis.get('similar_cases', []), 1):
        print(f"Case {i}:")
        print(f"Time Period: {case.get('time_period', 'Unknown')}")
        print(f"Similarity Score: {case.get('similarity', 0):.3f}")
        
        if case.get('similarity_breakdown'):
            print("Similarity Components:")
            for component, score in case['similarity_breakdown'].items():
                print(f"  - {component}: {score:.3f}")
        
        if case.get('winning_arguments'):
            print("Key Arguments:")
            for arg in case['winning_arguments']:
                print(f"- {arg.get('text', '')}")
        print()
    
    print(f"Total Similar Cases Analyzed: {analysis.get('number_similar_cases', 0)}")

def main():
    try:
        # Initialize analyzer
        logger.info("Initializing analyzer...")
        analyzer = StrategicCaseAnalyzer()
        
        # Load and prepare data
        logger.info("Loading data...")
        try:
            df = pd.read_csv("../datasets/test_all_tech_cases_fixed.csv")
        except FileNotFoundError:
            logger.error("Dataset file not found. Please check the file path.")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            logger.error("Dataset file is empty.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading dataset: {str(e)}")
            sys.exit(1)
            
        # Prepare cases
        cases = prepare_case_data(df)
        
        if not cases:
            logger.error("No valid cases found in dataset")
            sys.exit(1)
        
        # Fit the analyzer
        logger.info("Training model...")
        analyzer.fit(cases)
        
        # Test case
        test_case = """
        The plaintiff alleges patent infringement of their semiconductor manufacturing process. 
        The court finds that the defendant's implementation clearly falls within the scope of the 
        patent claims. Based on the evidence presented, the court concludes that the plaintiff has 
        successfully demonstrated infringement. The defendant's arguments regarding invalidity were 
        found to be unpersuasive. Therefore, the court grants the plaintiff's motion for summary judgment.
        """
        
        # Get recommendations
        logger.info("Analyzing test case...")
        analysis = analyze_test_case(analyzer, test_case)
        
        # Deduplicate similar cases with stricter criteria
        if 'similar_cases' in analysis:
            seen_signatures = set()
            unique_cases = []
            for case in analysis['similar_cases']:
                # Create a signature based on multiple components
                signature = (
                    case['time_period'],
                    round(case['similarity'], 3),
                    tuple(sorted((arg.get('text', '')[:100] for arg in case.get('winning_arguments', []))))
                )
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_cases.append(case)
            analysis['similar_cases'] = unique_cases
            analysis['number_similar_cases'] = len(unique_cases)
        
        # Print results
        print_analysis_results(analysis)
        
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        raise

if __name__ == "__main__":
    main()