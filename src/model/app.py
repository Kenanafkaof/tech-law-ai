# app.py
from flask import Flask, request, jsonify
from datetime import datetime
from model import LegalCaseAnalyzer
from case_classification_model import EnhancedTemporalClassifier, TemporalLegalClassifier
from strategic_cased_analyzer import StrategicCaseAnalyzer
import traceback
import numpy as np
import pandas as pd
from functools import lru_cache
import joblib
from patent_model.patent_model import EnhancedPatentAnalyzer
from temporal_analysis.visualizer import TechLawData
from flask_cors import CORS
from auth import require_auth

model = joblib.load("../models/legal_case_model.pkl")

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# Load label encoders
court_label_encoder = joblib.load("../models/court_label_encoder.pkl")
judge_label_encoder = joblib.load("../models/judge_label_encoder.pkl")

data_provider = TechLawData('../datasets/test_all_tech_cases_fixed.csv')

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

@lru_cache(maxsize=1000)
def cached_predict(text: str, date: str = None) -> tuple:
    """Cached prediction to improve performance for repeated queries."""
    return classifier.predict(text, date)

@lru_cache(maxsize=1000)
def cached_strategy_analysis(text: str) -> dict:
    """Cached strategic analysis to improve performance."""
    return strategic_analyzer.recommend_strategy(text)

def create_app():
    app = Flask(__name__)

    # Enable CORS
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Initialize both analyzer and classifier with proper error handling
    try:
        global analyzer, classifier, strategic_analyzer, patent_analyzer
        
        # Initialize analyzers
        print("Initializing analyzers...")
        analyzer = LegalCaseAnalyzer.get_instance()
        if not hasattr(analyzer, 'embedder'):
            print("Initializing analyzer models...")
            analyzer.initialize_models()
        
        strategic_analyzer = StrategicCaseAnalyzer()
        
        # Initialize classifier
        print("Initializing TemporalClassifier...")
        classifier = EnhancedTemporalClassifier()

        patent_analyzer = EnhancedPatentAnalyzer.get_instance(
            '../datasets/patent_data/patent_data_20250205_231903_combined_patents_training.csv'
        )
        
        # Load and prepare initial data
        print("Loading and cleaning case data...")
        df = pd.read_csv('../datasets/test_all_tech_cases_fixed.csv', encoding='utf-8')
        print(f"Initially loaded {len(df)} cases")
        
        df = remove_duplicates(df)
        print(f"After removing duplicates: {len(df)} cases")
        
        df = df.fillna('')
        cases = df.to_dict(orient='records')
        
        print("Training models...")
        classifier.fit(cases)
        strategic_analyzer.fit(cases)
        print("Models initialized successfully")
            
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        traceback.print_exc()
        raise


    @app.route("/search", methods=["POST"])
    @require_auth
    def search_endpoint():
        try:
            data = request.get_json()
            if not data or "query" not in data:
                return jsonify({"error": "Please provide a query."}), 400
            
            if not hasattr(analyzer, 'embedder'):
                analyzer.initialize_models()
            
            results = analyzer.search(data["query"], k=data.get("k", 5))
            
            return jsonify({
                "results": results,
                "metadata": {
                    "total_cases": len(analyzer.df),
                    "query_timestamp": datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            print(f"Error in search endpoint: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "error": "Internal server error",
                "details": str(e)
            }), 500

    @app.route("/case_network", methods=["GET"])
    @require_auth
    def get_case_network():
        try:
            return jsonify({
                "nodes": len(analyzer.citation_graph.nodes()),
                "edges": len(analyzer.citation_graph.edges()),
                "most_cited": sorted(
                    analyzer.citation_graph.in_degree(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            })
        except Exception as e:
            print(f"Error in case network endpoint: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "error": "Internal server error",
                "details": str(e)
            }), 500
        
    @app.route("/classify", methods=["POST"])
    @require_auth
    def classify_case():
        """Optimized endpoint for classifying legal cases"""
        try:
            data = request.get_json()
            if not data or "text" not in data:
                return jsonify({"error": "Please provide case text."}), 400
            
            case_text = data["text"].strip()
            if len(case_text) < 10:
                return jsonify({
                    "error": "Text too short. Please provide more detailed case text.",
                    "required_length": "At least 10 characters"
                }), 400
            
            case_date = data.get("date")
            
            # Use cached prediction for performance
            predictions, confidence_scores = cached_predict(case_text, case_date)
            
            # Only compute enhanced features if we have predictions
            time_period = classifier.get_time_period(case_date)
            enhanced_features = classifier.enhance_temporal_features(case_text, time_period)
            
            # Extract relevant patterns for better classification
            legal_issues = classifier.extract_temporal_issues(case_text, time_period)
            
            return jsonify({
                "predictions": predictions,
                "confidence_scores": {k: float(v) for k, v in confidence_scores.items()},
                "time_period": time_period,
                "enhanced_features": {k: float(v) for k, v in enhanced_features.items()},
                "detected_patterns": list(legal_issues),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_version": "enhanced_temporal_v1",
                    "text_length": len(case_text)
                }
            })
            
        except Exception as e:
            print(f"Error in classification endpoint: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "error": "Internal server error",
                "details": str(e)
            }), 500

    @app.route("/temporal_analysis", methods=["POST"])
    @require_auth
    def analyze_temporal():
        """Endpoint for detailed temporal analysis of legal cases"""
        try:
            data = request.get_json()
            if not data or "text" not in data:
                return jsonify({"error": "Please provide case text."}), 400
            
            case_text = data["text"]
            case_date = data.get("date")
            
            # Get time period
            time_period = classifier.get_time_period(case_date)
            
            # Extract all features
            features = classifier.extract_features({
                "full_text": case_text,
                "date_filed": case_date
            })
            
            # Get enhanced features
            enhanced_features = classifier.enhance_temporal_features(case_text, time_period)
            
            # Extract legal issues
            legal_issues = classifier.extract_temporal_issues(case_text, time_period)
            
            return jsonify({
                "time_period": time_period,
                "legal_issues": list(legal_issues),
                "citation_analysis": {
                    "citation_count": features["citation_count"],
                    "avg_citation_age": float(features["avg_citation_age"]),
                    "citation_span": float(features["citation_span"])
                },
                "enhanced_features": {k: float(v) for k, v in enhanced_features.items()},
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "analysis_version": "temporal_v1"
                }
            })
            
        except Exception as e:
            print(f"Error in temporal analysis endpoint: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "error": "Internal server error",
                "details": str(e)
            }), 500

    @app.route("/analyze_strategy", methods=["POST"])
    @require_auth
    def analyze_strategy():
        """Endpoint for strategic case analysis"""
        try:
            data = request.get_json()
            if not data or "text" not in data:
                return jsonify({"error": "Please provide case text."}), 400
            
            case_text = data["text"].strip()
            if len(case_text) < 50:
                return jsonify({
                    "error": "Text too short for meaningful analysis.",
                    "required_length": "At least 50 characters"
                }), 400
            
            # Use cached analysis for performance
            analysis = cached_strategy_analysis(case_text)
            
            # Format the response
            response = {
                "winning_arguments": [
                    {
                        "text": arg["text"],
                        "context": arg["context"],
                        "time_period": arg["time_period"]
                    } for arg in analysis["winning_arguments"]
                ],
                "key_reasoning": [
                    {
                        "text": reason["text"],
                        "context": reason["context"]
                    } for reason in analysis["key_reasoning"]
                ],
                "temporal_distribution": analysis["time_period_distribution"],
                "similar_cases": [
                    {
                        "time_period": case["time_period"],
                        "similarity_score": case["similarity"],
                        "similarity_breakdown": case.get("similarity_breakdown", {}),
                        "winning_arguments": case.get("winning_arguments", [])
                    } for case in analysis["similar_cases"]
                ],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "analysis_version": "strategic_v1",
                    "total_similar_cases": analysis["number_similar_cases"]
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error in strategy analysis endpoint: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "error": "Internal server error",
                "details": str(e)
            }), 500

    @app.route("/model_status", methods=["GET"])
    @require_auth
    def model_status():
        """Endpoint for checking model status"""
        try:
            model_attributes = {
                "classifier_initialized": hasattr(classifier, 'classifier'),
                "tfidf_vectorizer": hasattr(classifier, 'tfidf_vectorizer'),
                "feature_encoder": hasattr(classifier, 'time_period_encoder'),
                "trained": classifier.classifier is not None,
                "cache_info": cached_predict.cache_info()._asdict()
            }
            
            return jsonify({
                "status": "healthy" if all(model_attributes.values()) else "degraded",
                "model_attributes": model_attributes,
                "time_periods": list(classifier.time_periods.keys()),
                "supported_issues": list(classifier.base_patterns.keys()),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_version": "enhanced_temporal_v1"
                }
            })
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e)
            }), 500
         
    @app.route("/health", methods=["GET"])
    @require_auth
    def health_check():
        try:
            model_status = {
                "embedder": hasattr(analyzer, 'embedder'),
                "nlp": hasattr(analyzer, 'nlp'),
                "semantic_index": hasattr(analyzer, 'semantic_index'),
                "citation_graph": hasattr(analyzer, 'citation_graph'),
                "keyword_index": hasattr(analyzer, 'keyword_index')
            }
            
            return jsonify({
                "status": "healthy" if all(model_status.values()) else "degraded",
                "timestamp": datetime.now().isoformat(),
                "cuda_available": hasattr(analyzer, 'device') and str(analyzer.device).startswith('cuda'),
                "model_status": model_status
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e)
            }), 500
        
    @app.route("/diagnostic", methods=["POST"])
    def diagnostic_endpoint():
        """Endpoint for diagnosing search issues"""
        try:
            data = request.get_json()
            if not data or "query" not in data:
                return jsonify({"error": "Please provide a query."}), 400
            
            # Get index statistics
            index_stats = {
                "total_cases": len(analyzer.df),
                "index_size": analyzer.semantic_index.ntotal,
                "embedding_dim": analyzer.semantic_index.d
            }
            
            # Get query embedding stats
            query_embedding = analyzer.embedder.encode([data["query"]]).astype(np.float32)
            embedding_stats = {
                "min": float(query_embedding.min()),
                "max": float(query_embedding.max()),
                "mean": float(query_embedding.mean()),
                "std": float(query_embedding.std())
            }
            
            # Get sample of raw nearest neighbors
            k = data.get("k", 20)
            distances, indices = analyzer.semantic_index.search(query_embedding, k)
            
            raw_matches = []
            for idx, distance in zip(indices[0], distances[0]):
                case = analyzer.df.iloc[idx]
                raw_matches.append({
                    "case_name": case["case_name"],
                    "distance": float(distance),
                    "tech_relevance": case["tech_relevance_category"],
                    "tech_keywords": case["tech_keywords_found"]
                })
            
            return jsonify({
                "query": data["query"],
                "index_stats": index_stats,
                "embedding_stats": embedding_stats,
                "raw_matches": raw_matches
            })
            
        except Exception as e:
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500

    @app.route("/patent/analyze", methods=["POST"])
    @require_auth
    def analyze_patent_data():
        """Endpoint for patent data analysis"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Please provide analysis parameters."}), 400
            
            tech_center = data.get("tech_center", "2100")  # Default to TC 2100
            
            analysis = patent_analyzer.analyze_tech_center(tech_center)
            
            return jsonify({
                "analysis": analysis,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "tech_center": tech_center,
                    "analysis_version": "patent_v1"
                }
            })
            
        except Exception as e:
            print(f"Error in patent analysis endpoint: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "error": "Internal server error",
                "details": str(e)
            }), 500


    @app.route("/patent/health", methods=["GET"])
    @require_auth
    def patent_analyzer_health():
        """Endpoint for checking patent analyzer health"""
        try:
            return jsonify(patent_analyzer.get_health_status())
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e)
            }), 500
        
    @app.route("/predict", methods=["POST"])
    @require_auth
    def predict_case_outcome():
        """
        Predicts the tech relevance score of a given legal case.
        Handles missing judge information gracefully.
        """
        try:
            # Parse request data
            data = request.get_json()
            if not data or "full_text" not in data:
                return jsonify({"error": "Please provide case text."}), 400
            
            # Extract features
            case_text = data["full_text"]
            court = data.get("court", "")
            judge = data.get("judge", None)  # Judge may be missing
            citing_cases_count = len(data.get("citing_cases", []))
            cited_by_count = len(data.get("cited_by", []))

            # Preprocess text
            if not isinstance(case_text, str):
                return jsonify({"error": "Invalid input: full_text must be a string"}), 400
                
            # Convert to string and create a list with a single element
            processed_text = [str(case_text)]
            
            # Transform using TF-IDF
            text_vector = tfidf_vectorizer.transform(processed_text)


            # Encode court (if provided)
            court_encoded = court_label_encoder.transform([court])[0] if court else 0

            # Handle missing judge
            if judge and judge in judge_label_encoder.classes_:
                judge_encoded = judge_label_encoder.transform([judge])[0]
            else:
                judge_encoded = 0  # Default value for unknown/missing judge

            # Create input feature array
            input_features = np.hstack((text_vector.toarray(), [[court_encoded, judge_encoded, citing_cases_count, cited_by_count]]))

            # Make prediction
            prediction = model.predict(input_features)[0]

            return jsonify({
                "prediction": int(prediction),
                "confidence": model.predict_proba(input_features).tolist()
            })

        except Exception as e:
            print(f"Prediction Error: {str(e)}")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500

    @app.route("/cases_over_time", methods=["GET"])
    @require_auth
    def get_cases_over_time():
        return data_provider.get_cases_over_time()

    @app.route("/keyword_trends", methods=["GET"])
    @require_auth
    def get_keyword_trends():
        return data_provider.get_keyword_trends()

    @app.route("/citations_over_time", methods=["GET"])
    @require_auth
    def get_citations_over_time():
        return data_provider.get_citations_over_time()
    
    @app.route("/analyze_trends", methods=["GET"])
    @require_auth
    def analyze_trends():
        """Get trend analysis for specified keywords"""
        try:
            # Get keywords from query parameters, default to top tech keywords
            keywords = request.args.getlist('keywords')
            if not keywords:
                keywords = ['technology', 'patent', 'software']
                
            results = {}
            for keyword in keywords:
                forecast_result = data_provider.forecast_keyword_trends(keyword)
                
                results[keyword] = {
                    'historical_data': [
                        {
                            'year': point['Year'],
                            'value': point['Count']
                        } for point in forecast_result.historical_data
                    ],
                    'forecast': [
                        {
                            'year': point['Year'],
                            'value': point['Predicted_Count'],
                            'lowerBound': ci['Lower'],
                            'upperBound': ci['Upper']
                        } for point, ci in zip(
                            forecast_result.forecast,
                            forecast_result.confidence_intervals
                        )
                    ],
                    'metrics': {
                        'totalMentions': forecast_result.metrics['total_mentions'],
                        'meanCount': forecast_result.metrics['mean_count'],
                        'maxCount': forecast_result.metrics['max_count'],
                        'longTermTrend': forecast_result.metrics['long_term_trend'],
                        'recentTrend': forecast_result.metrics['recent_trend'],
                        'trendInterpretation': forecast_result.metrics['trend_interpretation']
                    }
                }
                
            return jsonify({
                'status': 'success',
                'data': results,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'keywords_analyzed': keywords
                }
            })
            
        except Exception as e:
            print(f"Error in trend analysis endpoint: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'error': str(e),
                'details': traceback.format_exc()
            }), 500

    @app.route("/available_keywords", methods=["GET"])
    @require_auth
    def get_available_keywords():
        """Get list of available keywords for trend analysis"""
        try:
            return jsonify(data_provider.get_available_keywords())
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500

    @app.route("/keyword_summary", methods=["GET"])
    @require_auth
    def get_keyword_summary():
        """Get summary statistics for keywords with proper JSON serialization."""
        try:
            analysis = data_provider.analyze_keyword_trends()
            
            # Convert NumPy types and booleans to standard Python types
            relevant_keywords = []
            for keyword in analysis['relevant_keywords']:
                relevant_keywords.append({
                    'keyword': str(keyword['keyword']),
                    'total_mentions': int(keyword['total_mentions']),
                    'years_present': int(keyword['years_present']),
                    'first_year': int(keyword['first_year']),
                    'last_year': int(keyword['last_year']),
                    'max_count': float(keyword['max_count']),
                    'mean_count': float(keyword['mean_count']),
                    'has_recent_data': bool(keyword['has_recent_data']),  # Convert numpy.bool_ to Python bool
                    'trend': float(keyword['trend'])
                })

            return jsonify({
                'status': 'success',
                'data': {
                    'relevantKeywords': relevant_keywords,
                    'topTrends': relevant_keywords[:10],
                    'summary': {
                        'totalKeywords': len(analysis['all_stats']),
                        'activeKeywords': len(relevant_keywords),
                        'timeRange': f"{min(k['first_year'] for k in relevant_keywords)} - {max(k['last_year'] for k in relevant_keywords)}"
                    }
                }
            })

        except Exception as e:
            print(f"Error in keyword summary: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500
            
    return app

# Initialize the analyzer
analyzer = None
classifier = None
strategic_analyzer = None


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
