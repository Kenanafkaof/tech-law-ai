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
    
    # Initialize both analyzer and classifier with proper error handling
    try:
        global analyzer, classifier, strategic_analyzer
        
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

    return app

# Initialize the analyzer
analyzer = None
classifier = None
strategic_analyzer = None


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)