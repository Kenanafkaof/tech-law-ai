from flask import Flask, jsonify, request
from patent_analysis import PatentNLPInterface, PatentFeatureExtractor
from datetime import datetime
import pandas as pd
import os
import joblib
from train_model import ConsistentFeaturePatentNLPInterface, train_and_save_model

app = Flask(__name__)

# Constants
MODEL_FILE = "./patent_nlp_model.joblib"

class PatentAPIWrapper:
    def __init__(self, model_path):
        self.model_path = model_path
        self.nlp_interface = None
        self.load_model()

    def load_model(self):
        """Load or create new model if needed"""
        try:
            self.nlp_interface = PatentNLPInterface.load(self.model_path)
            print(f"Successfully loaded model from: {self.model_path}")
        except FileNotFoundError:
            print(f"Model file not found at: {self.model_path}")
            print("Training new model...")
            self.model_path = train_and_save_model()
            self.nlp_interface = PatentNLPInterface.load(self.model_path)

    def process_query(self, query_text):
        """Process query with proper feature handling"""
        try:
            # Parse query
            parsed_info = self.nlp_interface.parse_user_query(query_text)
            
            # Create structured data with network features
            structured_data = {
                'examinerCitedReferenceIndicator': 'FALSE',
                'applicantCitedExaminerReferenceIndicator': 'FALSE',
                'nplIndicator': 'FALSE',
                'techCenter': self.nlp_interface._map_technical_field(parsed_info['technical_field']),
                'passageLocationText': str(parsed_info['rejection_types']),
                'publicationNumber': None,
                'citedDocumentIdentifier': None,
                # Add required network features
                'citation_network_in_degree': 0,
                'citation_network_out_degree': 0,
                'citation_network_pagerank': 0,
                'citation_network_betweenness': 0
            }

            # Convert to DataFrame with proper columns
            input_df = pd.DataFrame([structured_data])
            
            # Get prediction probabilities
            proba = self.nlp_interface.patent_analyzer.predict_proba(input_df)[0, 1]

            # Generate recommendations
            recommendations = []
            if proba > 0.7:
                recommendations.append("High risk of rejection - detailed review recommended")
                if "103" in str(parsed_info['rejection_types']):
                    recommendations.append("Potential obviousness issues - consider strengthening non-obviousness arguments")
                if "102" in str(parsed_info['rejection_types']):
                    recommendations.append("Potential novelty issues - review prior art carefully")
            elif proba > 0.5:
                recommendations.append("Moderate risk - consider strengthening claim language")

            return {
                'rejection_probability': float(proba),
                'confidence_level': 'High' if proba > 0.8 else 'Medium' if proba > 0.6 else 'Low',
                'query_analysis': {
                    'query_type': parsed_info['query_type'],
                    'technical_field': parsed_info['technical_field'],
                    'rejection_types': parsed_info['rejection_types']
                },
                'recommendations': recommendations
            }

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            raise


# Initialize API wrapper
api_wrapper = PatentAPIWrapper(MODEL_FILE)

def create_patent_routes(app: Flask, nlp_interface: ConsistentFeaturePatentNLPInterface):
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

if __name__ == "__main__":
    # Train and save the model
    model_path = train_and_save_model()
    
    # Load the model with custom interface
    nlp_interface = ConsistentFeaturePatentNLPInterface.load(model_path)
    
    # Create Flask app and run
    app = Flask(__name__)
    create_patent_routes(app, nlp_interface)
    app.run(port=5000, debug=True)
