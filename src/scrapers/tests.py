import requests
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timedelta

def test_tech_cases_request():
    # Load environment variables
    load_dotenv("../../.env")
    api_token = os.getenv("API_TOKEN")
    
    # Setup request
    headers = {
        'Authorization': f'Token {api_token}',
        'Accept': 'application/json'
    }
    
    # Make a single request to get tech-related cases
    base_url = "https://www.courtlistener.com/api/rest/v4/dockets/"
    
    # Get date range for last 5 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # Parameters targeting tech cases
    params = {
        'limit': 5,  # Get 5 results to see variety
        'nature_of_suit__icontains': 'patent',  # Start with patent cases
        'date_filed__gte': start_date,
        'date_filed__lte': end_date
    }
    
    try:
        # Make the request with increased timeout
        response = requests.get(
            base_url,
            headers=headers,
            params=params,
            timeout=30
        )
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            print("Successfully connected to API!")
            print(f"\nFound {data.get('count', 0)} potential tech-related cases")
            
            if data.get('results'):
                print("\nSample of cases found:")
                for i, result in enumerate(data['results'], 1):
                    print(f"\nCase {i}:")
                    print(f"Docket ID: {result.get('id')}")
                    print(f"Case Name: {result.get('case_name')}")
                    print(f"Nature of Suit: {result.get('nature_of_suit')}")
                    print(f"Cause: {result.get('cause', 'N/A')}")
                    print(f"Date Filed: {result.get('date_filed')}")
                    print(f"Court: {result.get('court_id')}")
                    
                # Print available fields for analysis
                print("\nAvailable fields in response:")
                for field in data['results'][0].keys():
                    print(f"- {field}")
            else:
                print("No results found")
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def test_search_api():
    """Alternative test using the search API"""
    load_dotenv("../../.env")
    api_token = os.getenv("API_TOKEN")
    
    headers = {
        'Authorization': f'Token {api_token}',
        'Accept': 'application/json'
    }
    
    base_url = "https://www.courtlistener.com/api/rest/v4/search/"
    
    # Try a more targeted search
    params = {
        'q': 'artificial intelligence OR machine learning OR software patent',
        'type': 'o',  # opinions
        'order_by': 'score desc',
        'limit': 5
    }
    
    try:
        response = requests.get(
            base_url,
            headers=headers,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("\nSearch API Results:")
            print(f"Found {data.get('count', 0)} matching documents")
            
            if data.get('results'):
                for i, result in enumerate(data['results'], 1):
                    print(f"\nResult {i}:")
                    print(f"Case Name: {result.get('caseName')}")
                    print(f"Date Filed: {result.get('dateFiled')}")
                    print(f"Court: {result.get('court')}")
                    # Print a snippet of the text
                    snippet = result.get('snippet', '')[:200] + '...' if result.get('snippet') else 'No snippet available'
                    print(f"Snippet: {snippet}")
            
            return True
        else:
            print(f"Search API Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Search API Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing CourtListener API for tech cases...")
    print("\nTesting Dockets API:")
    test_tech_cases_request()
    
    print("\nTesting Search API:")
    test_search_api()