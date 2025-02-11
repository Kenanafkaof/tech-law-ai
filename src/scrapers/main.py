import requests
import json
from datetime import datetime
import pandas as pd
import numpy as np
from time import sleep
from typing import Dict, List, Optional, Set
import logging
from bs4 import BeautifulSoup
import re
import sys
import os
from tqdm import tqdm

# Configure logging
def setup_logging(log_file: str = 'tech_law_scraper.log'):
    """Set up enhanced logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class EnhancedCourtListenerClient:
    """Enhanced client for CourtListener API with better error handling and rate limiting."""
    
    def __init__(self, api_token: str, rate_limit_delay: float = 1.0):
        if not api_token:
            raise ValueError("API token is required")
        
        self.base_url = "https://www.courtlistener.com/api/rest/v4"
        self.headers = {
            'Authorization': f'Token {api_token}',
            'Content-Type': 'application/json'
        }
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        logger.info("Enhanced CourtListener client initialized")

    def _make_request(self, endpoint: str, params: Optional[Dict] = None, 
                     retry_count: int = 3, retry_delay: float = 2.0) -> Dict:
        """Make API request with enhanced error handling and retry logic."""
        url = f"{self.base_url}/{endpoint}/"
        
        for attempt in range(retry_count):
            try:
                # Add delay before request
                sleep(self.rate_limit_delay)
                
                # Make request with increased timeout
                response = self.session.get(url, params=params, timeout=60)
                
                # Log the URL for debugging
                logger.info(f"Requesting URL: {response.url}")
                
                # Handle various response codes
                if response.status_code == 400:
                    logger.error(f"Bad request: {response.text}")
                    return {'results': [], 'count': 0, 'next': None}
                elif response.status_code == 429:  # Rate limit
                    logger.warning("Rate limit hit, waiting...")
                    sleep(10)  # Wait longer for rate limit
                    continue
                    
                response.raise_for_status()
                
                # Parse response data
                try:
                    data = response.json()
                    result_count = len(data.get('results', []))
                    logger.info(f"Received {result_count} results")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {str(e)}")
                    return {'results': [], 'count': 0, 'next': None}
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out (attempt {attempt + 1}/{retry_count})")
                if attempt < retry_count - 1:
                    sleep(retry_delay * (attempt + 1))
                    continue
                return {'results': [], 'count': 0, 'next': None}
                
            except requests.exceptions.RequestException as e:
                if attempt == retry_count - 1:
                    logger.error(f"Request failed after {retry_count} attempts: {str(e)}")
                    return {'results': [], 'count': 0, 'next': None}
                logger.warning(f"Request failed, attempt {attempt + 1}/{retry_count}: {str(e)}")
                sleep(retry_delay * (attempt + 1))

    def search_opinions(self, params: Optional[Dict] = None) -> Dict:
        """Search opinions with enhanced params handling."""
        return self._make_request("search", params)
    
    def get_clusters(self, params: Optional[Dict] = None) -> Dict:
        """Get opinion clusters with enhanced params handling."""
        return self._make_request("clusters", params)
    
    def get_dockets(self, params: Optional[Dict] = None) -> Dict:
        """Get dockets with enhanced params handling."""
        return self._make_request("dockets", params)

class EnhancedTechLawScraper:
    """Enhanced scraper for technology-related legal cases."""
    
    def __init__(self, api_token: str):
        self.client = EnhancedCourtListenerClient(api_token)
        self.tech_cases = []
        self.empty_field_placeholder = "Not Available"
        self.tech_keywords = {
        'primary': {
            # Patent and IP terms
            'patent', 'intellectual property', 'trademark', 'copyright',
            
            # Core tech terms
            'software', 'computer', 'internet', 'technology',
            'artificial intelligence', 'machine learning',
            
            # Cybersecurity
            'cybersecurity', 'data breach', 'hacking', 'security vulnerability',
            
            # Digital commerce
            'ecommerce', 'electronic commerce', 'digital payment',
            
            # Emerging tech
            'blockchain', 'cryptocurrency', 'virtual reality', 'autonomous',
            
            # Data privacy
            'data privacy', 'personal information', 'data protection'
        },
        'secondary': {
            # Technical terms
            'algorithm', 'database', 'network', 'encryption', 'source code',
            'application', 'digital', 'electronic', 'server', 'cloud',
            
            # Industry terms
            'startup', 'tech company', 'silicon valley', 'semiconductor',
            
            # Common contexts
            'user data', 'online platform', 'website', 'mobile app',
            'digital service', 'tech industry', 'software company',
            
            # Regulatory terms
            'DMCA', 'CFAA', 'data regulation', 'tech standard',
            
            # Infrastructure terms
            'data center', 'cloud computing', 'broadband', 'wireless'
        }
    }

        logger.info("Enhanced TechLawScraper initialized")

    def _is_tech_related(self, text: str) -> bool:
        """
        Enhanced detection of technology-related content using primary and secondary keywords.
        Returns True if the text contains either:
        1. Any primary keyword
        2. Multiple secondary keywords
        """
        text = text.lower()
        primary_matches = sum(1 for kw in self.tech_keywords['primary'] if kw in text)
        if primary_matches > 0:
            return True
            
        secondary_matches = sum(1 for kw in self.tech_keywords['secondary'] if kw in text)
        return secondary_matches >= 2

    def _extract_opinion_text(self, opinion_data: Dict) -> str:
        """Extract and clean opinion text with enhanced parsing."""
        try:
            text_fields = [
                opinion_data.get('plain_text', ''),
                opinion_data.get('html_with_citations', ''),
                opinion_data.get('html', ''),
                opinion_data.get('text_excerpt', ''),
                opinion_data.get('snippet', '')
            ]

            for text in text_fields:
                if text and len(text.strip()) > 0:
                    # Enhanced cleaning with better HTML handling
                    soup = BeautifulSoup(text, 'html.parser')
                    cleaned_text = soup.get_text()
                    # Normalize whitespace while preserving paragraph breaks
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
                    return cleaned_text.strip()

            return self.empty_field_placeholder

        except Exception as e:
            logger.error(f"Error extracting opinion text: {str(e)}")
            return self.empty_field_placeholder

    def _extract_case_data(self, result: Dict) -> Dict:
        """Extract case data with fixed ID handling."""
        try:
            # Gather all text content
            text_sources = []
            
            # Add case name and any available text fields
            if result.get('caseName'):
                text_sources.append(result['caseName'])
            if result.get('caseNameFull'):
                text_sources.append(result['caseNameFull'])
            if result.get('suitNature'):
                text_sources.append(result['suitNature'])
            
            # Get opinion text
            for opinion in result.get('opinions', []):
                if opinion.get('snippet'):
                    text_sources.append(opinion['snippet'])
                if opinion.get('html'):
                    soup = BeautifulSoup(opinion['html'], 'html.parser')
                    text_sources.append(soup.get_text())
                    
            # Join all text content
            full_text = ' '.join(text_sources)
            
            # Calculate tech relevance
            tech_relevance = self._calculate_tech_relevance(full_text)
            matching_keywords = self._get_matching_keywords(full_text)
            
            if tech_relevance < 0.05:
                return None
                
            # Extract case data with proper ID handling
            case_data = {
                'id': str(result.get('cluster_id', '')),  # Use cluster_id as primary ID
                'case_name': result.get('caseName', ''),
                'date_filed': result.get('dateFiled', ''),
                'court': result.get('court', ''),
                'court_id': result.get('court_id', ''),
                'docket_number': result.get('docketNumber', ''),
                'judge': result.get('judge', ''),
                'precedential_status': result.get('status', ''),
                'citation_count': int(result.get('citeCount', 0)),
                'docket_id': result.get('docket_id'),
                'cluster_id': result.get('cluster_id'),
                'text_excerpt': full_text,
                'tech_relevance_score': tech_relevance,
                'tech_keywords_found': matching_keywords,
                'suit_nature': result.get('suitNature', ''),
                'procedural_history': result.get('procedural_history', '')
            }
            
            # Add opinion URLs and text
            if result.get('opinions'):
                primary_opinion = result['opinions'][0]
                case_data.update({
                    'download_url': primary_opinion.get('download_url', ''),
                    'sha1': primary_opinion.get('sha1', ''),
                    'opinion_type': primary_opinion.get('type', '')
                })
            
            return case_data
                
        except Exception as e:
            logger.error(f"Error extracting case data: {str(e)}")
            return None

    def _calculate_tech_relevance(self, text: str) -> float:
        """Calculate tech relevance with improved word boundary matching."""
        if not text or text == self.empty_field_placeholder:
            return 0.0
            
        text = text.lower()
        
        # Use word boundaries for more accurate matching
        def count_keyword_matches(keyword):
            # Handle multi-word keywords
            if ' ' in keyword:
                return 1 if keyword in text else 0
            else:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                return len(re.findall(pattern, text))
        
        # Count matches with word boundaries
        primary_matches = sum(count_keyword_matches(kw) for kw in self.tech_keywords['primary'])
        secondary_matches = sum(count_keyword_matches(kw) for kw in self.tech_keywords['secondary'])
        
        # Context terms with word boundaries
        context_terms = {
            'algorithm', 'digital', 'electronic', 'internet', 'computer',
            'software', 'hardware', 'network', 'cyber', 'data', 'tech',
            'system', 'device', 'platform', 'online', 'virtual'
        }
        
        context_matches = sum(count_keyword_matches(term) for term in context_terms)
        
        # Calculate weighted score with adjusted weights
        weight_sum = (
            (primary_matches * 5) +     # Increased primary weight
            (secondary_matches * 3) +   # Increased secondary weight
            (context_matches * 1)       # Context terms weight
        )
        
        # Normalize by a smaller factor to get higher scores
        score = weight_sum / (len(self.tech_keywords['primary']) * 3)
        
        return min(1.0, score)

    def _get_matching_keywords(self, text: str) -> List[str]:
        """Get list of matching tech keywords for transparency."""
        if not text:
            return []
            
        text = text.lower()
        matches = []
        
        # Check primary keywords
        for kw in self.tech_keywords['primary']:
            if kw in text:
                matches.append(f"primary:{kw}")
                
        # Check secondary keywords
        for kw in self.tech_keywords['secondary']:
            if kw in text:
                matches.append(f"secondary:{kw}")
                
        return matches

    def _get_precedential_status(self, case_data: Dict) -> str:
        """Determine precedential status with improved accuracy."""
        try:
            status = case_data.get('precedential_status') or \
                    case_data.get('status') or \
                    case_data.get('published_status')
            
            if status:
                status = status.lower()
                if any(term in status for term in ['published', 'precedential']):
                    return 'Published'
                elif any(term in status for term in ['unpublished', 'non-precedential']):
                    return 'Unpublished'
            
            # Additional heuristics
            if case_data.get('citation_count', 0) > 5:
                return 'Published'
            if case_data.get('source') == 'C':  # Direct court source
                return 'Published'
                
            return self.empty_field_placeholder
            
        except Exception as e:
            logger.error(f"Error determining precedential status: {str(e)}")
            return self.empty_field_placeholder

    def fetch_tech_cases(self, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None,
                    max_pages: int = 10,
                    min_tech_relevance: float = 0.2) -> pd.DataFrame:
        """
        Fetch technology-related cases with improved querying.
        """
        try:
            logger.info(f"Starting enhanced tech case fetch: {start_date} to {end_date}")
            
            # Base search parameters
            base_params = {
                'court': 'cafc dcd txed nysd',
                'order_by': 'dateFiled desc',
                'type': 'o'  # Ensure we're getting opinions
            }
            
            if start_date:
                base_params['filed_after'] = start_date
            if end_date:
                base_params['filed_before'] = end_date

            # Improved query groups
            query_groups = [
                'patent OR "intellectual property" OR trademark OR copyright',
                'software OR "source code" OR "computer program"',
                'cybersecurity OR "data breach" OR encryption OR "data privacy"',
                'internet OR website OR "e-commerce" OR "online service"',
                'technology OR "tech company" OR startup OR digital',
                'blockchain OR cryptocurrency OR "smart contract"',
                '"artificial intelligence" OR "machine learning" OR algorithm'
            ]
            
            # Process each query group
            for query in query_groups:
                try:
                    search_params = base_params.copy()
                    search_params['q'] = query
                    
                    logger.info(f"\nProcessing query group: {query}")
                    self._fetch_cases_with_pagination(search_params, max_pages // len(query_groups))
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {str(e)}")
                    continue
                
                sleep(2)
            
            # Convert to DataFrame
            df = pd.DataFrame(self.tech_cases)
            if len(df) == 0:
                logger.warning("No cases found matching criteria")
                return pd.DataFrame()
            
            # Process and clean the data
            df = self._process_raw_data(df)
            logger.info(f"Successfully processed {len(df)} tech-related cases")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching tech cases: {str(e)}")
            return pd.DataFrame()


    def _fetch_cases_with_pagination(self, params: Dict, max_pages: int):
        """
        Fetch cases with enhanced pagination handling.
        """
        pages_fetched = 0
        cursor = None
        
        progress_bar = tqdm(total=max_pages, desc="Fetching tech law cases")
        
        while pages_fetched < max_pages:
            try:
                # Update params with cursor if we have one
                current_params = params.copy()
                if cursor:
                    current_params['cursor'] = cursor
                
                # Get results
                response = self.client.search_opinions(current_params)
                
                # Process each result
                results = response.get('results', [])
                logger.info(f"Processing {len(results)} results from page {pages_fetched + 1}")
                
                for result in results:
                    case_data = self._extract_case_data(result)
                    if case_data:  # Only process if we successfully extracted data
                        self.tech_cases.append(case_data)
                        logger.debug(f"Added case: {case_data.get('case_name')}")
                
                # Update progress
                pages_fetched += 1
                progress_bar.update(1)
                logger.info(f"Fetched page {pages_fetched}/{max_pages} "
                          f"({len(self.tech_cases)} tech cases found)")
                
                # Check for next page
                next_url = response.get('next')
                if not next_url:
                    logger.info("Reached end of available pages")
                    break
                
                # Extract cursor from next URL
                cursor_match = re.search(r'cursor=([^&]+)', next_url)
                if not cursor_match:
                    logger.warning("Could not find cursor in next URL")
                    break
                    
                cursor = cursor_match.group(1)
                sleep(2)  # Rate limiting delay
                
            except Exception as e:
                logger.error(f"Error fetching page: {str(e)}")
                break
                
        progress_bar.close()
                

    def _enrich_case_data(self, case_data: Dict) -> Dict:
        """
        Enrich case data with additional information from related endpoints.
        
        Args:
            case_data: Base case data dictionary
            
        Returns:
            Enriched case data dictionary
        """
        try:
            enriched_data = case_data.copy()

            # Fetch additional cluster information if available
            if case_data.get('cluster_id'):
                try:
                    cluster_data = self.client.get_clusters({'id': case_data['cluster_id']})
                    if cluster_data:
                        enriched_data.update({
                            'precedential_status': cluster_data.get('precedential_status', 
                                                                  case_data['precedential_status']),
                            'citation_count': cluster_data.get('citation_count', 
                                                             case_data['citation_count']),
                            'judges': cluster_data.get('judges', ''),
                            'nature_of_suit': cluster_data.get('nature_of_suit', 
                                                             case_data['nature_of_suit'])
                        })
                except Exception as e:
                    logger.warning(f"Error fetching cluster data: {str(e)}")

            # Fetch additional docket information if available
            if case_data.get('docket_id'):
                try:
                    docket_data = self.client.get_dockets({'id': case_data['docket_id']})
                    if docket_data:
                        enriched_data.update({
                            'nature_of_suit': docket_data.get('nature_of_suit', 
                                                            enriched_data['nature_of_suit']),
                            'cause': docket_data.get('cause', enriched_data['cause']),
                            'jurisdiction_type': docket_data.get('jurisdiction_type', 
                                                               enriched_data['jurisdiction_type']),
                            'jury_demand': docket_data.get('jury_demand', 
                                                         self.empty_field_placeholder),
                            'assigned_to': docket_data.get('assigned_to_str', 
                                                         enriched_data['assigned_to'])
                        })
                except Exception as e:
                    logger.warning(f"Error fetching docket data: {str(e)}")

            return enriched_data

        except Exception as e:
            logger.error(f"Error enriching case data: {str(e)}")
            return case_data

    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the raw data with enhanced features.
        
        Args:
            df: Raw DataFrame of cases
            
        Returns:
            Processed DataFrame with additional features
        """
        try:
            logger.info("Processing raw data...")
            
            # Fill missing values
            df = df.fillna(self.empty_field_placeholder)
            
            # Convert dates
            date_columns = ['date_filed', 'date_created', 'date_modified']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
            
            # Process text fields
            text_columns = ['opinion_text', 'text_excerpt']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: x if x != self.empty_field_placeholder 
                                          else self.empty_field_placeholder)
            
            # Create category encodings
            categorical_columns = {
                'court': df['court'].unique(),
                'nature_of_suit': df['nature_of_suit'].unique(),
                'precedential_status': df['precedential_status'].unique(),
                'jurisdiction_type': df['jurisdiction_type'].unique()
            }
            
            for col, unique_vals in categorical_columns.items():
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df[f'{col}_encoded'] = df[col].map(mapping)
            
            # Add derived features
            df['citation_count'] = pd.to_numeric(df['citation_count'], errors='coerce').fillna(0)
            df['citation_count_log'] = np.log1p(df['citation_count'])
            
            # Add tech relevance features
            df['tech_keyword_count'] = df['opinion_text'].apply(
                lambda x: len([kw for kw in self.tech_keywords['primary'] 
                             if kw in x.lower()]) if x != self.empty_field_placeholder else 0
            )
            
            # Sort by tech relevance and citation count
            df = df.sort_values(['tech_relevance_score', 'citation_count'], 
                              ascending=[False, False])
            
            logger.info(f"Processed {len(df)} cases successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error processing raw data: {str(e)}")
            return df