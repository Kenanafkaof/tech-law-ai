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
# Convert PDF to text
from io import BytesIO
import pdfplumber
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
                sleep(self.rate_limit_delay)
                
                response = self.session.get(url, params=params, timeout=60)
                
                # Add detailed logging
                logger.info(f"Request URL: {response.url}")
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        result_count = len(data.get('results', []))
                        logger.info(f"Results found: {result_count}")
                        logger.debug(f"First result: {data['results'][0] if result_count > 0 else 'None'}")
                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parse error: {str(e)}")
                        
                elif response.status_code == 400:
                    logger.error(f"Bad request: {response.text}")
                elif response.status_code == 429:
                    logger.warning("Rate limit hit, waiting...")
                    sleep(10)
                    continue
                    
                response.raise_for_status()
                
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                if attempt < retry_count - 1:
                    sleep(retry_delay * (attempt + 1))
                    continue
                    
        return {'results': [], 'count': 0, 'next': None}

    def search_opinions(self, params: Optional[Dict] = None) -> Dict:
        """Search opinions with enhanced params handling."""
        # Log the URL for transparency
        
        return self._make_request("search", params)
    
    def get_clusters(self, params: Optional[Dict] = None) -> Dict:
        """Get opinion clusters with enhanced params handling."""
        return self._make_request("clusters", params)
    
    def get_dockets(self, params: Optional[Dict] = None) -> Dict:
        """Get dockets with enhanced params handling."""
        return self._make_request("dockets", params)
    
    def get_parties(self, params: Optional[Dict] = None) -> Dict:
        return self._make_request("parties", params)

    def get_attorneys(self, params: Optional[Dict] = None) -> Dict:
        return self._make_request("attorneys", params)
    
    def get_opinion_text(self, cluster_id: str) -> Dict:
        """Get full opinion text and details"""
        return self._make_request(f"clusters/{cluster_id}")

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

            opinions_data = result.get('opinions', [{}])[0]
            full_text = self._extract_opinion_text(opinions_data)
            citing_cases = opinions_data.get('cites', [])
            opinion = result.get('opinions', [{}])[0] if result.get('opinions') else {}

            
            if tech_relevance < 0.05:
                return None
                
            # Extract case data with proper ID handling
            case_data = {
                # Core case identification
                'id': str(result.get('cluster_id', '')),
                'case_name': result.get('caseName', ''),
                'case_name_full': result.get('caseNameFull', ''),
                'date_filed': result.get('dateFiled', ''),
                'court': result.get('court', ''),
                'court_id': result.get('court_id', ''),
                'docket_number': result.get('docketNumber', ''),
                'docket_id': result.get('docket_id', ''),
                
                # Judges and attorneys
                'judge': result.get('judge', ''),
                'attorney': result.get('attorney', ''),
                'panel_names': result.get('panel_names', []),
                
                # Case status and citations
                'precedential_status': result.get('status', ''),
                'citation_count': int(result.get('citeCount', 0)),
                'citations': result.get('citation', []),
                'scdb_id': result.get('scdb_id', ''),
                
                # Dates
                'date_filed': result.get('dateFiled', ''),
                'date_argued': result.get('dateArgued', ''),
                'date_reargued': result.get('dateReargued', ''),
                'date_reargument_denied': result.get('dateReargumentDenied', ''),
                
                # Case content
                'procedural_history': result.get('procedural_history', ''),
                'syllabus': result.get('syllabus', ''),
                'posture': result.get('posture', ''),
                'suit_nature': result.get('suitNature', ''),
                
                # Opinion specific data
                'opinion_url': opinion.get('download_url', ''),
                'opinion_sha1': opinion.get('sha1', ''),
                'opinion_type': opinion.get('type', ''),
                'opinion_author_id': opinion.get('author_id', ''),
                'opinion_cites': opinion.get('cites', []),
                'snippet': opinion.get('snippet', ''),
                
                # Text content (will be filled by PDF processor)
                'full_text': self._extract_opinion_text(result.get('opinions', [{}])[0]),
                'tech_relevance_score': 0.0,
                'tech_keywords_found': []
            }

            text_content = ' '.join(filter(None, [
                case_data['case_name'],
                case_data['syllabus'],
                case_data['procedural_history'],
                case_data['full_text']
            ]))

            opinion = result.get('opinions', [{}])[0] 
            case_data.update({
                'opinion_url': opinion.get('download_url', ''),
                'opinion_sha1': opinion.get('sha1', ''),
                'opinion_type': opinion.get('type', ''),
                'opinion_author_id': opinion.get('author_id', ''),
                'opinion_cites': opinion.get('cites', []),
                'snippet': opinion.get('snippet', '')
            })
            
            case_data['tech_relevance_score'] = self._calculate_tech_relevance(text_content)
            case_data['tech_keywords_found'] = self._get_matching_keywords(text_content)

            return case_data
                
        except Exception as e:
            logger.error(f"Error extracting case data: {str(e)}")
            return None

    def _download_and_process_pdf(self, url: str, sha1: str = None) -> str:
        """Download and extract text from PDF with better error handling"""
        try:
            if not url:
                return ''
            
            # Stream PDF
            response = requests.get(url, stream=True, timeout=30, verify=False)
            if response.status_code != 200:
                logger.warning(f"Failed to download PDF: {url}")
                return ''
                
            try:
                with pdfplumber.open(BytesIO(response.content)) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        try:
                            text = page.extract_text()
                            if text:
                                text_parts.append(text)
                        except Exception as e:
                            logger.warning(f"Error extracting page text: {e}")
                            continue
                            
                    return '\n'.join(text_parts)
            except Exception as e:
                logger.error(f"Error processing PDF {url}: {e}")
                return ''
                
        except Exception as e:
            logger.error(f"Error downloading PDF {url}: {e}")
            return ''

    def _clean_text_fields(self, text: str) -> str:
        """Clean and standardize text fields"""
        if not text:
            return ''
        
        # Remove common boilerplate
        text = re.sub(r'Case:\s*\d+-\d+\s*Document:\s*\d+.*?Filed:', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve important punctuation
        text = re.sub(r'[^\w\s\.,;:\-\'\"()]', ' ', text)
        
        return text.strip()

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
        """Fetch technology-related cases with improved querying."""
        try:
            logger.info(f"Starting tech case fetch for dates: {start_date} to {end_date}")
            
            # Clear previous cases
            self.tech_cases = []
            
            # Base parameters
            base_params = {
                'type': 'o',  
                'order_by': 'dateFiled desc'
            }
            
            if start_date:
                base_params['filed_after'] = start_date
            if end_date:
                base_params['filed_before'] = end_date

            # Process search queries
            search_queries = [
                'technology patent software',
                'artificial_intelligence machine_learning',
                'computer internet digital electronic',
                'data_privacy cybersecurity encryption'
            ]
            
            for query in search_queries:
                logger.info(f"Processing query: {query}")
                search_params = base_params.copy()
                search_params['q'] = query
                
                if hasattr(self, 'current_court'):
                    search_params['court'] = self.current_court
                    
                self._fetch_cases_with_pagination(search_params, max_pages // len(search_queries))
                sleep(2)
                    
            # Convert to DataFrame with proper structure
            df = self.to_dataframe()
            
            if len(df) == 0:
                logger.warning("No cases found matching criteria")
                return pd.DataFrame()
                
            # Filter by tech relevance
            df = df[df['tech_relevance_score'] >= min_tech_relevance]
            
            logger.info(f"Found {len(df)} cases after filtering")
            return df
                
        except Exception as e:
            logger.error(f"Error in fetch_tech_cases: {str(e)}")
            return pd.DataFrame()
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert cases to DataFrame with consistent structure"""
        # Convert cases to DataFrame
        df = pd.DataFrame(self.tech_cases)
        
        # Ensure all required columns exist
        required_columns = [
            'id', 'case_name', 'date_filed', 'court', 'court_id',
            'docket_number', 'judge', 'precedential_status',
            'citation_count', 'text_excerpt', 'full_text',
            'tech_relevance_score', 'tech_keywords_found'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
                
        # Convert dates to datetime
        date_cols = ['date_filed', 'date_argued', 'date_reargued']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
        # Convert lists to strings
        list_cols = ['tech_keywords_found', 'opinion_cites', 'citing_cases']
        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x) if x else '[]')
                
        df = df.sort_values('id').reset_index(drop=True)
        
        return df

    def _fetch_cases_with_pagination(self, params: Dict, max_pages: int):
        """Fetch cases with improved pagination handling"""
        pages_fetched = 0
        cursor = None
        
        with tqdm(total=max_pages, desc="Fetching tech law cases") as progress_bar:
            while pages_fetched < max_pages:
                try:
                    current_params = params.copy()
                    if cursor:
                        current_params['cursor'] = cursor
                    
                    # Make request
                    response = self.client.search_opinions(current_params)
                    
                    if not response or not isinstance(response, dict):
                        logger.error(f"Invalid response: {response}")
                        break
                        
                    results = response.get('results', [])
                    # Print the URL for transparency
                    logger.info(f"Page {pages_fetched + 1}: Found {len(results)} results")
                    
                    # Process results
                    for result in results:
                        case_data = self._extract_case_data(result)
                        if case_data and self._validate_case_data(case_data):
                            # Process PDF if available
                            if case_data['opinion_url']:
                                full_text = self._download_and_process_pdf(
                                    case_data['opinion_url'],
                                    case_data['opinion_sha1']
                                )
                                case_data['full_text'] = full_text
                                
                                # Recalculate tech relevance with full text
                                if full_text:
                                    case_data['tech_relevance_score'] = self._calculate_tech_relevance(full_text)
                                    case_data['tech_keywords_found'] = self._get_matching_keywords(full_text)
                            
                            self.tech_cases.append(case_data)
                    
                    pages_fetched += 1
                    progress_bar.update(1)
                    
                    # Check for next page
                    next_url = response.get('next')
                    if not next_url:
                        logger.info("No more pages available")
                        break
                    
                    # Try to get cursor from next URL
                    try:
                        # URL decode the cursor
                        from urllib.parse import unquote, parse_qs, urlparse
                        parsed_url = urlparse(next_url)
                        query_params = parse_qs(parsed_url.query)
                        if 'cursor' in query_params:
                            cursor = query_params['cursor'][0]
                            # Make sure cursor is properly decoded
                            cursor = unquote(cursor)
                        else:
                            logger.warning("No cursor found in next URL")
                            break
                    except Exception as e:
                        logger.error(f"Error parsing next URL: {str(e)}")
                        break
                    
                    sleep(2)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error fetching page: {str(e)}")
                    break



                
    def _enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for ML"""
        
        # Case duration (if dates available)
        df['case_duration'] = pd.to_datetime(df['date_terminated']) - pd.to_datetime(df['date_filed'])
        
        # Citation impact
        df['has_citations'] = df['citation_count'] > 0
        df['citation_score'] = np.log1p(df['citation_count'])
        
        # Court level features
        df['is_appellate'] = df['court_id'].isin(['ca1', 'ca2', 'ca3', 'ca4', 'ca5', 'ca6', 'ca7', 'ca8', 'ca9', 'ca10', 'ca11', 'cadc', 'cafc'])
        
        # Text length features
        df['opinion_length'] = df['text_excerpt'].str.len()
        
        # Categorical encodings
        for col in ['court_id', 'precedential_status', 'nature_of_suit']:
            df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
        
        return df

    def _validate_case_data(self, case_data: Dict) -> bool:
        """Validate and fill missing fields"""
        required_fields = {
            'id': '',
            'case_name': '',
            'date_filed': None,
            'court': '',
            'court_id': '',
            'docket_number': '',
            'judge': '',
            'precedential_status': '',
            'citation_count': 0,
            'docket_id': None,
            'cluster_id': None, 
            'text_excerpt': '',
            'tech_relevance_score': 0.0,
            'tech_keywords_found': [],
            'suit_nature': '',
            'procedural_history': '',
            'full_text': '',
            'opinion_url': '',
            'opinion_sha1': '',
            'opinion_type': '',
            'opinion_cites': [],
            'citing_cases': [],
            'cited_by': []
        }
        
        # Fill missing fields with defaults
        for field, default in required_fields.items():
            if field not in case_data or case_data[field] is None:
                case_data[field] = default
                
        # Core fields must have values
        return bool(case_data.get('case_name')) and bool(case_data.get('court_id'))

    def _enrich_case_data(self, case_data: Dict) -> Dict:
        """
        Enhanced data enrichment to capture additional features
        """
        enriched_data = case_data.copy()
        
        # Get docket details with disposition
        if case_data.get('docket_id'):
            docket_params = {
                'id': case_data['docket_id'],
                'fields': [
                    'assignedTo',
                    'referredTo', 
                    'cause',
                    'nature_of_suit',
                    'jury_demand',
                    'jurisdiction_type',
                    'date_filed',
                    'date_terminated',
                    'date_last_filing',
                    'case_name_full',
                    'appellate_fee_status',
                    'appellate_case_type_information'
                ]
            }
            docket_data = self.client.get_dockets(docket_params)
            
            # Extract party information from case name
            parties = self._parse_case_name(docket_data.get('case_name_full', ''))
            
            case_data.update({
                'assigned_judge': docket_data.get('assignedTo'),
                'case_type': docket_data.get('nature_of_suit'),
                'jurisdiction': docket_data.get('jurisdiction_type'),
                'appellate_status': docket_data.get('appellate_fee_status'),
                'parties': parties,
                'procedural_events': self._extract_procedural_history(docket_data)
            })
        
        # Get party details
        if enriched_data.get('parties'):
            party_details = []
            for party in enriched_data['parties']:
                party_params = {'id': party['id']}
                party_data = self.client.get_parties(party_params)
                if party_data:
                    party_details.append({
                        'name': party_data.get('name'),
                        'type': party_data.get('type'),
                        'role': party_data.get('role'),
                        'attorneys': party_data.get('attorneys', [])
                    })
            enriched_data['party_details'] = party_details
        
        # Get related cases from clusters
        if case_data.get('cluster_id'):
            cluster_params = {
                'id': case_data['cluster_id'],
                'fields': 'citing,cited_by,precedential_status'
            }
            cluster_data = self.client.get_clusters(cluster_params)
            if cluster_data:
                enriched_data.update({
                    'citing_cases': cluster_data.get('citing', []),
                    'cited_by_cases': cluster_data.get('cited_by', []),
                    'precedential_status': cluster_data.get('precedential_status')
                })
        
        return enriched_data
    
    def _extract_procedural_history(self, docket_data: Dict) -> List[Dict]:
        """Extract procedural history from docket entries"""
        history = []
        
        # Get key dates
        key_dates = {
            'filed': docket_data.get('date_filed'),
            'terminated': docket_data.get('date_terminated'),
            'last_filing': docket_data.get('date_last_filing'),
            'argued': docket_data.get('date_argued'),
            'reargued': docket_data.get('date_reargued')
        }
        
        for date_type, date in key_dates.items():
            if date:
                history.append({
                    'date': date,
                    'event_type': date_type,
                    'description': f'Case {date_type}'
                })
        
        return sorted(history, key=lambda x: x['date'])

    def _parse_case_name(self, case_name: str) -> Dict:
        """Extract party information from case name"""
        parties = {'plaintiff': '', 'defendant': ''}
        if ' v. ' in case_name:
            plaintiff, defendant = case_name.split(' v. ')
            parties['plaintiff'] = plaintiff.strip()
            parties['defendant'] = defendant.strip()
        return parties
    
    def _process_party_data(self, party_data: Dict) -> Dict:
        """Process party-specific information"""
        return {
            'name': party_data.get('name', self.empty_field_placeholder),
            'role': party_data.get('role', self.empty_field_placeholder),
            'type': party_data.get('type', self.empty_field_placeholder),
            'attorneys': [att.get('name') for att in party_data.get('attorneys', [])]
        }

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