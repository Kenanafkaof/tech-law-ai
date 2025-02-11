import requests
import pandas as pd
from typing import Dict, List, Optional, Union
import time
from datetime import datetime
import logging
import json

class USPTOScraper:
    """USPTO Data Set API Scraper for enriched citation metadata."""
    
    def __init__(self):
        self.base_url = "https://developer.uspto.gov/ds-api"
        self.dataset = "enriched_cited_reference_metadata"
        self.version = "v3"
        self.fields = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("uspto_scraper")
        
        # Headers based on documentation
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # Available fields from documentation
        self.known_fields = [
            "officeActionDate",
            "relatedClaimNumberText",
            "applicantCitedExaminerReferenceIndicator",
            "createUserIdentifier",
            "kindCode",
            "nplIndicator",
            "workGroupNumber",
            "officeActionCategory",
            "patentApplicationNumber",
            "inventorNameText",
            "groupArtUnitNumber",
            "qualitySummaryText",
            "createDateTime",
            "techCenter",
            "citedDocumentIdentifier",
            "countryCode",
            "passageLocationText",
            "obsoleteDocumentIdentifier",
            "citationCategoryCode",
            "examinerCitedReferenceIndicator",
            "publicationNumber"
        ]

    def get_available_fields(self) -> List[str]:
        """Fetch available fields from the API."""
        url = f"{self.base_url}/{self.dataset}/{self.version}/fields"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            self.fields = data.get('fields', [])
            field_count = data.get('fieldCount', 0)
            
            self.logger.info(f"Retrieved {field_count} searchable fields")
            self.logger.info(f"Last data update: {data.get('lastDataUpdatedDate')}")
            
            return self.fields
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching fields: {str(e)}")
            raise

    def build_query(self, 
                   date_range: Optional[tuple] = None,
                   tech_center: Optional[str] = None,
                   citation_category: Optional[str] = None,
                   inventor: Optional[str] = None) -> str:
        """Build Lucene query string based on parameters."""
        query_parts = []
        
        if date_range:
            start_date, end_date = date_range
            query_parts.append(f"officeActionDate:[{start_date} TO {end_date}]")
            
        if tech_center:
            query_parts.append(f"techCenter:{tech_center}")
            
        if citation_category:
            query_parts.append(f"citationCategoryCode:{citation_category}")
            
        if inventor:
            # Escape special characters in inventor name
            inventor = inventor.replace(':', r'\:').replace(' ', r'\ ')
            query_parts.append(f"inventorNameText:{inventor}")
        
        # Return all records if no filters specified
        return " AND ".join(query_parts) if query_parts else "*:*"

    def search_patents(self, 
                  criteria: str,
                  start: int = 0,
                  rows: int = 100,
                  max_records: int = 5000,  # Add this parameter
                  max_retries: int = 3,
                  retry_delay: int = 5) -> pd.DataFrame:
        """Search patents using Lucene query syntax."""
        url = f"{self.base_url}/{self.dataset}/{self.version}/records"
    
        all_records = []
        total_records = None
        retry_count = 0
        
        while True:
            try:
                # Add limit check here
                if len(all_records) >= max_records:
                    self.logger.info(f"Reached maximum records limit of {max_records}")
                    break
                    
                data = {
                    "criteria": criteria,
                    "start": start,
                    "rows": rows
                }
                
                self.logger.info(f"Fetching records {start} to {start + rows}")
                response = requests.post(url, headers=self.headers, data=data)

                if response.status_code == 404:
                    self.logger.warning("No matching records found")
                    break
                    
                response.raise_for_status()
                
                result = response.json()
                if 'response' not in result:
                    self.logger.error("Invalid API response format")
                    break
                    
                records = result['response'].get('docs', [])
                if not records:
                    break
                    
                total_records = result['response'].get('numFound', 0)
                all_records.extend(records)
                
                if len(all_records) >= total_records:
                    break
                    
                start += rows
                time.sleep(1)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count >= max_retries:
                    self.logger.error(f"Max retries reached. Error: {str(e)}")
                    break
                    
                self.logger.warning(f"Retry {retry_count} after error: {str(e)}")
                time.sleep(retry_delay)
        
        df = pd.DataFrame(all_records)
        self.logger.info(f"Retrieved {len(df)} records out of {total_records} total matches")
        return df

    def get_tech_patents(self,
                        start_date: str,
                        end_date: str,
                        tech_centers: Optional[List[str]] = None) -> pd.DataFrame:
        """Get technology-related patents within a date range."""
        query_parts = [
            f"officeActionDate:[{start_date}T00:00:00Z TO {end_date}T23:59:59Z]"
        ]
        
        if tech_centers:
            tech_center_query = " OR ".join([f"techCenter:{tc}" for tc in tech_centers])
            query_parts.append(f"({tech_center_query})")
        else:
            # Default to technology-focused tech centers
            query_parts.append("(techCenter:2100 OR techCenter:2400 OR techCenter:2600 OR techCenter:2800)")
        
        criteria = " AND ".join(query_parts)
        return self.search_patents(criteria)

    def process_patent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean patent data."""
        if df.empty:
            return df
            
        # Convert dates
        date_cols = ['officeActionDate', 'createDateTime']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Convert boolean flags
        bool_cols = [
            'applicantCitedExaminerReferenceIndicator',
            'examinerCitedReferenceIndicator',
            'nplIndicator'
        ]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Clean text fields
        text_cols = [
            'inventorNameText', 
            'citedDocumentIdentifier',
            'passageLocationText'
        ]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        return df

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save patent data to CSV with timestamp."""
        if df.empty:
            self.logger.warning("No data to save")
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"patent_data_{timestamp}_{filename}.csv"
        df.to_csv(filename, index=False)
        self.logger.info(f"Saved {len(df)} records to {filename}")

def main():
    scraper = USPTOScraper()
    
    # Get available fields first
    fields = scraper.get_available_fields()
    print("Available fields:", fields)
    
    # Example search for tech patents from last 5 years
    tech_patents = scraper.get_tech_patents(
        start_date="2020-01-01",
        end_date="2025-02-05",
        tech_centers=["2100", "2400", "2600", "2800"]
    )
    
    if not tech_patents.empty:
        processed_df = scraper.process_patent_data(tech_patents)
        scraper.save_to_csv(processed_df, "tech_patents")
    else:
        print("No patent data retrieved")

if __name__ == "__main__":
    main()