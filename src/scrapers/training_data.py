from scraper_extended import EnhancedTechLawScraper
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import pandas as pd

def setup_enhanced_logging():
    """Configure detailed logging for the scraping process."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tech_law_scraper_detailed.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('tech_law_scraper')

def scrape_tech_cases_by_year(scraper, start_year: int, end_year: int, courts: list):
    """Scrape cases year by year to handle large date ranges efficiently."""
    all_cases = []
    logger = logging.getLogger('tech_law_scraper')
    
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        logger.info(f"\nProcessing year {year}...")
        
        # Process each court group separately to avoid timeouts
        for court_group in courts:
            logger.info(f"Processing courts: {court_group}")
            
            try:
                # Update scraper court configuration
                scraper.current_courts = court_group
                
                # Fetch cases for this year and court group
                df_year = scraper.fetch_tech_cases(
                    start_date=start_date,
                    end_date=end_date,
                    max_pages=20,  # Increased for better coverage
                )
                
                if not df_year.empty:
                    all_cases.append(df_year)
                    logger.info(f"Found {len(df_year)} cases for {year} in {court_group}")
                
            except Exception as e:
                logger.error(f"Error processing {year} for courts {court_group}: {str(e)}")
                continue
    
    return pd.concat(all_cases, ignore_index=True) if all_cases else pd.DataFrame()

def main():
    # Setup logging
    logger = setup_enhanced_logging()
    
    # Load environment variables
    load_dotenv("../../.env")
    api_token = os.getenv("API_TOKEN")
    
    if not api_token:
        logger.error("API token not found!")
        return
    
    # Initialize scraper
    scraper = EnhancedTechLawScraper(api_token)
    
    # Define court groups to process separately
    court_groups = [
        ['cafc'],  # Federal Circuit (primary tech/patent court)
        ['cand', 'nysd', 'txed'],  # Major district courts
        ['dcd', 'ded'],  # DC and Delaware
        ['ca9', 'ca2', 'ca5']  # Important circuit courts
    ]
    
    # Calculate date range
    end_year = datetime.now().year
    start_year = end_year - 25
    
    # Process cases
    logger.info(f"Starting tech case collection for {start_year}-{end_year}")
    
    df_all = scrape_tech_cases_by_year(scraper, start_year, end_year, court_groups)
    
    if df_all.empty:
        logger.error("No cases found!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs("../datasets", exist_ok=True)
    
    # Save results
    df_all.to_csv("../datasets/test_all_tech_cases_5year.csv", index=False, encoding='utf-8')
    
    # Save high relevance cases
    df_high_rel = df_all[df_all['tech_relevance_score'] >= 0.2]
    df_high_rel.to_csv("../datasets/test_high_relevance_tech_cases_5year.csv", index=False, encoding='utf-8')
    
    # Print summary statistics
    print("\nScraping Summary:")
    print(f"Total cases found: {len(df_all)}")
    print(f"High relevance cases: {len(df_high_rel)}")
    print("\nTech relevance score distribution:")
    print(df_all['tech_relevance_score'].describe())
    
    # Print court distribution
    print("\nCases by court:")
    print(df_all['court'].value_counts())
    
    # Print year distribution
    print("\nCases by year:")
    df_all['year'] = pd.to_datetime(df_all['date_filed']).dt.year
    print(df_all['year'].value_counts().sort_index())

if __name__ == "__main__":
    main()