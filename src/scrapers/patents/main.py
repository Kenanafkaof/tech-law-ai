from patent_scraper import USPTOScraper
import pandas as pd
import os 
from dotenv import load_dotenv

load_dotenv("../../../.env")

API_KEY = os.getenv("USPTO_API_KEY")

print(f"Initializing USPTO Scraper with API key: {API_KEY}")

# Initialize scraper
scraper = USPTOScraper()

# 1. Get tech patents data with limit
tech_patents = scraper.get_tech_patents(
    start_date="2020-01-01",
    end_date="2025-02-05",
    tech_centers=["2100", "2400"]  # Software & Communications
)[:5000]  # Limit to 5000 records

# Process the tech patents data
if not tech_patents.empty:
    processed_tech_patents = scraper.process_patent_data(tech_patents)
    scraper.save_to_csv(processed_tech_patents, "tech_patents_training")

# 2. Get citation-specific patents with limit
query = scraper.build_query(
    date_range=("2020-01-01", "2025-02-05"),
    tech_center="2400",
    citation_category="X"
)
citation_patents = scraper.search_patents(query, rows=2000)  # Limit to 2000 records

# Process citation patents data
if not citation_patents.empty:
    processed_citation_patents = scraper.process_patent_data(citation_patents)
    scraper.save_to_csv(processed_citation_patents, "citation_patents_training")

# 3. Combine datasets if needed
combined_patents = pd.concat([processed_tech_patents, processed_citation_patents]).drop_duplicates()
scraper.save_to_csv(combined_patents, "combined_patents_training")

# Print summary statistics
print(f"\nDataset Summary:")
print(f"Tech Patents: {len(processed_tech_patents)} records")
print(f"Citation Patents: {len(processed_citation_patents)} records")
print(f"Combined Dataset: {len(combined_patents)} records")

# Print field coverage for training
print("\nField Coverage:")
for column in combined_patents.columns:
    non_null = combined_patents[column].count()
    coverage = (non_null / len(combined_patents)) * 100
    print(f"{column}: {coverage:.1f}% coverage ({non_null} non-null values)")