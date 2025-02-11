from main import EnhancedTechLawScraper
import os
from dotenv import load_dotenv
import logging

# Enable debug logging
logging.getLogger('main').setLevel(logging.DEBUG)

load_dotenv("../../.env")
api_token = os.getenv("API_TOKEN")

scraper = EnhancedTechLawScraper(api_token)

# Fetch cases with lower threshold for testing
df = scraper.fetch_tech_cases(
    start_date="2000-02-01",
    end_date="2025-02-01",
    max_pages=10,
    min_tech_relevance=0.1  # Lower threshold for initial testing
)

# Save both full results and filtered results
df.to_csv("../datasets/all_tech_cases.csv", index=False, encoding='utf-8')
df[df['tech_relevance_score'] >= 0.2].to_csv("../datasets/high_relevance_tech_cases.csv", index=False, encoding='utf-8')

# Print summary
print(f"\nTotal cases found: {len(df)}")
print(f"High relevance cases: {len(df[df['tech_relevance_score'] >= 0.2])}")
print("\nTech relevance score distribution:")
print(df['tech_relevance_score'].describe())