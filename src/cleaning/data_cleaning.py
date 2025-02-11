import pandas as pd
import io

def fix_misaligned_csv(input_file, output_file):
    """
    Fixes misaligned CSV data where full_text appears in empty cells below their proper row.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save fixed CSV file
    """
    # Read the CSV but handle it as text first to examine structure
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Use pandas to parse the CSV
    df = pd.read_csv(io.StringIO(data))
    
    # Create a mask for rows where text is in wrong place
    empty_mask = df['full_text'].isna()
    next_row_has_text = empty_mask & df['full_text'].shift(-1).notna()
    
    # For each row where text is misaligned
    for idx in df.index[next_row_has_text]:
        # Move text from next row up to current row
        df.at[idx, 'full_text'] = df.at[idx + 1, 'full_text']
        # Clear the moved text from the next row
        df.at[idx + 1, 'full_text'] = None
    
    # Drop rows that are now empty (had their text moved up)
    df = df.dropna(subset=['full_text'], how='all')
    
    # Save the fixed CSV
    df.to_csv(output_file, index=False)
    
    print(f"Fixed CSV saved to {output_file}")
    return df

if __name__ == "__main__":
    input_file = "../datasets/test_all_tech_cases_5year.csv"
    output_file = "../datasets/test_all_tech_cases_fixed.csv"
    
    fix_misaligned_csv(input_file, output_file)