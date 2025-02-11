# precompute.py
import torch.multiprocessing as mp
from model import LegalCaseAnalyzer
import time

def main():
    # Required for Windows
    mp.freeze_support()
    
    print("Starting pre-computation of model data...")
    start_time = time.time()
    
    try:
        analyzer = LegalCaseAnalyzer.get_instance()
        end_time = time.time()
        print(f"Pre-computation complete! Time taken: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error during pre-computation: {e}")
        raise

if __name__ == "__main__":
    main()