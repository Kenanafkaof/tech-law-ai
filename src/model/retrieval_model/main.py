import os
from pathlib import Path
import numpy as np
from data_prep import TechLawDataPrep
from model import TechLawModel
from sklearn.model_selection import train_test_split
import pandas as pd

import os
from pathlib import Path
import numpy as np
import pandas as pd
from data_prep import TechLawDataPrep
from model import TechLawModel
from sklearn.model_selection import train_test_split

import os
from pathlib import Path
import numpy as np
import pandas as pd
from data_prep import TechLawDataPrep
from model import TechLawModel
from sklearn.model_selection import train_test_split

def main():
    # Load raw data
    df = pd.read_csv("../../datasets/all_tech_cases_5year.csv")
    
    # Remove duplicate rows and reset index on the full DataFrame
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Data shape after removing duplicates: {df.shape}")
    
    # Split raw dataframe
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # Reset index on the split DataFrames
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Initialize and fit data preparation on training data
    data_prep = TechLawDataPrep(data_path="../../datasets/all_tech_cases_5year.csv")
    X_train, y_train = data_prep.fit_transform(train_df)
    
    # Transform test data using the fitted pipeline
    X_test, y_test = data_prep.transform(test_df)
    
    # Initialize and train model
    model = TechLawModel(data_prep)
    model.train(X_train, y_train)
    
    # Evaluate model
    results = model.evaluate(X_test, y_test)
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot feature importance
    model.plot_feature_importance()
    
    # Save model
    model.save_model("../../models/tech_law_model.joblib")

if __name__ == "__main__":
    main()



