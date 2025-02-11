from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from data_prep import TechLawDataPrep

class TechLawModel:
    def __init__(self, data_prep: TechLawDataPrep):
        self.data_prep = data_prep
        # Adjust parameters for imbalanced classes
        self.model = RandomForestClassifier(
            n_estimators=200,  # More trees
            max_depth=15,      # Slightly deeper trees
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced_subsample',  # Better for imbalanced data
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train model with cross-validation."""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import precision_recall_fscore_support
        
        # Perform cross-validation
        cv_scores = []
        pr_scores = []  # Track precision and recall
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            # Split data
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Train on this fold
            self.model.fit(X_fold_train, y_fold_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_fold_val)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_fold_val, y_pred, average='binary'
            )
            
            cv_scores.append(f1)
            pr_scores.append((precision, recall))
            
            print(f"\nFold {fold}:")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            
        print("\nCross-validation scores:")
        print(f"Mean F1: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        print(f"Mean Precision: {np.mean([p for p, r in pr_scores]):.4f}")
        print(f"Mean Recall: {np.mean([r for p, r in pr_scores]):.4f}")
        
        # Train final model on full training set
        print("\nTraining final model on full training set...")
        self.model.fit(X_train, y_train)

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Perform grid search for hyperparameter tuning."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        print("\nBest parameters:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model with focus on imbalanced metrics."""
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            precision_recall_curve, average_precision_score
        )
        
        # Get predictions and probabilities
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        
        # Calculate average precision
        ap_score = average_precision_score(y_test, y_prob)
        
        results = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'average_precision': ap_score,
            'precision_recall': (precision, recall, thresholds),
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        print("\nDetailed Evaluation Metrics:")
        print(f"Average Precision Score: {ap_score:.4f}")
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        print("\nClassification Report:")
        print(results['classification_report'])
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get most important features."""
        feature_names = self.data_prep.get_feature_names()
        importances = self.model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        return feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance with better formatting."""
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=importance_df,
            x='importance',
            y='feature',
            color='purple'  # Use single color instead of palette
        )
        plt.title('Top Feature Importance')
        plt.xlabel('Relative Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
        # Print actual values
        print("\nTop feature importances:")
        for _, row in importance_df.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

    
    def save_model(self, path: str):
        """Save model and data preparation pipeline."""
        model_data = {
            'model': self.model,
            'data_prep': self.data_prep
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'TechLawModel':
        """Load saved model."""
        model_data = joblib.load(path)
        model = cls(model_data['data_prep'])
        model.model = model_data['model']
        return model

def main():
    # Initialize data preparation
    data_prep = TechLawDataPrep("../datasets/all_tech_cases_5year.csv")
    
    # Prepare features and target
    X, y = data_prep.prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
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
    model.save_model("../models/tech_law_model.joblib")

if __name__ == "__main__":
    main()