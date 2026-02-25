import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

class FailureRiskModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        )
        self.feature_cols = [
            'phase', 'enrollment_size', 'duration_months', 
            'num_primary_endpoints', 'indication_encoded',
            'sponsor_type_encoded', 'is_adaptive'
        ]

    def train(self, data_path: str):
        print(f"Loading data from {data_path}...")
        # Mock data loading
        df = pd.DataFrame({
            'phase': np.random.choice([1, 2, 3], 1000),
            'enrollment_size': np.random.randint(20, 1000, 1000),
            'duration_months': np.random.randint(6, 60, 1000),
            'num_primary_endpoints': np.random.randint(1, 5, 1000),
            'indication_encoded': np.random.randint(0, 10, 1000),
            'sponsor_type_encoded': np.random.randint(0, 3, 1000),
            'is_adaptive': np.random.choice([0, 1], 1000),
            'outcome': np.random.choice([0, 1], 1000) # 1 = failure, 0 = success
        })

        X = df[self.feature_cols]
        y = df['outcome']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training Gradient Boosting model...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print(f"Model trained. AUC: {auc:.4f}")
        
        joblib.dump(self.model, 'failure_risk_model.pkl')

    def predict(self, features: dict):
        # Load model if not loaded
        # model = joblib.load('failure_risk_model.pkl')
        # Placeholder prediction
        return np.random.random()

if __name__ == "__main__":
    model = FailureRiskModel()
    model.train("data/processed/trial_outcomes.csv")
