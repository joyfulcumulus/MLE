import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Create custom data processing pipeline
class CreditProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # To store clipping thresholds after running fit fxn
        self.annual_income_clip = None
        self.emi_to_salary_clip = None

    def fit(self, X, y=None):
        X = X.copy() # make a copy of X that was passed in safer
        # Get winsorize thresholds for specific features
        self.annual_income_clip = np.percentile(X['Annual_Income'].dropna(), [1, 99])
        self.emi_to_salary_clip = np.percentile(X['EMI_to_Salary'].dropna(), [1, 99])
        return self

    def transform(self, X):
        X = X.copy() # make a copy of X that was passed in safer
        # Drop features
        features_to_drop = ['Occupation', 'Age', 'Changed_Credit_Limit', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly']
        X = X.drop(columns=features_to_drop)
        
        # Generate binned features
        X['Is_Interest_Rate_Low'] = (X['Interest_Rate'] <= 15).astype(int)
        X['Is_Credit_History_Age_Low'] = (X['Credit_History_Age'] <= 250).astype(int)

        # Winsorize using fitted thresholds
        X['Annual_Income'] = X['Annual_Income'].clip(
            lower=self.annual_income_clip[0],
            upper=self.annual_income_clip[1]
        )
        X['EMI_to_Salary'] = X['EMI_to_Salary'].clip(
            lower=self.emi_to_salary_clip[0],
            upper=self.emi_to_salary_clip[1]
        )

        # Drop raw columns of binned features
        X = X.drop(columns=['Interest_Rate', 'Credit_History_Age'])
        return X