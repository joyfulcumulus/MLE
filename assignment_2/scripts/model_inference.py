import argparse
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
import pickle

import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator, TransformerMixin

import xgboost as xgb

# Import CreditProcessor 
from model_train_credit_processor import CreditProcessor


# to call this script: python model_inference.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_09_01.pkl"

def main(snapshotdate, modelname):
    print('\n---starting job---\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("model_inference") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    try: 
        # --- set up config ---
        config = {}
        config["snapshot_date_str"] = snapshotdate
        config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
        config["model_name"] = modelname
        config["model_bank_directory"] = "model_bank/"
        config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
        
        pprint.pprint(config)
        
    
        # --- load model artefact from model bank ---
        # Load the model from the pickle file
        with open(config["model_artefact_filepath"], 'rb') as file:
            model_artefact = pickle.load(file)
        
        print("Model loaded successfully! " + config["model_artefact_filepath"])
    
    
        # --- load feature store ---
        # connect to silver attributes table
        folder_path = "datamart/silver/attr/"
        
        # read specific parquet file for that snapshot_date_str
        attributes_sdf = spark.read.parquet(folder_path + 'silver_attr_mthly_' + config["snapshot_date_str"].replace("-", "_") + '.parquet')
        
        # take only important features
        attributes_cols = ['Customer_ID', 'Age', 'Occupation', 'snapshot_date']
        attributes_sdf_subset = attributes_sdf[attributes_cols]
        print("attributes row_count:",attributes_sdf_subset.count())
        
        # connect to silver financials table
        folder_path = "datamart/silver/fin/"
        
        # read specific parquet file for that snapshot_date_str
        financials_sdf = spark.read.parquet(folder_path + 'silver_fin_mthly_' + config["snapshot_date_str"].replace("-", "_") + '.parquet')
        
        # take only important features
        financials_cols = [
            'Customer_ID', 'Annual_Income', 'Monthly_Inhand_Salary',
            'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
            'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
            'Credit_Mix', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
            'Total_EMI_per_month', 'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance',
            'Num_Fin_Pdts', 'Loans_per_Credit_Item', 'Debt_to_Salary', 'EMI_to_Salary', 'Repayment_Ability', 'Loan_Extent'
        ]
        financials_sdf_subset = financials_sdf[financials_cols]
        print("financials row_count:",financials_sdf_subset.count())
    
    
        # --- preprocess data for modeling ---
        # Merge attributes and financials into 1 table (no labels at this point in time)
        # use inner join coz all customer ID records must have all features from both tables to make inference
        merged_df = attributes_sdf_subset.select([col(c) for c in attributes_sdf_subset.columns]) # make a fresh copy of one table
        merged_df = merged_df.join(financials_sdf_subset, on="Customer_ID", how="inner")
        
        # Check size of resultant table. 
        print(f"merged_df row_count: {merged_df.count()}")
        
        # Convert to Python pandas, prepare data for modeling
        merged_df = merged_df.toPandas()
        
        # After merging successfully, remove Customer_ID (join key) as it is not a feature
        merged_df_clean = merged_df.drop(columns=['Customer_ID', 'snapshot_date'])
        
        # Apply data processing steps from saved transformers
        transformer_processor = model_artefact['preprocessing_transformers']['credit_cleaner']
        transformer_ohe = model_artefact['preprocessing_transformers']['one_hot_encoder']
        cat_cols = model_artefact['preprocessing_transformers']['one_hot_encoder_columns']
        
        # data cleaning
        X_inference = transformer_processor.transform(merged_df_clean)
        
        # one hot encoding
        X_inference_cat = transformer_ohe.transform(X_inference[cat_cols])
        X_inference_cat_df = pd.DataFrame(X_inference_cat, columns=transformer_ohe.get_feature_names_out(cat_cols), index=X_inference.index)
        X_inference_fe = pd.concat([X_inference, X_inference_cat_df], axis=1)
        X_inference_fe = X_inference_fe.drop(columns=cat_cols)
        
        print('X_inference_fe rows: ', X_inference_fe.shape[0])
    
    
    
        # --- model prediction inference ---
        # load model
        model = model_artefact["model"]
        
        # predict model
        y_inference = model.predict_proba(X_inference_fe)[:, 1]
        
        # prepare output
        y_inference_pdf = merged_df[["Customer_ID","snapshot_date"]].copy()
        y_inference_pdf["model_name"] = config["model_name"]
        y_inference_pdf["model_predictions"] = y_inference
        
    
    
        # --- save model inference to datamart gold table ---
        # create gold datalake
        gold_directory = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
        print(gold_directory)
        
        if not os.path.exists(gold_directory):
            os.makedirs(gold_directory)
        
        # save gold table - IRL connect to database to write
        partition_name = config["model_name"][:-4] + "_preds_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
        filepath = gold_directory + partition_name
        
        # Convert pandas df to spark df and write to parquet
        spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath)

    finally:
        print("Stopping Spark session...")
        spark.stop()
        print('\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
