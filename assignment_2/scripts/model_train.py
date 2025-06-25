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

# to call this script: python model_train.py --snapshotdate "2024-09-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("model_train_main") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # --- set up config ---
        model_train_date_str = snapshotdate
        train_test_period_months = 12
        oot_period_months = 2
        train_test_ratio = 0.8
        
        config = {}
        config["model_train_date_str"] = model_train_date_str
        config["train_test_period_months"] = train_test_period_months
        config["oot_period_months"] =  oot_period_months
        config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
        config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
        config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
        config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
        config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
        config["train_test_ratio"] = train_test_ratio 
        pprint.pprint(config)
    
    
        
        # --- get label ---
        # connect to label store
        folder_path = "datamart/gold/label_store/"
        files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
        label_store_sdf = spark.read.parquet(*files_list)
        print("label_store row_count:",label_store_sdf.count())
        label_store_sdf.show()
        
        # extract label store
        labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))
        print("extracted labels_sdf rows:", labels_sdf.count())
        print("train_test_start_date: ", config["train_test_start_date"])
        print("oot_end_date: ", config["oot_end_date"])
    
    
        
        # --- get features ---
        # connect to silver attributes table
        folder_path = "datamart/silver/attr/"
        files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
        attributes_sdf = spark.read.parquet(*files_list)
        print("silver_attributes row_count:",attributes_sdf.count())
        attributes_sdf_subset = attributes_sdf.filter(col("snapshot_date") <= config["oot_end_date"])
        print(f"extracted attributes_sdf_subset rows up to {config['oot_end_date']}:", attributes_sdf_subset.count())
        attributes_cols = ['Customer_ID', 'Age', 'Occupation']
        attributes_sdf_subset = attributes_sdf_subset[attributes_cols]
        attributes_sdf_subset.show(5)
        
        # connect to silver financials table
        folder_path = "datamart/silver/fin/"
        files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
        financials_sdf = spark.read.parquet(*files_list)
        print("silver_financials row_count:",financials_sdf.count())
        financials_sdf_subset = financials_sdf.filter(col("snapshot_date") <= config["oot_end_date"])
        print(f"extracted financials_sdf_subset rows up to {config['oot_end_date']}:", financials_sdf_subset.count())
        financials_cols = [
            'Customer_ID', 'Annual_Income', 'Monthly_Inhand_Salary',
            'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
            'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
            'Credit_Mix', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
            'Total_EMI_per_month', 'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance',
            'Num_Fin_Pdts', 'Loans_per_Credit_Item', 'Debt_to_Salary', 'EMI_to_Salary', 'Repayment_Ability', 'Loan_Extent'
        ]
        financials_sdf_subset = financials_sdf_subset[financials_cols]
        financials_sdf_subset.show(5)
    
    
        
        # --- prepare data for modeling ---
        # Merge label data to attributes and financials into 1 table
        merged_df = labels_sdf.select([col(c) for c in labels_sdf.columns]) # make a fresh copy
        merged_df = merged_df.join(attributes_sdf_subset, on="Customer_ID", how="left")
        merged_df = merged_df.join(financials_sdf_subset, on="Customer_ID", how="left")
        print(f"Merged_df row count: {merged_df.count()}")
        merged_df = merged_df.toPandas()
        merged_df_clean = merged_df.drop(columns=['Customer_ID','loan_id', 'label_def'])
        
        # split data into train - test - oot
        oot_pdf = merged_df_clean[(merged_df_clean['snapshot_date'] >= config["oot_start_date"].date()) & (merged_df_clean['snapshot_date'] <= 
                                                                                                           config["oot_end_date"].date())]
        train_test_pdf = merged_df_clean[(merged_df_clean['snapshot_date'] >= config["train_test_start_date"].date()) &(merged_df_clean['snapshot_date'] <= config["train_test_end_date"].date())]
        
        feature_cols = ['Age', 'Occupation', 'Annual_Income',
               'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
               'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
               'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
               'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
               'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
               'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance', 
                'Num_Fin_Pdts', 'Loans_per_Credit_Item', 'Debt_to_Salary',
               'EMI_to_Salary', 'Repayment_Ability', 'Loan_Extent']
        
        X_oot = oot_pdf[feature_cols]
        y_oot = oot_pdf["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            train_test_pdf[feature_cols], train_test_pdf["label"], 
            test_size= 1 - config["train_test_ratio"],
            random_state=42,     # Ensures reproducibility
            shuffle=True,        # Shuffle the data before splitting
            stratify=train_test_pdf["label"]           # Stratify based on the label column as target is imbalanced
        )
        
        print('X_train', X_train.shape[0])
        print('X_test', X_test.shape[0])
        print('X_oot', X_oot.shape[0])
        print('y_train', y_train.shape[0], round(y_train.mean(),2))
        print('y_test', y_test.shape[0], round(y_test.mean(),2))
        print('y_oot', y_oot.shape[0], round(y_oot.mean(),2))
        
        # Use imported custom data processing class
        processor = CreditProcessor()
        
        X_train_clean = processor.fit_transform(X_train)  # thresholds learned here
        
        # Apply cleaning logic to validation and test set
        X_test_clean = processor.transform(X_test)           # thresholds reused
        X_oot_clean = processor.transform(X_oot)         # thresholds reused
        
        
        
        # --- model specific feature engineering ---
        # implement one hot encoding manually here to avoid unseen category problem in val/test set
        cat_cols = ['Credit_Mix', 'Payment_Behaviour']
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        X_train_cat = ohe.fit_transform(X_train_clean[cat_cols])
        X_train_cat_df = pd.DataFrame(X_train_cat, columns=ohe.get_feature_names_out(cat_cols), index=X_train_clean.index)
        X_train_fe = pd.concat([X_train_clean, X_train_cat_df], axis=1)
        X_train_fe = X_train_fe.drop(columns=cat_cols)
        
        # Repeat for X_test_clean, X_oot_clean using the above functions
        X_test_cat = ohe.transform(X_test_clean[cat_cols])
        X_test_cat_df = pd.DataFrame(X_test_cat, columns=ohe.get_feature_names_out(cat_cols), index=X_test_clean.index)
        X_test_fe = pd.concat([X_test_clean, X_test_cat_df], axis=1)
        X_test_fe = X_test_fe.drop(columns=cat_cols)
        
        # X_oot_clean
        X_oot_cat = ohe.transform(X_oot_clean[cat_cols])
        X_oot_cat_df = pd.DataFrame(X_oot_cat, columns=ohe.get_feature_names_out(cat_cols), index=X_oot_clean.index)
        X_oot_fe = pd.concat([X_oot_clean, X_oot_cat_df], axis=1)
        X_oot_fe = X_oot_fe.drop(columns=cat_cols)
        
        
        # --- train model ---
        # Define hyperparameter tuned model that was found through random search
        # This was done in XGBoost 3.0.0 but due to incompatibility with SKLearn random search, we cant use it here
        
        # Define hyper parameter tuned model with params from XGBoost 3.0.0
        best_model = xgb.XGBClassifier(
            colsample_bytree=0.8,
            gamma=0,
            learning_rate=0.1,
            max_depth=3,
            min_child_weight=5,
            n_estimators=200,
            reg_alpha=0,
            reg_lambda=2,
            subsample=0.8,
            eval_metric='logloss',
            random_state=42
        )
        
        # Fit on training data
        best_model.fit(X_train_fe, y_train)
        
        # Evaluate on train set
        y_train_proba = best_model.predict_proba(X_train_fe)[:, 1]
        train_auc_score = roc_auc_score(y_train, y_train_proba)
        
        # Evaluate on test set
        y_test_proba = best_model.predict_proba(X_test_fe)[:, 1]
        test_auc_score = roc_auc_score(y_test, y_test_proba)
        
        # Evaluate on OOT set
        y_oot_proba = best_model.predict_proba(X_oot_fe)[:, 1]
        oot_auc_score = roc_auc_score(y_oot, y_oot_proba)
        
        # Print results
        print("Tuned Model Performance (based on best model derived from v3.0.0):")
        print(f"Train AUC: {train_auc_score:.4f} | GINI: {2 * train_auc_score - 1:.4f}")
        print(f"Test AUC: {test_auc_score:.4f}  | GINI: {2 * test_auc_score - 1:.4f}")
        print(f"OOT AUC: {oot_auc_score:.4f}   | GINI: {2 * oot_auc_score - 1:.4f}")
        
        
        # --- prepare model artefact to save ---
        model_artefact = {}
        
        model_artefact['model'] = best_model
        model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
        model_artefact['preprocessing_transformers'] = {}
        model_artefact['preprocessing_transformers']['credit_cleaner'] = processor
        model_artefact['preprocessing_transformers']['one_hot_encoder'] = ohe
        model_artefact['preprocessing_transformers']['one_hot_encoder_columns'] = cat_cols
        model_artefact['data_dates'] = config
        model_artefact['data_stats'] = {}
        model_artefact['data_stats']['X_train'] = X_train.shape[0]
        model_artefact['data_stats']['X_test'] = X_test.shape[0]
        model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
        model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
        model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
        model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
        model_artefact['results'] = {}
        model_artefact['results']['auc_train'] = train_auc_score
        model_artefact['results']['auc_test'] = test_auc_score
        model_artefact['results']['auc_oot'] = oot_auc_score
        model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
        model_artefact['results']['gini_test'] = round(2*test_auc_score-1,3)
        model_artefact['results']['gini_oot'] = round(2*oot_auc_score-1,3)
        model_artefact['hp_params'] = best_model.get_params()
        
        pprint.pprint(model_artefact)
        
        
        # --- save artefact to model bank ---
        # create model_bank dir
        model_bank_directory = "model_bank/"
        
        if not os.path.exists(model_bank_directory):
            os.makedirs(model_bank_directory)
        
        # Full path to the file
        file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')
        
        # Write the model to a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(model_artefact, file)
        
        print(f"Model saved to {file_path}")
        
        
        # --- test load pickle and make model inference ---    
        # Load the model from the pickle file
        with open(file_path, 'rb') as file:
            loaded_model_artefact = pickle.load(file)
        
        y_pred_proba = loaded_model_artefact['model'].predict_proba(X_oot_fe)[:, 1]
        oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
        print("OOT AUC score: ", oot_auc_score)
        print("Model loaded successfully!")

        # --- use model_artefact to make inference on train_test_end_date and save into model_bank as baseline (for PSI Metric) ---
        # reformat the date in config["train_test_end_date"]
        formatted_date = config["train_test_end_date"].strftime('%Y_%m') + '_01'
        
        # connect to silver attributes table
        folder_path = "datamart/silver/attr/"
        
        # read specific parquet file for that train_test_end_date
        attributes_sdf = spark.read.parquet(folder_path + 'silver_attr_mthly_' + formatted_date + '.parquet')
        
        # take only important features
        attributes_cols = ['Customer_ID', 'Age', 'Occupation', 'snapshot_date']
        attributes_sdf_subset = attributes_sdf[attributes_cols]
        print("attributes row_count:",attributes_sdf_subset.count())
        
        # connect to silver financials table
        folder_path = "datamart/silver/fin/"
        
        # read specific parquet file for that train_test_end_date
        financials_sdf = spark.read.parquet(folder_path + 'silver_fin_mthly_' + formatted_date + '.parquet')
        
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
        
        # load model and predict
        model = model_artefact["model"]
        y_inference = model.predict_proba(X_inference_fe)[:, 1]
        
        # prepare output
        y_inference_pdf = merged_df[["Customer_ID","snapshot_date"]].copy()
        y_inference_pdf["model_name"] = model_artefact['model_version'] + '.pkl'
        y_inference_pdf["model_predictions"] = y_inference
        
        # save baseline prediction into model_bank 
        partition_name = model_artefact['model_version'] + "_psi_ref_preds" + '.parquet'
        filepath = model_bank_directory + partition_name
        
        # Convert pandas df to spark df and write to parquet
        spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
        print('PSI baseline prediction saved to:', filepath)  
    
        print('\n\n---completed job---\n\n')

    finally:
        print("Stopping Spark session...")
        spark.stop()
    
if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)
