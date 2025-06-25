import argparse
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
import pickle

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import DateType, StringType, FloatType, StructType, StructField

from sklearn.metrics import recall_score, brier_score_loss

def main(snapshotdate, modelname):
    print('\n---starting job---\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("model_monitoring") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    try: 
        # --- set up config ---
        config = {}
        config["snapshot_date_str"] = snapshotdate # updated from ipynb
        config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
        config["model_name"] = modelname # updated from ipynb
        config["model_bank_directory"] = "model_bank/"
        config["model_psi_ref_preds_filepath"] = config["model_bank_directory"] + config["model_name"][:-4] + "_psi_ref_preds.parquet"
        
        pprint.pprint(config)

        # --- fetch baseline month for psi computation using model name ---
        psi_ref_sdf = spark.read.parquet(config["model_psi_ref_preds_filepath"])
        print("psi_ref_df row_count:",psi_ref_sdf.count())

        # --- fetch predictions and labels based on snapshot_date_str ---
        # format date properly
        formatted_date = config["snapshot_date_str"].replace('-', '_')

        # Fetch the labels first
        # If no label yet, or label count = 0, exit task
        label_directory =  "datamart/gold/label_store/"
        filename = f"gold_label_store_{formatted_date}.parquet"
        file_path = os.path.join(label_directory, filename)
        
        try:
            label_sdf = spark.read.parquet(file_path)
            print("label_sdf row_count:",label_sdf.count())
        
            if label_sdf.count() == 0:
                print(f'Zero label data for {formatted_date}, exiting flow')
                return # early return here, cannot compute metrics (finally block still runs)
        
        except Exception as e:
            print(f'No label data file for {formatted_date}, exiting flow')
            return # early return here, cannot compute metrics (finally block still runs)
        
        # Compute date 6 months ago and format this date
        past_date = config["snapshot_date"] - relativedelta(months=6)
        formatted_past_date = past_date.strftime("%Y_%m_%d")
        
        # Fetch predictions
        model_pred_directory = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
        filename = f"{config['model_name'][:-4]}_preds_{formatted_past_date}.parquet"
        file_path = os.path.join(model_pred_directory, filename)
        
        model_pred_sdf = spark.read.parquet(file_path)
        print("model_pred_sdf row_count:",model_pred_sdf.count())
        
        # --- extract required fields ---
        # Match prediction to label for each Customer_ID so that both in order
        pred_label_sdf = label_sdf.select([col(c) for c in label_sdf.columns]) # make a fresh copy of one table
        pred_label_sdf = pred_label_sdf.join(model_pred_sdf, on="Customer_ID", how="inner")
        
        # Check size of resultant table. 
        print(f"pred_label_sdf row_count: {pred_label_sdf.count()}")
        
        # Convert to pandas df
        psi_ref_df = psi_ref_sdf.toPandas()
        pred_label_df = pred_label_sdf.toPandas()
        
        # Extract relevant data
        y_pred_proba = pred_label_df['model_predictions']
        y_pred = (y_pred_proba >= 0.5).astype(int) # Assume threshold 0.5
        y_true = pred_label_df['label']
        y_pred_proba_ref = psi_ref_df['model_predictions']
        
        # --- compute metrics ---
        def calculate_psi(y_pred_proba_ref, y_pred_proba, buckets=10):
            """
            Calculate the Population Stability Index (PSI) between two distributions.
            Parameters:
                y_pred_proba_ref: reference distribution of model prediction
                y_pred_proba: current distribution of model prediction (for that month)
                buckets: number of quantile bins to split the scores. Default 10 for 10% buckets in classification probabilities
            Returns:
                psi_value: float
            """
            def get_bin_proportions(values, breakpoints):
                counts, _ = np.histogram(values, bins=breakpoints)
                return counts / len(values)
        
            # Extract proba values at 0, 10, 20..., 100th pctile from y_pred_ref
            breakpoints = np.percentile(y_pred_proba_ref, np.linspace(0, 100, buckets + 1))
            breakpoints[0] = -np.inf  # handle outliers below 0
            breakpoints[-1] = np.inf # handle outliers above 1
        
            expected_percents = get_bin_proportions(y_pred_proba_ref, breakpoints)
            actual_percents = get_bin_proportions(y_pred_proba, breakpoints)
        
            # Avoid division by zero or log(0) by giving those with 0 some value
            expected_percents = np.where(expected_percents == 0, 1e-6, expected_percents)
            actual_percents = np.where(actual_percents == 0, 1e-6, actual_percents)
        
            psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
        
            return psi
        
        # Recall
        recall = recall_score(y_true, y_pred, pos_label=1)
        print('recall: ', recall)
        
        # Brier Score
        brier = brier_score_loss(y_true, y_pred_proba)  # proba for class 1
        print('brier score:', brier)
        
        # Population Stability Index
        psi = calculate_psi(y_pred_proba_ref, y_pred_proba)
        print('Population stability index:', psi)

        # save model metrics to model_metrics_tab
        # Define schema of correct data types
        column_type_map = StructType([
            StructField("snapshot_date", DateType(), True),
            StructField("model_version", StringType(), True),
            StructField("recall", FloatType(), True),
            StructField("brier_score", FloatType(), True),
            StructField("psi", FloatType(), True)
        ])
        
        # Create spark dataframe to store results
        results_row = {
            "snapshot_date": config["snapshot_date"],
            "model_version": str(config['model_name']),
            "recall": float(recall),
            "brier_score": float(brier),
            "psi": float(psi)
        }
        
        # Create df with schema enforced
        metrics_df = spark.createDataFrame([results_row], schema=column_type_map)
        
        # Create directory
        gold_metrics_directory = "datamart/gold/model_metrics/"
        if not os.path.exists(gold_metrics_directory):
            os.makedirs(gold_metrics_directory)
        
        # Concat path
        metrics_table_path = f"{gold_metrics_directory}gold_metrics_tab.parquet"
        
        # Check if table exists
        if not os.path.exists(metrics_table_path):
            # create table and save
            metrics_df.write.mode("overwrite").parquet(metrics_table_path)
            print('created gold metrics table and saved results_row')
        else:
            # read existing table
            existing_df = spark.read.parquet(metrics_table_path)
            
            # Union and drop duplicates
            combined_df = existing_df.unionByName(metrics_df)
            combined_df = combined_df.dropDuplicates(["snapshot_date", "model_version"])
            
            # Overwrite table with deduplicated data
            combined_df.write.mode("overwrite").parquet(metrics_table_path)
            print("Updated and deduplicated (if applicable) gold metrics table")

        print('\n---completed job---\n\n')

    finally:
        print("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run model monitoring job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)