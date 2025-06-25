import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# to call this script: python data_processing_main.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('---starting data processing job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("data_processing_main") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments into variable date_str (to use for calling subsequent util data processing functions)
    date_str = snapshotdate

    try:
        # ---------------------------------------
        # Build Bronze Tables for given date_str
        # ---------------------------------------

        # Loan Management System Data
        bronze_lms_directory = "datamart/bronze/lms/"

        if not os.path.exists(bronze_lms_directory):
            os.makedirs(bronze_lms_directory)
        
        utils.data_processing_bronze_table.process_bronze_loan_table(date_str, bronze_lms_directory, spark)

        # Clickstream Data
        bronze_clks_directory = "datamart/bronze/clks/"

        if not os.path.exists(bronze_clks_directory):
            os.makedirs(bronze_clks_directory)

        utils.data_processing_bronze_table.process_bronze_clickstream_table(date_str, bronze_clks_directory, spark)

        # Attributes Data
        bronze_attr_directory = "datamart/bronze/attr/"

        if not os.path.exists(bronze_attr_directory):
            os.makedirs(bronze_attr_directory)

        utils.data_processing_bronze_table.process_bronze_attributes_table(date_str, bronze_attr_directory, spark)

        # Financials Data
        bronze_fin_directory = "datamart/bronze/fin/"

        if not os.path.exists(bronze_fin_directory):
            os.makedirs(bronze_fin_directory)

        utils.data_processing_bronze_table.process_bronze_financials_table(date_str, bronze_fin_directory, spark)


        # ---------------------------------------  
        # Build Silver Tables for given date_str
        # --------------------------------------- 

        # Loan Management System Data
        silver_lms_directory = "datamart/silver/lms/"

        if not os.path.exists(silver_lms_directory):
            os.makedirs(silver_lms_directory)

        utils.data_processing_silver_table.process_silver_loan_table(date_str, bronze_lms_directory, silver_lms_directory, spark)

        # Clickstream Data
        silver_clks_directory = "datamart/silver/clks/"

        if not os.path.exists(silver_clks_directory):
            os.makedirs(silver_clks_directory)

        utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_clks_directory, silver_clks_directory, spark)

        # Attributes Data
        silver_attr_directory = "datamart/silver/attr/"

        if not os.path.exists(silver_attr_directory):
            os.makedirs(silver_attr_directory)

        utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_attr_directory, silver_attr_directory, spark)

        # Financials Data
        silver_fin_directory = "datamart/silver/fin/"

        if not os.path.exists(silver_fin_directory):
            os.makedirs(silver_fin_directory)

        utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_fin_directory, silver_fin_directory, spark)


        # -------------------------------------- 
        # Build Gold Tables for given date_str
        # --------------------------------------

        # Build Feature Store
        # engagement_tab
        gold_clks_directory = "datamart/gold/feature_store/eng/"

        if not os.path.exists(gold_clks_directory):
            os.makedirs(gold_clks_directory)

        utils.data_processing_gold_table.process_fts_gold_engag_table(date_str, silver_clks_directory, gold_clks_directory, spark)

        # cust_fin_risk_tab
        gold_fin_directory = "datamart/gold/feature_store/cust_fin_risk/"

        if not os.path.exists(gold_fin_directory):
            os.makedirs(gold_fin_directory)

        utils.data_processing_gold_table.process_fts_gold_cust_risk_table(date_str, silver_fin_directory, gold_fin_directory, spark)


        # Build Label Store (based on Loan Mgmt System Data)
        gold_label_store_directory = "datamart/gold/label_store/"

        if not os.path.exists(gold_label_store_directory):
            os.makedirs(gold_label_store_directory)

        utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_lms_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

        print('---completed data processing job---\n\n') # print this when entire job succeeded
    
    finally:
        # ---------------------- 
        # Stop Spark Session
        # ---------------------- 
        print('Stopping Spark session...')
        spark.stop()
        print('Spark session stopped')
        

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run data processing job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)

    