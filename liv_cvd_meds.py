'''
Python file that loads necessary dependencies and then extracts each of the cvd medications from NHS BSA primary care meds data.
It does this using liverpool lsoa data.
'''

#––––––Medications Data
spark.sql ('CLEAR CACHE')

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from pandas.api.types import CategoricalDtype
import pyspark.sql.functions as f
from pyspark.sql.functions import col, to_date, collect_list, collect_set, concat_ws, lit, substring, date_format, first
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType
from datetime import date
from pyspark.sql.functions import datediff, round, months_between, when, explode, split
from pyspark.sql import Row
from pyspark.sql.window import Window
import contextlib
import io
import os


#to save the data
# regional_cvd_ineq.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_regional_cvd_ineq_ic_ddmmyyyy")

new = spark.table('hive_metastore.dss_corporate.ons_lsoa_ccg_lad')\
    .filter(f.col("LAD13CD").isin(
        "E06000049","E06000050","E06000006","E08000011","E08000013","E08000012","E08000014","E08000015","E06000007"))\
    .dropDuplicates(['LSOA11CD', 'LSOA11NM', 'LAD13NM','LAD13CD'])\
    .select('LSOA11CD', 'LSOA11NM', 'LAD13NM','LAD13CD')

liv_lsoa = new.select(col('LSOA11CD').alias('lsoa'), col('LAD13NM').alias('lad'))
dem = spark.table('dsa_391419_j3w9t_collab.hds_curated_assets__demographics_2024_04_25')#.filter(f.col('in_gdppr') ==1)
dem = (
    dem
    .select(f.col('person_id').alias('pid'), 
            f.col('date_of_birth').alias('dob'), 
            'sex', 
            f.col('ethnicity_5_group').alias('ethnicity'), 
            f.col('imd_decile').alias('imd10'),
            f.col('imd_quintile').alias('imd5'), 
            f.col('date_of_death').alias('dod'), 
            'lsoa'
            )
    .join(liv_lsoa, on='lsoa', how='inner')
    .withColumn('age', f.datediff(f.to_date(f.lit('2018-04-01')), f.col('dob')) / 365.25) #IKE CHANGED END DATE FOR AGE CALC TO 2025 FROM 2018
    .withColumn('sex', f.when(f.col('sex').isNull(), 'I')
                        .otherwise(f.col('sex')))
    .withColumn('ethnicity', f.when(f.col('ethnicity').isNull() | (f.col('ethnicity')=='Other ethnic group'), 
                                    'Other/Unknown')
                              .otherwise(f.col('ethnicity')))
    .withColumn('area_code_prefix', f.substring(f.col("lsoa"), 0, 3))
)
dem = dem.filter(f.col('age')>=18)
dem = dem.filter(f.col('age')<=112)
dem = dem.filter((f.col('sex') != 'I') & f.col('sex').isNotNull())
dem = dem.filter((f.col('area_code_prefix')=='E01') & f.col('imd10').isNotNull())  
demo= dem.select('pid', 'dob', 'sex', 'ethnicity','lsoa', 'lad', 'imd10', 'imd5')
demo_lsoa = demo.select('pid', 'lsoa')
demo_dob = demo.select('pid', 'dob')
demo_pid = demo.select('pid')


#––––––
#Load RT table
cvd_path = "hive_metastore.dars_nic_391419_j3w9t_collab.ccu014_01_cvd_drugs_rt_210616"
cvd_df = spark.table(cvd_path).select(f.col("key_group").alias("category"), f.col("BNF_PRESENTATION_CODE").alias("drug_code"))

# my_email = "ikechukwu.chukwudi@liverpool.ac.uk"
def csv_to_sparkdf(path):
    p_df =pd.read_csv(path, keep_default_na = False)
    return spark.createDataFrame(p_df)

#–––––
#select liverpool meds

# Load primary care meds table and select relevant columns
liv_cvd_meds = (
    spark.table("hive_metastore.dars_nic_391419_j3w9t.primary_care_meds_dars_nic_391419_j3w9t")
    .select(
        f.col("Person_ID_DEID").alias("pid"),
        f.col("ProcessingPeriodDate").alias("process_date"),
        f.col("PaidBNFName").alias("med_name"),
        f.col("PaidBNFCode").alias("drug_code"),
        f.col("PaidQuantity").alias("med_qty")
    )
    .withColumn("year_process", f.substring("process_date", 1, 4))
    .withColumn("yr_mo_process", f.substring("process_date", 1, 7))
    .join(cvd_df, on="drug_code", how="inner")
    .join(demo, on="pid", how="inner")
)

# Define a window spec to get the first event per person-category
window_spec = Window.partitionBy("pid", "category").orderBy("process_date")

# Add row number and clean up columns
liv_cvd_meds = (
    liv_cvd_meds
    .withColumn("row_number", f.row_number().over(window_spec))
    .withColumnRenamed("process_date", "event_date")
    .drop("age_gp")
)

# Filter to only the first medication event
incident_meds = liv_cvd_meds.filter(f.col("row_number") == 1)

# Create 'event_mo' column and filter events from 2017 onward
incident_meds2 = (
    incident_meds
    .withColumn("event_mo", f.to_date("yr_mo_process", "yyyy-MM"))
    .filter(f.col("event_mo") >= "2017-01-01")
)


incident_meds2= spark.table("dsa_391419_j3w9t_collab.ccu071_01_incident_meds2_ic_29052025")
# Subset by medication categories
htn     = incident_meds2.filter(f.col("category") == "antihypertensives")
lipid   = incident_meds2.filter(f.col("category") == "lipid_lowering")
antiplt = incident_meds2.filter(f.col("category") == "antiplatelets_secondary")
t2dm    = incident_meds2.filter(f.col("category") == "type_2_diabetes")
doac    = incident_meds2.filter(f.col("category") == "anticoagulant_DOAC")
vka     = incident_meds2.filter(f.col("category") == "anticoagulant_warfarins")

def pivot_count(df):
    df = df.groupBy('lsoa').pivot('process_mo').count()
    return df


#––––Sample meds transform and write to save

# lipid_final, label_order = join_imd_df(age_standard(lad_pivot(lipid)))
# transformed_lipid =lipid_final
# transformed_lipid.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_transformed_lipid_df_ic_27052025")

# antiplt_final, label_order = join_imd_df(age_standard(lad_pivot(antiplt)))
# transformed_antiplt =antiplt_final
# transformed_antiplt.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_transformed_antiplt_df_ic_27052025")

# t2dm_final, label_order = join_imd_df(age_standard(lad_pivot(t2dm)))
# transformed_t2dm =t2dm_final
# transformed_t2dm.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_transformed_t2dm_df_ic_27052025")
