'''
This notebook needs functions_ic to run
It generates a demography data for london population using the ONS nomis code for london LAD and used to filter the ONS lsoa_lad mapping of the UK - a dataset in the NHS SDE
The HES data set is then filtered for only london data and and stroke and compared with the SSNAP dataset for london
The same is done for MI and MINAP is used to compare.
Deaths in London are plotted as all cause and the CVD deaths are plotted using the I chapter of ICD 10 
The resulting stroke, MI, all cause death and CV death are then age standardised and joined to IMD to create a transformed table of each outcome which then used to plot the graphs
'''
lon_lad = ["E09000001","E09000002","E09000003","E09000004","E09000005","E09000006","E09000007","E09000008","E09000009","E09000010","E09000011","E09000012","E09000013","E09000014","E09000015","E09000016","E09000017","E09000018","E09000019","E09000020","E09000021","E09000022","E09000023","E09000024","E09000025","E09000026","E09000027","E09000028","E09000029","E09000030","E09000031","E09000032","E09000033"]

new = spark.table('hive_metastore.dss_corporate.ons_lsoa_ccg_lad')\
    .filter(f.col("LAD13CD").isin(lon_lad))\
    .dropDuplicates(['LSOA11CD', 'LSOA11NM', 'LAD13NM','LAD13CD'])\
    .select('LSOA11CD', 'LSOA11NM', 'LAD13NM','LAD13CD')

lon_lsoa = new.select(col('LSOA11CD').alias('lsoa'), col('LAD13NM').alias('lad'))
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
    .join(lon_lsoa, on='lsoa', how='inner')
    .withColumn('age', f.datediff(f.to_date(f.lit('2018-04-01')), f.col('dob')) / 365.25) #IKE CHANGED END DATE FOR AGE CALC TO 2025 FROM 2018
    .withColumn('sex', f.when(f.col('sex').isNull(), 'I')
                        .otherwise(f.col('sex')))
    .withColumn('ethnicity', f.when(f.col('ethnicity').isNull() | (f.col('ethnicity')=='Other ethnic group'), 
                                    'Other/Unknown')
                              .otherwise(f.col('ethnicity')))
    .withColumn('area_code_prefix', f.substring(f.col("lsoa"), 0, 3))
)
dem =dem.withColumn(
    "age_gp",
    f.when((f.col("age") >= 18) & (f.col("age") <= 39), "18-39")
     .when((f.col("age") >= 40) & (f.col("age") <= 59), "40-59")
     .when((f.col("age") >= 60) & (f.col("age") <= 79), "60-79")
     .when(f.col("age") >= 80, "80+")
     .otherwise("Unknown")
)

dem = dem.filter(f.col('age')>=18)
dem = dem.filter(f.col('age')<=112)
dem = dem.filter((f.col('sex') != 'I') & f.col('sex').isNotNull())
dem = dem.filter((f.col('area_code_prefix')=='E01') & f.col('imd10').isNotNull())  
demo= dem.select('pid', 'dob', 'sex', 'ethnicity','lsoa', 'lad','imd5','imd10','age_gp')
demo_lsoa = demo.select('pid', 'lsoa')
demo_pid = demo.select('pid')


#––––––––––––

pop_24 = predict_population_2024(f"/Workspace/Users/ikechukwu.chukwudi@liverpool.ac.uk/lsoa_lad/em_popu_18_23.csv")

#–––––––––

# GO to population_predictor notebook FOR PREDICTION of pop_24
from pyspark.sql.functions import col, round, first, substring, concat_ws, lpad, lit
# Load population files
pop_18_23 = csv_to_sparkdf("/Workspace/Users/ikechukwu.chukwudi@liverpool.ac.uk/lsoa_lad/em_popu_18_23.csv")
pop_24 = spark.createDataFrame(pop_24)
# Drop unnecessary column
pop_18_23 = pop_18_23.drop("lad_code")
# Prepare predicted 2024 population
pop_24 = pop_24.select("year", "age_gp", "lad", col("predicted_population").alias("population"))
pop_24 = pop_24.withColumn("population", round(col("population"), 0))
# Combine both datasets
population = pop_18_23.union(pop_24)
# Pivot to wide format: one row per LAD + year, columns = age groups
pop_df = population.groupBy("lad", "year").pivot("age_gp").agg(first("population"))
# Extract year and generate months 1–12
pop_df = pop_df.withColumn("year_int", substring("year", 1, 4))
months_df = spark.createDataFrame([(i,) for i in range(1, 13)], ["month"])
pop_df = pop_df.crossJoin(months_df)
# Create a yyyy-mm-01 formatted date column
pop_df = pop_df.withColumn(
    "year_mo",
    concat_ws("-", col("year_int").cast("string"), lpad(col("month"), 2, "0"), lit("01"))
).drop("year_int", "month")

exp_pop = pop_df.withColumnRenamed("80", "80+")

#––––––


#HES Stroke counting
#Load saved hes latests archive 
hes_path = "dsa_391419_j3w9t_collab.ccu071_01_hes_latest_archive_ic_21032025"
hes_icd =spark.table(hes_path)
# hes_icd= hes_icd.filter("(EPISTART > '2017-03-31') AND (EPISTART < '2025-04-01')")
hes_icd= hes_icd.filter("(ADMIDATE > '2017-03-31') AND (ADMIDATE < '2024-04-01')")
hes_df = hes_icd.withColumn("diag3arr", split(col("DIAG_3_CONCAT"), ","))
hes_df = hes_df.withColumn("diag3exp", explode(col("diag3arr")))
hes_df = hes_df.withColumnRenamed("diag3exp", "icd_code")

icd_code = ['I61','I63','I64']

df = hes_df.filter(hes_df.icd_code.isin(icd_code))

df =df.select(col("PERSON_ID_DEID").alias("pid"), col("ADMIDATE").alias("event_date"), "icd_code") #Event date changed to EPISTART from ADMIDATE and then back to admidate 13 May - SDE guys used this
df= df.join(demo, on = 'pid', how='inner')
df = df.withColumn("event_mo", to_date(substring("event_date", 0,7)))

df = df.withColumn("row_num", row_number().over(Window.partitionBy('pid').orderBy('event_date')))
df = df.filter("row_num = 1").drop("row_num")
stroke = df.select('pid','dob','event_date','event_mo','lsoa','lad', 'imd5', 'imd10')

diff_stroke = df.groupBy('event_mo', 'lad').count()

# df_grouped_pandas = diff_stroke.toPandas()

count_df = df.groupBy("event_mo").count()
count_df = count_df.withColumnRenamed("count", "hes_count")


#–––––––––
#example plotting
trans_strokeem, label_order = join_imd_df(age_standard(lad_pivot(stroke)))
plot_covid_by_lad(trans_strokeem, 'Age-standardised plots of Stroke in The East Midland Region with Regression Line')

#–––
SSNAP
#SSNAP Stroke Count
from pyspark.sql.functions import col, year, month, concat_ws


snap_path = "dars_nic_391419_j3w9t_collab.ssnap_dars_nic_391419_j3w9t_archive"
df_snap = spark.table(snap_path).filter("archived_on = '2024-12-02'")
df_snap = df_snap.select(
    f.col('S1FIRSTARRIVALDATETIME').alias('event_datetime'), f.col('Person_ID_DEID').alias('pid')).withColumn("event_date", f.col("event_datetime").substr(1,10))
df_snap = df_snap.filter("(pid IS NOT NULL)")
# df_snap =df_snap.filter("(event_date > '2017-03-31') AND (event_date < '2025-04-01')")
df_snap =df_snap.filter("(event_date > '2018-04-31') AND (event_date < '2024-11-01')")
df_snap = df_snap.select('event_date', 'pid').distinct()
# df_snap = df_snap.withColumn("event_mo", to_date(substring("event_date",0,7)))
df_snap = df_snap.join(demo, on ='pid', how='inner')

df_snap = df_snap.withColumn("event_mo", to_date(substring("event_date",0,7)))
count_snap = df_snap.groupBy("event_mo").count()
count_snap = count_snap.withColumnRenamed("count", "snap_count")
#Join monthly HES stroke count with SSNAP Stroke Count and calculate difference
snap_hes = count_df.join(count_snap, on = "event_mo", how = "inner")
snap_hes = snap_hes.withColumn("diff", col('hes_count')-col('snap_count'))

#–––
#snap plot vs hes vs diff
df = snap_hes.toPandas()
#Look at national Plot in slide 20- pre-2020 SSNAP is here and there is a sharp SSNAP spike mid 2021.
df =df.sort_values('event_mo')
plt.figure(figsize=(10, 6))
plt.plot(df['event_mo'], df['hes_count'], label='HES Count')
plt.plot(df['event_mo'], df['snap_count'], label='SNAP Count')
plt.plot(df['event_mo'], df['diff'], label = "Difference")

plt.xlabel('Event Month')
plt.ylabel('Count')
plt.title('HES and SSNAP Stroke Counts Over Time')
# plt.title('SSNAP Counts Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

#––––– MI

hes_path = "dsa_391419_j3w9t_collab.ccu071_01_hes_latest_archive_ic_21032025"
hes_icd =spark.table(hes_path)
hes_icd= hes_icd.filter("(ADMIDATE > '2017-03-31') AND (ADMIDATE < '2025-04-01')")
hes_df = hes_icd.withColumn("diag3arr", split(col("DIAG_3_CONCAT"), ","))
hes_df = hes_df.withColumn("diag3exp", explode(col("diag3arr")))
hes_df = hes_df.withColumnRenamed("diag3exp", "icd_code")
icd_code4 =['I21','I22']
df = hes_df.filter(hes_df.icd_code.isin(icd_code4))                  
df =df.select(col("PERSON_ID_DEID").alias("pid"), col("LSOA11").alias("lsoa"), col("ADMIDATE").alias("event_date"), "icd_code")
df = df.withColumn("event_mo", to_date(substring("event_date", 0,7)))
df = df.join(demo, on='pid', how ='inner')
df = df.withColumn("row_num", row_number().over(Window.partitionBy('pid').orderBy('event_date')))
df = df.filter("row_num = 1").drop("row_num")

mi_df = df

#–––––MINAP
# MINAP
# Set the names of the databases to a parameter for ease
db = 'dars_nic_391419_j3w9t' # provisioned - read-only 
dbc = f'dsa_391419_j3w9t_collab' # read-and-write
dbc_old = f'{db}_collab' # read-only 
dss = 'dss_corporate' # read-only

minap_archive = spark.table(f'{dbc_old}.nicor_minap_dars_nic_391419_j3w9t_archive')
minap = (
  minap_archive
  .filter(f.col('archived_on') == '2025-04-24')
  .withColumnRenamed('1_03_NHS_NUMBER_DEID', 'pid')
  .withColumn('ADMIDATE', f.date_format('ARRIVAL_AT_HOSPITAL', 'yyyy-MM-dd'))
  .withColumn("ADMIDATE", f.to_date("ADMIDATE"))
  .select('pid', 'ADMIDATE')
)
minap_df = (
  minap
  .filter(f.col('pid').isNotNull())
  .filter("(ADMIDATE > '2017-03-31') AND (ADMIDATE < '2025-04-01')") # filter for years of interest
  .distinct()
)
minap_df = minap_df.withColumn("event_mo", to_date(substring("ADMIDATE",0,7)))
minap_df = minap_df.join(demo, 'pid', how='inner')
# minap_df = minap_df.withColumn('event_mo', f.col("ADMIDATE").substr(0,7))
count_minap  = minap_df.groupBy('event_mo').count()
count_minap = count_minap.withColumnRenamed("count", "minap_count")

minap_hes = count_mi_df.join(count_minap, on = "event_mo", how = "inner")
# display(count_snap)
minap_hes = minap_hes.withColumn("diff", col('hes_count')-col('minap_count'))

# display(minap_hes)

#––––minap hes plot
df = minap_hes.toPandas()
df['event_mo'] = pd.to_datetime(df['event_mo'], format='%Y-%m')

df =df.sort_values('event_mo')
plt.figure(figsize=(10, 6))
plt.plot(df['event_mo'], df['hes_count'], label='HES Count')
plt.plot(df['event_mo'], df['minap_count'], label='MINAP Count')
plt.plot(df['event_mo'], df['diff'], label = "Difference")

plt.xlabel('Event Month')
plt.ylabel('Count')
plt.title('HES and MINAP Counts Over Time')
# plt.title('MINAP Counts Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

#–––––––Death - All cause
death_path = "hive_metastore.dars_nic_391419_j3w9t.deaths_dars_nic_391419_j3w9t"
death=spark.table(death_path)
death_post_2019 = spark.table(death_path).select(
    f.col("REG_DATE_OF_DEATH").alias("reg_dod"),
    f.col("S_UNDERLYING_COD_ICD10").alias("u_cause_death"),
    f.col("S_COD_CODE_1").alias("cod_1"),
    f.col("S_COD_CODE_2").alias("cod_2"),
    f.col("S_COD_CODE_3").alias("cod_3"),
    f.col("S_COD_CODE_4").alias("cod_4"),
    f.col("DEC_CONF_NHS_NUMBER_CLEAN_DEID").alias("pid")
).filter(col("reg_dod")>="20180101")

liv_deaths = demo.join(death_post_2019, how = "inner", on = "pid")

death_liv2 = liv_deaths.select(
    f.col("pid"), f.col("reg_dod"), f.col("u_cause_death"), f.col("cod_1"), 'lsoa', 'lad', 'dob', 'sex',
    ).withColumn(
        "event_year", substring(col("reg_dod"), 1, 4)
        )
death_liv2 = death_liv2.withColumn("death_mo",
    concat_ws("-", substring(col("reg_dod"), 1, 4), substring(col("reg_dod"), 5, 2))
)                                
death_liv22 =death_liv2.withColumn('death_mo', to_date(col('death_mo'), 'yyyy-MM'))
death = death_liv22.withColumnRenamed('death_mo', 'event_mo')
death = death.withColumn('event_date', to_date(col('reg_dod'), 'yyyyMMdd'))

# death_liv_yr = death_liv2.groupBy("event_year").count()
# death_liv_mo = death_liv2.groupBy("death_mo").count()
# ls_deaths = death_liv2.groupBy("lsoa").count()

# df = df.withColumn('reg_dod', to_date(col('reg_dod'), "yyyyMMdd"))
# df = df.withColumn('age', f.datediff(f.col('reg_dod'), f.col('dob'))/365.25)
# df= df.withColumn('age_gp', f.when(f.col('age')<65, '18-65').when(f.col('age')>65, '65+'))
# # display(df)
# diff_death = df.groupBy('death_mo', 'sex', 'age_gp', 'lad').count()
# display(ls_deaths)

#––––– sample transform
death_piv_in = lad_pivot(death)
death_stan = age_standard(death_piv_in)
death_pdf, label_order = join_imd_df(death_stan)

#––– sample save outcome
transformed_death = death_pdf
transformed_death.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_transformed_death_pdf2_ic_27052025")


#–––––plot all cuase death
death_liv_mo = death_liv2.groupBy("death_mo").count()
death_liv_mo2_pd = death_liv_mo.toPandas()
# Convert 'year_month' to datetime for better formatting on the plot
death_liv_mo2_pd['death_mo'] = pd.to_datetime(death_liv_mo2_pd['death_mo'], format='%Y-%m')
# Plot the line chart
plt.figure(figsize=(12, 6))
sns.lineplot(x='death_mo', y='count', data=death_liv_mo2_pd, color='r')
# Rotate x-axis labels for better readability
plt.xticks(rotation=90)
# Add title and labels
plt.title('Trend of Monthly All Cause Liverpool Deaths', fontsize=16)
plt.xlabel('Months', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()


#––––– cvd death


# CVD Deaths I00 to I99 in ICD - chapeter I
cvd_icd_path = "hive_metastore.dss_corporate.icd10_codes"
cvd_icd1 = spark.table(cvd_icd_path).select(
    f.col("CODE").alias("icd_dot_code"), f.col("DESCRIPTION").alias("icd_description"), f.col("ALT_CODE")
    ).filter(
        f.col("CODE").startswith("I"))
cvd_icd10 =cvd_icd1.select(f.col("ALT_CODE").alias("cod"), f.col("icd_description"))
cvd_icd10= cvd_icd10.dropDuplicates(["cod"]) #as there are duplicate icd10 codes. this inflated my previous table number. cvd1cd10 is 1384 and cvdicd10 distinct is 475.

death_liv2 = death_liv2.withColumnRenamed("u_cause_death", "cod")
only_cvd_death = cvd_icd10.join(death_liv2, on="cod", how="inner")
only_cvd_death1= only_cvd_death.withColumn('death_mo', to_date(col('death_mo'), 'yyyy-MM'))
# display(only_cvd_death.count()) = 105K

cv_death = only_cvd_death1.withColumnRenamed('death_mo', 'event_mo')
cv_death = cv_death.withColumn('event_date', to_date(col('reg_dod'), 'yyyyMMdd'))

#––––sample transform
cvd_piv_in = lad_pivot(cv_death)
cvd_stan = age_standard(cvd_piv_in)
cvd_pdf, label_order = join_imd_df(cvd_stan)

transformed_cvdeath = cvd_pdf
transformed_cvdeath.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_transformed_cv_death_ic_27052025")

#–––––––

# CVD Death
only_cvd_death.groupBy('event_year').count().display()
cvd_death_mo =only_cvd_death.groupBy("death_mo").count()
cvd_death_pd = cvd_death_mo.toPandas()
# Convert 'year_month' to datetime for better formatting on the plot
cvd_death_pd['death_mo'] = pd.to_datetime(cvd_death_pd['death_mo'], format='%Y-%m')
# Plot the line chart
plt.figure(figsize=(12, 6))
sns.lineplot(x='death_mo', y='count', data=cvd_death_pd, color='g')
# Rotate x-axis labels for better readability
plt.xticks(rotation=90)
# Add title and labels
plt.title('Trend of Monthly CVD Deaths in Cheshire and Merseyside', fontsize=16)
plt.xlabel('Months', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()

#–––––

