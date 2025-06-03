
# lad_pivot(input_df)
# age_standard(df)
# join_imd_df(df)
# use case of the above 3; trans_df, label_order = join_imd_df(age_standard(lad_pivot(df)))
# prepare_plot()
# plot_facetgrid(trans_df, y_label, legend) eg plot_facetgrid(trans_htn, "Age-standardised Anti-HTN Prescription Rates per 100K", "Age Standardised Anti-HTN Rates")
# plot_covid_by_lad(trans_df, suptitle)
# plot_df_boxplot(trans_df, plot_title, ylabel)
# plot_htn_heatmap(trans_df, title)
# plot_forest(reg_coeff(trans_df), title)

def csv_to_sparkdf(path):
    '''function that takes a file path from workspace and returns it as a spark dataframe'''
    p_df =pd.read_csv(path, keep_default_na = False)
    return spark.createDataFrame(p_df)

def predict_population_2024(path2):
    '''
    function that predictes the 2024 midyear population for each combination of lad and age-group.
    It takes a csv file that contains mid-year population between 2018 and 2023 for each lad and age_group (downloaded from ons and table transformed locally to this format)
    '''
    from pyspark.sql.functions import to_date, year
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    #first, load population dataframe forn 2018 to 2023 gotten from ONS
    population = csv_to_sparkdf(path2)
    # 2: Convert date to year
    df = population.withColumn("date", to_date("year", "yyyy-MM-dd"))
    df = df.withColumn("year_num", year("date"))
    
    #3: Convert to pandas DataFrame
    pdf = df.select("year_num", "age_gp", "lad", "population").toPandas()
    
    # 4: Fit regression model and predict
    results = []

    for (age_gp, lad), group in pdf.groupby(["age_gp", "lad"]):
        group = group.sort_values("year_num")
        x = group["year_num"].values.reshape(-1, 1)
        y = group["population"].values
        
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict([[2024]])

        results.append({
            "age_gp": age_gp,
            "lad": lad,
            "year": "2024-06-01",
            "predicted_population": y_pred[0]
        })

    # 5: Return predictions as DataFrame
    predicted_df = pd.DataFrame(results)
    return predicted_df


#–––––––
def roll_ave(df):
    ''' calculate rolling average for each outcome and appends it as a new column'''
    window_spec = Window \
        .partitionBy("lad")\
        .orderBy("year_mo") \
        .rowsBetween(-2, 0)  # previous 2 rows and current row
    df= df.withColumn("rolling_avg", f.avg(col("age_standardised_rate")).over(window_spec))
    return df.filter(f.col('year_mo')>='2018-04-01')

#–––––––––––––


#function to pivot input_df so each cell contains the number of stroke per month per lad
def lad_pivot(input_df):
    '''
    this function takes a raw outcomes/med data table gotten from filtered outcome eg stroke of interest and filtered for the demography of interest and adds age and age_gp cols.
    The age is calculated as the age event occured (event_date).
    The table is then grouped by lad, event_mo and age_gp and aggregated by individual patient counts.
    This is then pivoted so that we have counts of each age group for each lad and event month column.
    '''
    df = input_df
    df =df.filter(f.col("event_mo")>= "2018-04-01")
    df=df.filter(f.col("event_mo")<="2024-10-01")
    df = df.withColumn('age', f.datediff(f.col('event_date'), f.col('dob'))/365.25)
    df= df.withColumn('age_gp', f.when(f.col('age') < 40, '18-39')
                         .when(f.col('age') < 60, '40-59')
                         .when(f.col('age') < 80, '60-79')
                         .otherwise('80+'))
    # df = df.withColumn("age_sex", concat_ws("_", "age_gp", "sex"))
    # # 2. Group by lad, month, and age_sex and count unique pids
    grouped = df.groupBy("lad", "event_mo", "age_gp").agg(f.count("pid").alias("pid_count"))
    # # 3. Pivot table: rows are (lad, event_mo), columns are age_sex categories
    pivot_df = grouped.groupBy("lad", "event_mo") \
        .pivot("age_gp") \
        .agg(first("pid_count"))
    pivot_df = pivot_df.fillna(0)
    return pivot_df


# See Liverpool Pop ration - very high 18-39 accounting for 45%, compared to 25% for ESP.
# ESP pop ratio:{'18-39': 0.255,
#     	'40-59': 0.275,
#     	'60-79': 0.205,
#     	'80+': 0.050}
     
def age_standard(input_df):
    """ this function sets up the different weights for the age bands with  liverpool lad as standard pop and then divides
    the count of outcome per age bracket by population of that age bracket and multiplies by 100,000 to get the age standardised rate per 100,000.
    It then multiples each of these rates by the standard weighting for the age gp and then the weighted rates are summed together to get the age standardised rate
    The weights are derived from liverpool LAD 2021 census population weighting
    """ 
    age_brackets = ['18-39', '40-59', '60-79', '80+']
    weights = {
        '18-39': 0.44789403905971137,
        '40-59': 0.2903854979348325,
        '60-79': 0.21385191984090562,
        '80+': 0.047868543164550505}
    df_cond =input_df
    #exp_pop is table of our denominator population by lad and month- it was derived from ONS midyear population and exploded to give values from Jan to Dec
    df_pop = exp_pop
    df_cond = df_cond.withColumnRenamed('event_mo', 'year_mo')
    df_joined = df_cond.alias("cond").join(
        df_pop.alias("pop"),
        on=["lad", "year_mo"],
        how="inner"
        )
    for age in age_brackets:
        cond_col = f"cond.`{age}`"
        pop_col = f"pop.`{age}`"
        rate_col = f"{age}_rate"
        weighted_col = f"{age}_weighted"
        
        df_joined = df_joined.withColumn(
            rate_col, (col(f"cond.`{age}`") / col(f"pop.`{age}`")) * 100000
            ).withColumn(
                weighted_col, col(rate_col) * weights[age]
                )
    # from pyspark.sql.functions import sum as spark_sum
    weighted_cols = [f"{age}_weighted" for age in age_brackets]
    df_result = df_joined.withColumn(
        "age_standardised_rate",
        sum([col(c) for c in weighted_cols])
        ).select("lad", "year_mo", "age_standardised_rate")
    return df_result


def join_imd_df(input_df):
    '''function that joins outcome pd with imd and returns joined outcome pd with imd and a label_order'''
    imd = demo.groupBy('lad') \
        .agg(f.round(f.avg('imd10'), 2).alias('mean_imd'), f.round(f.stddev_pop('imd10'),2).alias('sd_imd')) \
        .orderBy('mean_imd')
    out_df= input_df.join(imd, on ='lad', how = 'left')
    #Create a new column for labeling: "LAD (Mean IMD Decile: x.xx)"
    out_df = out_df.withColumn("lad_label",concat_ws("", col("lad"),lit("(Mean IMD Decile:"),col("mean_imd"),lit(")")))
    #Create an ordered list of facet labels sorted by mean IMD (ascending)
    label_order = (out_df.select("lad_label", "mean_imd").dropDuplicates().orderBy("mean_imd").select("lad_label").rdd.flatMap(lambda x: x).collect())
    return out_df, label_order

#–––––––––––––––––

def prepare_plot(trans_df):
    ''' takes trans_df data as pyspark df.
    it also calls rolling average function first to return the rolling average column which we use to plot
    returns df and label_order. df is categorised by lad_label and covid period
    '''
    label_order = (
        trans_df.select("lad_label", "mean_imd")
        .dropDuplicates()
        .orderBy("mean_imd")
        .select("lad_label")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    trans_df = roll_ave(trans_df)
    df = trans_df.toPandas()
    df['date'] = pd.to_datetime(df['year_mo'])

    df['covid_period'] = df['date'].apply(
        lambda d: "Pre-COVID" if d < pd.Timestamp("2020-03-01")
        else "During COVID" if d <= pd.Timestamp("2022-05-01")
        else "Post-COVID"
    )
    df['lad_label'] = pd.Categorical(df['lad_label'], categories=label_order, ordered=True)
    df['covid_period'] = pd.Categorical(df['covid_period'], categories=['Pre-COVID', 'During COVID', 'Post-COVID'], ordered=True)

    return df, label_order

#–––––––––––––––––––––––––––––––

def plot_facetgrid(trans_df, suptitle):
    '''
    Plots a facet line plot of each lad's rolling average of a variables (eg stroke) age standardised rates, arranged by IMD decile and dotted line to show covid transitions
    Not prepare plot is an earlier function that prepared the table for plotting, by transforming to pandas and arranging table by mean IMD decile.
    '''
    # Prepare data and label order
    df, label_order = prepare_plot(trans_df)

    ncols = 3
    nrows = int(np.ceil(len(label_order) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, lad in enumerate(label_order):
        ax = axes[i]
        lad_df = df[df['lad_label'] == lad]

        # Plot rolling average
        ax.plot(lad_df['date'], lad_df['rolling_avg'], label='Rolling Avg')

        # COVID transition lines
        for line_date in ["2020-03-01", "2022-05-01"]:
            ax.axvline(pd.Timestamp(line_date), color='grey', linestyle=':', linewidth=0.8)

        ax.set_title(lad)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        ax.tick_params(axis='x', rotation=90, labelsize=8)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add shared axis labels and title
    fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12)
    fig.text(0.04, 0.5, '3-Monthly Rolling Average Rate per 100,000 People (Age-Standardised)', va='center', rotation='vertical', fontsize=12)
    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0.06, 0.06, 1, 0.95])
    plt.show()

#–––––––––––––––

def plot_covid_by_lad(trans_df, suptitle):
    '''
    This plots a facet scatterplot of the rolling average age-standardised rates of a variable of interest eg stroke.
    It divides each plot into pre durig and post covid periods. It then uses month index 0-end to draw a regression line across each covid period.
    Note, it uses linregress
    '''
    from scipy.stats import linregress

    df, label_order = prepare_plot(trans_df)

    # Reindex month within each lad and covid period group
    df['month_index'] = df.groupby(['lad_label', 'covid_period'])['date'].transform(lambda x: (x.dt.to_period("M") - x.min().to_period("M")).apply(lambda p: p.n))

    # Define color palette
    # palette = {'Pre-COVID': 'skyblue', 'During COVID': 'salmon', 'Post-COVID': 'limegreen'}
    # Define color palette
    palette = {'Pre-COVID': '#0072B2', 'During COVID': '#E69F00', 'Post-COVID': '#009E73'}

    # Setup subplots
    n = len(label_order)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    # Loop through LADs
    for i, lad in enumerate(label_order):
        ax = axes[i]
        df_lad = df[df['lad_label'] == lad].copy()
        df_lad = df_lad.sort_values('date').reset_index(drop=True)

        # Scatterplot
        sns.scatterplot(data=df_lad, x='date', y='rolling_avg',
                        hue='covid_period', palette=palette, legend=False, ax=ax)

        # Plot regressions by period
        for period, sub in df_lad.groupby('covid_period'):
            if len(sub) >= 2:
                # x = sub['date'].map(pd.Timestamp.toordinal)
                x = sub['month_index']
                y = sub['rolling_avg']
                slope, intercept, r, p, se = linregress(x, y)

                # Generate monthly-aligned dates for regression line
                # x_dates = pd.date_range(start=sub['date'].min(), end=sub['date'].max(), freq='MS')
                x_month_index = pd.Series(range(sub['month_index'].min(), sub['month_index'].max()+1))
                # x_ordinal = x_dates.map(pd.Timestamp.toordinal)
                y_pred = intercept + slope * x_month_index

                # Confidence bands
                y_ci_lower = y_pred - 1.96 * se
                y_ci_upper = y_pred + 1.96 * se

                #Convert month_index back to datetime for plotting
                ref_date = sub.sort_values('month_index')['date'].iloc[0].to_period("M").to_timestamp()
                # create matching dates for x-axis
                x_dates = pd.date_range(start=ref_date, periods=len(x_month_index), freq='MS')


                ax.plot(x_dates, y_pred, '--', color=palette[period])
                ax.fill_between(x_dates, y_ci_lower, y_ci_upper, color=palette[period], alpha=0.2)

                # Add slope and p-value annotation
                ax.text(0.01, 0.95 - 0.08 * list(palette).index(period),
                        f"{period}: β={slope:.4f}, p={p:.4f}",
                        transform=ax.transAxes,
                        fontsize=8, color=palette[period], verticalalignment='top')

        # COVID transition lines
        for date in ["2020-03-01", "2022-05-01"]:
            ax.axvline(pd.Timestamp(date), color='grey', linestyle=':', linewidth=0.5)

        # Format x-axis
        ax.set_title(lad)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Set ticks every 6 months from the start date
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))  # Optional: nice date labels

        ax.tick_params(axis='x', rotation=90, labelsize=8)
    # Adjust layout
    plt.suptitle(suptitle, fontsize=16)
    fig.text(-0.04, 0.5, '3-Monthly Rolling Average Rate per 100,000 People (Age-Standardised)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.5, -0.05, 'Date', ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

#–––––––––––––––––––––

def plot_df_boxplot(trans_df, plot_title):
    '''
    Plot a boxplot of the median and IQR of a variable of interest. Note, age-standardised rates is used for this and not rolling average
    '''
    # Prepare the plot data
    df, label_order = prepare_plot(trans_df)

    # Create the figure
    plt.figure(figsize=(16, 10))

    # Create box plot
    sns.boxplot(
        x='lad_label',
        y = 'age_standardised_rate',
        # y='rolling_avg',
        hue='covid_period',
        data=df,
        palette={'Pre-COVID': 'skyblue', 'During COVID': 'coral', 'Post-COVID': 'lightgreen'}
    )

    # Adjust labels and title
    plt.title(plot_title, fontsize=14)
    plt.xlabel('Local Authority Districts (Mean IMD)', fontsize=10)
    plt.ylabel('Age Standardised Rates per 100,000', fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(title='COVID Period')

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Final layout
    plt.tight_layout()
    plt.show()

#––––––––––––––––––
def plot_htn_heatmap(trans_df, title):
    '''
    Plots a heatmap using a table of interest called trans_df. It transforms te date to prepare plot, so it sorts by lad mean IMD decile
    '''
    df, label_order = prepare_plot(trans_df)
    heatmap_data = df.groupby(['lad_label', 'covid_period'])[ 'age_standardised_rate'].mean().unstack()

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(heatmap_data, cmap='RdYlGn_r', annot=True, fmt='.2f', linewidths=0.5)

    # Adjust labels and title
    plt.title(title, fontsize=16)
    plt.ylabel('Region', fontsize=14)

    # Rotate the tick labels
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

#––––––––––––

def reg_coeff(trans_df):
    '''
    This function calcuates the regression coefficients used to plot the regression plots as it will then be used to plot a forest plot of the slope with SE
    '''
    from scipy.stats import linregress
    df, label_order = prepare_plot(trans_df)
    results = []
    # Reindex month within each lad and covid period group
    df['month_index'] = df.groupby(['lad_label', 'covid_period'])['date'].transform(
        lambda x: (x.dt.to_period("M") - x.min().to_period("M")).apply(lambda p: p.n)
    )
    for (lad, period), group in df.groupby(['lad_label', 'covid_period']):
        if len(group) > 2:
            # group = group.sort_values('date')
            # x = group['date'].map(pd.Timestamp.toordinal)
            x = group['month_index']
            y = group['rolling_avg']
            slope, intercept, r, p, se = linregress(x, y)

            results.append({
                'lad_label': lad,
                'covid_period': period,
                'slope': slope,
                'se': se,
                'p_value': p,
                'n_obs': len(group)
            })
    return pd.DataFrame(results)

#––––––––––––
def plot_forest(df, title):
    '''
    This plots a forest plot of the slopes
    '''
    periods = ['Pre-COVID', 'During COVID', 'Post-COVID']
    fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(12, len(df['lad_label'].unique()) * 0.5))

    # Fix LAD order (consistent y-axis)
    lad_order = df['lad_label'].unique()[::-1]
    y_pos = range(len(lad_order))
    # Compute bounds of the error bars
    df['upper'] = df['slope'] + df['se']
    df['lower'] = df['slope'] - df['se']

    # Find the farthest extent from 0 in either direction
    max_bound = max(abs(df['upper'].max()), abs(df['lower'].min())) + 0.02

    # Symmetric limits
    min_slope = -max_bound
    max_slope = max_bound

    for i, period in enumerate(periods):
        ax = axes[i]
        df_p = df[df['covid_period'] == period].copy()
        df_p = df_p.set_index('lad_label').reindex(lad_order).reset_index()

        ax.errorbar(df_p['slope'], y_pos, xerr=df_p['se'], fmt='o', color='tab:blue')
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_xlim(min_slope, max_slope)
        ax.set_title(period)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_p['lad_label'])
       

    fig.suptitle(title, fontsize=14, x=0.5, ha='center')
    fig.text(0.5, 0.04, 'Slope', ha='center', fontsize=12)
    plt.show()
#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
'''
Functions end
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



#–––––––––––––––––––––

# trans_stroke = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_stroke_pdf_ic_27052025").select('lad', 'year_mo','mean_imd', 'lad_label', f.col('age_standardised_rate').alias('stroke_rate'))
# trans_mi = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_mi_pdf_ic_27052025").select('lad', 'year_mo','mean_imd', 'lad_label', f.col('age_standardised_rate').alias('mi_rate'))
# trans_cvd = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_cv_death_ic_27052025").select('lad', 'year_mo','mean_imd', 'lad_label', f.col('age_standardised_rate').alias('cvdeath_rate'))
# trans_death = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_death_pdf2_ic_27052025").select('lad', 'year_mo','mean_imd', 'lad_label', f.col('age_standardised_rate').alias('death_rate'))

# df_outcome1 = trans_stroke \
#     .join(trans_mi, on=['lad', 'year_mo', 'mean_imd', 'lad_label'], how='inner') \
#     .join(trans_cvd, on=['lad', 'year_mo', 'mean_imd', 'lad_label'], how='inner') \
#     .join(trans_death, on=['lad', 'year_mo', 'mean_imd', 'lad_label'], how='inner')




# trans_htn = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_htn_df_ic_27052025") \
#     .select("lad", "year_mo", "mean_imd", "lad_label", f.col("age_standardised_rate").alias("htn_rate"))
# trans_lipid = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_lipid_df_ic_27052025") \
#     .select("lad", "year_mo", "mean_imd", "lad_label", f.col("age_standardised_rate").alias("lipid_rate"))
# trans_plt = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_antiplt_df_ic_27052025") \
#     .select("lad", "year_mo", "mean_imd", "lad_label", f.col("age_standardised_rate").alias("plt_rate"))
# trans_t2dm = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_t2dm_df_ic_27052025") \
#     .select("lad", "year_mo", "mean_imd", "lad_label", f.col("age_standardised_rate").alias("t2dm_rate"))
# trans_doac = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_doac_df_ic_27052025") \
#     .select("lad", "year_mo", "mean_imd", "lad_label", f.col("age_standardised_rate").alias("doac_rate"))
# trans_vka = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_vka_df_ic_27052025") \
#     .select("lad", "year_mo", "mean_imd", "lad_label", f.col("age_standardised_rate").alias("vka_rate"))


# # Join all medication datasets into df_outcome
# df_outcome2 = trans_htn \
#     .join(trans_lipid, on=["lad", "year_mo", "mean_imd", "lad_label"], how="inner") \
#     .join(trans_plt, on=["lad", "year_mo", "mean_imd", "lad_label"], how="inner") \
#     .join(trans_t2dm, on=["lad", "year_mo", "mean_imd", "lad_label"], how="inner") \
#     .join(trans_doac, on=["lad", "year_mo", "mean_imd", "lad_label"], how="inner")

# comb_df = df_outcome1.join(df_outcome2, on =["lad", "year_mo", "mean_imd", "lad_label"], how = 'left')


#–––––
trans_htn = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_htn_df_ic_27052025").filter(f.col('year_mo')>='2019-01-01')
trans_lipid = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_lipid_df_ic_27052025").filter(f.col('year_mo')>='2019-01-01')
trans_plt = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_antiplt_df_ic_27052025").filter(f.col('year_mo')>='2019-01-01')
trans_t2dm = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_t2dm_df_ic_27052025").filter(f.col('year_mo')>='2019-01-01')
trans_doac = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_doac_df_ic_27052025").filter(f.col('year_mo')>='2019-01-01')
trans_vka = spark.table("dsa_391419_j3w9t_collab.ccu071_01_transformed_vka_df_ic_27052025").filter(f.col('year_mo')>='2019-01-01')

#––––––––
'''
function for saving all plots
'''

my_email = 'ikechukwu.chukwudi@liverpool.ac.uk'
def save_all_plots(trans_df, base_name, my_email):
    # Define all plot calls with their respective titles
    plots = [
        ("facetgrid", f"Age-standardised {base_name} per 100,000"),
        ("covid_by_lad", f"Age Standardised {base_name} Per 100,000 by LAD and COVID period with Regression Line"),
        ("boxplot", f"Age-Standardised {base_name}"),
        ("htn_heatmap", f"Heat Map Showing Average Age-Standardised {base_name} by LAD"),
        ("forest", f"Forest Plot of Monthly {base_name} Change by COVID Period")
    ]

    for plot_type, title in plots:
        save_path = f"/Workspace/Users/{my_email}/meds_output/{plot_type}_{title.replace(' ', '_')}.png"

        if plot_type == "facetgrid":
            plot_facetgrid(trans_df, title, save_path=save_path)
        elif plot_type == "covid_by_lad":
            plot_covid_by_lad(trans_df, title, save_path=save_path)
        # elif plot_type == "boxplot":
        #     plot_df_boxplot(trans_df, title, save_path=save_path)
        # elif plot_type == "htn_heatmap":
        #     plot_htn_heatmap(trans_df, title, save_path=save_path)
        elif plot_type == "forest":
            reg_df = reg_coeff(trans_df)
            plot_forest(reg_df, title, save_path=save_path)


# Define your email for the save path
my_email = "ikechukwu.chukwudi@liverpool.ac.uk"

# Dictionary of dataset and base_name
medication_datasets = {
    trans_htn: "Antihypertensive Prescription Rates",
    trans_lipid: "Lipid-Lowering Prescription Rates",
    trans_plt: "Antiplatelet Prescription Rates",
    trans_t2dm: "Type 2 Diabetes Medication Initiation Rates",
    trans_doac: "DOAC Prescription Rates",
    trans_vka: "Warfarin Prescription Rates"
}

# Loop through each and generate plots with saved files
for trans_df, base_name in medication_datasets.items():
    save_all_plots(trans_df, base_name, my_email)


#––––––
#outcomes plot saving
# /Workspace/Users/ikechukwu.chukwudi@liverpool.ac.uk/output
my_email = 'ikechukwu.chukwudi@liverpool.ac.uk'
def save_all_plots(trans_df, base_name, my_email):
    # Define all plot calls with their respective titles
    plots = [
        ("facetgrid", f"Age-standardised {base_name} per 100,000"),
        ("covid_by_lad", f"Age Standardised {base_name} Per 100,000 by LAD and COVID period with Regression Line"),
        ("boxplot", f"Age-Standardised {base_name}"),
        ("htn_heatmap", f"Heat Map Showing Average Age-Standardised {base_name} by LAD"),
        ("forest", f"Forest Plot of Monthly {base_name} Change by COVID Period")
    ]

    for plot_type, title in plots:
        save_path = f"/Workspace/Users/{my_email}/output/{plot_type}_{title.replace(' ', '_')}.png"

        if plot_type == "facetgrid":
            plot_facetgrid(trans_df, title, save_path=save_path)
        elif plot_type == "covid_by_lad":
            plot_covid_by_lad(trans_df, title, save_path=save_path)
        elif plot_type == "boxplot":
            plot_df_boxplot(trans_df, title, save_path=save_path)
        elif plot_type == "htn_heatmap":
            plot_htn_heatmap(trans_df, title, save_path=save_path)
        elif plot_type == "forest":
            reg_df = reg_coeff(trans_df)
            plot_forest(reg_df, title, save_path=save_path)

save_all_plots(trans_mi, "Incident MI Rates", my_email)
save_all_plots(trans_cvd, "Cardiovascular Death Rates", my_email)
save_all_plots(trans_death, "All-Cause Death Rates", my_email)
save_all_plots(trans_stroke, "Stroke Rates", my_email)



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


#––––––SAVE FIG FUNCTIONS OF PLOTS WITH SAVE PATH AS NONE
#––––––save_fig function

def plot_facetgrid(trans_df, suptitle, save_path=None):
    df, label_order = prepare_plot(trans_df)

    ncols = 3
    nrows = int(np.ceil(len(label_order) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, lad in enumerate(label_order):
        ax = axes[i]
        lad_df = df[df['lad_label'] == lad]
        ax.plot(lad_df['date'], lad_df['rolling_avg'], label='Rolling Avg')
        for line_date in ["2020-03-01", "2022-05-01"]:
            ax.axvline(pd.Timestamp(line_date), color='grey', linestyle=':', linewidth=0.8)
        ax.set_title(lad)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12)
    fig.text(0.04, 0.5, '3-Monthly Rolling Average Rate per 100,000 People (Age-Standardised)', 
             va='center', rotation='vertical', fontsize=12)
    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0.06, 0.06, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


#–––––
def plot_covid_by_lad(trans_df, suptitle, save_path=None):
    from scipy.stats import linregress
    df, label_order = prepare_plot(trans_df)
    df['month_index'] = df.groupby(['lad_label', 'covid_period'])['date'].transform(
        lambda x: (x.dt.to_period("M") - x.min().to_period("M")).apply(lambda p: p.n)
    )
    palette = {'Pre-COVID': '#0072B2', 'During COVID': '#E69F00', 'Post-COVID': '#009E73'}
    n = len(label_order)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, lad in enumerate(label_order):
        ax = axes[i]
        df_lad = df[df['lad_label'] == lad].copy().sort_values('date').reset_index(drop=True)
        sns.scatterplot(data=df_lad, x='date', y='rolling_avg', hue='covid_period',
                        palette=palette, legend=False, ax=ax)

        for period, sub in df_lad.groupby('covid_period'):
            if len(sub) > 2:
                x = sub['month_index']
                y = sub['rolling_avg']
                slope, intercept, r, p, se = linregress(x, y)
                x_month_index = pd.Series(range(sub['month_index'].min(), sub['month_index'].max()+1))
                y_pred = intercept + slope * x_month_index
                y_ci_lower = y_pred - 1.96 * se
                y_ci_upper = y_pred + 1.96 * se
                ref_date = sub.sort_values('month_index')['date'].iloc[0].to_period("M").to_timestamp()
                x_dates = pd.date_range(start=ref_date, periods=len(x_month_index), freq='MS')
                ax.plot(x_dates, y_pred, '--', color=palette[period])
                ax.fill_between(x_dates, y_ci_lower, y_ci_upper, color=palette[period], alpha=0.2)
                ax.text(0.01, 0.95 - 0.08 * list(palette).index(period),
                        f"{period}: β={slope:.4f}, p={p:.4f}",
                        transform=ax.transAxes, fontsize=8, color=palette[period], verticalalignment='top')

        for date in ["2020-03-01", "2022-05-01"]:
            ax.axvline(pd.Timestamp(date), color='grey', linestyle=':', linewidth=0.5)
        ax.set_title(lad)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        ax.tick_params(axis='x', rotation=90, labelsize=8)

    plt.suptitle(suptitle, fontsize=16)
    fig.text(-0.04, 0.5, '3-Monthly Rolling Average Rate per 100,000 People (Age-Standardised)', 
             va='center', rotation='vertical', fontsize=12)
    fig.text(0.5, -0.05, 'Date', ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_df_boxplot(trans_df, plot_title, save_path=None):
    df, label_order = prepare_plot(trans_df)
    plt.figure(figsize=(16, 10))
    sns.boxplot(
        x='lad_label',
        y='age_standardised_rate',
        hue='covid_period',
        data=df,
        palette={'Pre-COVID': 'skyblue', 'During COVID': 'coral', 'Post-COVID': 'lightgreen'}
    )
    plt.title(plot_title, fontsize=14)
    plt.xlabel('Local Authority Districts (Mean IMD)', fontsize=10)
    plt.ylabel('Age Standardised Rates per 100,000', fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(title='COVID Period')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_htn_heatmap(trans_df, title, save_path=None):
    df, label_order = prepare_plot(trans_df)
    heatmap_data = df.groupby(['lad_label', 'covid_period'])['age_standardised_rate'].mean().unstack()
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(heatmap_data, cmap='RdYlGn_r', annot=True, fmt='.2f', linewidths=0.5)
    plt.title(title, fontsize=16)
    plt.ylabel('Region', fontsize=14)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_forest(df, title, save_path=None):
    periods = ['Pre-COVID', 'During COVID', 'Post-COVID']
    fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(12, len(df['lad_label'].unique()) * 0.5))
    lad_order = df['lad_label'].unique()[::-1]
    y_pos = range(len(lad_order))
    # Compute bounds of the error bars
    df['upper'] = df['slope'] + df['se']
    df['lower'] = df['slope'] - df['se']
    # Find the farthest extent from 0 in either direction
    max_bound = max(abs(df['upper'].max()), abs(df['lower'].min())) + 0.02
    # Symmetric limits
    min_slope = -max_bound
    max_slope = max_bound


    for i, period in enumerate(periods):
        ax = axes[i]
        df_p = df[df['covid_period'] == period].copy()
        df_p = df_p.set_index('lad_label').reindex(lad_order).reset_index()
        ax.errorbar(df_p['slope'], y_pos, xerr=df_p['se'], fmt='o', color='tab:blue')
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_xlim(min_slope, max_slope)
        ax.set_title(period)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_p['lad_label'])

    fig.suptitle(title, fontsize=14, x=0.5, ha='center')
    fig.text(0.5, 0.04, 'Slope', ha='center', fontsize=12)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

#–––––––––––––––––––––ÇOUNT OF RISK FACTORS
spark.sql ('CLEAR CACHE')

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from pandas.api.types import CategoricalDtype
import pyspark.sql.functions as F
from pyspark.sql.functions import col, to_date, collect_list, collect_set, concat_ws, lit, substring, lpad
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

#––––

# Databricks notebook source

# COMMAND ----------

codelist = spark.table("dss_corporate.gdppr_cluster_refset")

# COMMAND ----------

# DBTITLE 1,show all clusters
#codelist.groupBy("Cluster_ID", 'Cluster_Desc').count().orderBy(F.col("count"), ascending=False).display()

# COMMAND ----------

# DBTITLE 1,check cluster
(codelist
 .filter(F.col("Cluster_ID").isin(["CHOL2_COD"]))
 #.filter(~F.col("ConceptId_Description").contains("Average") & 
 #                  ~F.col("ConceptId_Description").contains("Self measured") & 
 #                  ~F.col("ConceptId_Description").contains("24 hour") & 
 #                  ~F.col("ConceptId_Description").contains("Minimum") & 
 #                  ~F.col("ConceptId_Description").contains("Maximum"))
 #.join(codelist.filter(F.col("Cluster_ID")=="ABPM_COD").select("ConceptId"), ["ConceptId"], how='left_anti')
 .select("ConceptId", "ConceptId_Description")
 .dropDuplicates()
 #.display()
)

# COMMAND ----------

# DBTITLE 1,extract codes
codelist_sel = (
    codelist.filter(F.col("Cluster_ID")=="NDABMI_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description")
            .filter(~F.col("ConceptId_Description").contains("Child"))
    .union(codelist.filter(F.col("Cluster_ID")=="NDASMOK_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))
    .union(codelist.filter(F.col("Cluster_ID")=="ALC_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))

    .union(codelist.filter(F.col("Cluster_ID")=="NDABP_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))
    .union(codelist.filter(F.col("Cluster_ID")=="FASPLASGLUC_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))
    .union(codelist.filter(F.col("Cluster_ID")=="IFCCHBAM_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))

    .union(codelist.filter(F.col("Cluster_ID")=="CHOL_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))
    .union(codelist.filter(F.col("Cluster_ID")=="HDLCCHOL_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))
    .union(codelist.filter(F.col("Cluster_ID")=="LDLCCHOL_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))
    .union(codelist.filter(F.col("Cluster_ID")=="TRIGLYC_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))

    .union(codelist.filter(F.col("Cluster_ID")=="EGFR_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))
    .union(codelist.filter(F.col("Cluster_ID")=="LFT_COD").select("Cluster_ID", "ConceptId", "ConceptId_Description"))
)

codelist_sel = (
    codelist_sel.withColumn('rf', 
                            F.when(F.col('Cluster_ID')=='NDABMI_COD', 'BMI')
                             .when(F.col('Cluster_ID')=='NDASMOK_COD', 'Smoking')
                             .when(F.col('Cluster_ID')=='ALC_COD', 'Alcohol')

                             .when(F.col('Cluster_ID')=='NDABP_COD', 'BP')
                             .when(F.col('Cluster_ID')=='FASPLASGLUC_COD', 'Fasting glucose')
                             .when(F.col('Cluster_ID')=='IFCCHBAM_COD', 'HbA1c')

                             .when(F.col('Cluster_ID')=='CHOL_COD', 'Total cholesterol')
                             .when(F.col('Cluster_ID')=='HDLCCHOL_COD', 'HDL cholesterol')
                             .when(F.col('Cluster_ID')=='LDLCCHOL_COD', 'LDL cholesterol')
                             .when(F.col('Cluster_ID')=='TRIGLYC_COD', 'Triglycerides')
                             
                             .when(F.col('Cluster_ID')=='EGFR_COD', 'eGFR')
                             .when(F.col('Cluster_ID')=='LFT_COD', 'LFT'))
)
    

# COMMAND ----------

# codelist_sel.display()

# COMMAND ----------

#codelist_sel.display()

#––––
import pyspark.sql.functions as f
my_email = "ikechukwu.chukwudi@liverpool.ac.uk"
def csv_to_sparkdf(path):
    p_df =pd.read_csv(path, keep_default_na = False)
    return spark.createDataFrame(p_df)
path1 = f'/Workspace/Users/{my_email}/lsoa_lad/part_1_lsoa_ward.csv'
path2 = f'/Workspace/Users/{my_email}/lsoa_lad/part_2_lsoa_ward.csv'
path3 = f'/Workspace/Users/{my_email}/lsoa_lad/part_3_lsoa_ward.csv'
path4 = f'/Workspace/Users/{my_email}/lsoa_lad/part_4_lsoa_ward.csv'
df1 = csv_to_sparkdf(path1)
df2 = csv_to_sparkdf(path2)
df3 = csv_to_sparkdf(path3)
df4 =csv_to_sparkdf(path4)
lw_df = df1.union(df2).union(df3).union(df4)
lw_df = lw_df.filter(f.col("LAD24CD").isin(
        "E06000049","E06000050","E06000006","E08000011","E08000013","E08000012","E08000014","E08000015","E06000007"))
liverpool_lsoa = lw_df.select(
    f.col('LSOA21CD').alias('lsoa'), f.col('LAD24NM').alias('lad'), f.col('WD24CD').alias('ward'), f.col('WD24NM').alias('ward_name')
    )
#liverpool_lsoa.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_liverpool_lsoa_20250425")

lsoa_path = "dsa_391419_j3w9t_collab.ccu071_01_liverpool_lsoa_20250425"
liv_lsoa = spark.table(lsoa_path).select(col('LSOA21CD').alias('lsoa'))


#–––––––
# COMMAND ----------

# DBTITLE 1,Import curated demographic data
demo = spark.table('dsa_391419_j3w9t_collab.hds_curated_assets__demographics_2024_04_25')
demo = (
    demo
    .filter(F.col('in_gdppr')==1)
    .select(F.col('person_id').alias('pid'), 
            F.col('date_of_birth').alias('dob'), 
            'sex', 
            F.col('ethnicity_5_group').alias('ethnicity'), 
            F.col('imd_quintile').alias('imd5'), 
            F.col('date_of_death').alias('dod'), 
            'lsoa'
            )
    .join(liv_lsoa, on='lsoa', how='inner') #IKE JOINED LIV lsoa
    .withColumn('age', F.datediff(F.to_date(F.lit('2025-03-25')), F.col('dob')) / 365.25) #IKE CHANGED END DATE FOR AGE CALC TO 2025 FROM 2018
    .withColumn('age_gp', F.when(F.col('age') < 40, '18-39')
                         .when(F.col('age') < 60, '40-59')
                         .when(F.col('age') < 80, '60-79')
                         .otherwise('80+'))
    .withColumn('sex', F.when(F.col('sex').isNull(), 'I')
                        .otherwise(F.col('sex')))
    .withColumn('ethnicity', F.when(F.col('ethnicity').isNull() | (F.col('ethnicity')=='Other ethnic group'), 
                                    'Other/Unknown')
                              .otherwise(F.col('ethnicity')))
    .withColumn('area_code_prefix', F.substring(F.col("lsoa"), 0, 3))
)
# print(f'In GDPPR, n= {demo.count()}')

demo = demo.filter(F.col('age') >= 18) 
# print(f'Adults at Nov 2018, n= {demo.count()}')

demo = demo.filter((F.col('sex') != 'I') & F.col('sex').isNotNull())
# print(f'Known sex, n= {demo.count()}')

demo2 = demo.filter((F.col('area_code_prefix')=='E01') & F.col('imd5').isNotNull())  
# print(f'Valid address in England only, n= {demo.count()}')

# demo.select('lsoa').distinct().count() = after joining lsoa; i get 1562 as expected
# display(demo.select('lsoa').distinct().count()) = 14K - joined on pid - 14k lsoas

# display(demo.select('pid').distinct().count()) = 2.3 million
# COMMAND ----------


#––––––

# DBTITLE 1,Import GP data
gp_all = spark.table('hive_metastore.dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive')

# DBTITLE 1,Find start and end dates for individuals
popu = (
    gp_all.groupBy("NHS_NUMBER_DEID")
    .agg(F.min("REPORTING_PERIOD_END_DATE").alias("start_date"))
    .withColumn('start_date', F.date_sub("start_date", 731))
    .withColumnRenamed("NHS_NUMBER_DEID", "pid")
)

# COMMAND ----------

# DBTITLE 1,Join demographics and dates
demo = (
    demo2.join(popu, ['pid'], how='inner')
    .withColumn("end_date", F.when(F.col("dod").isNull(), F.to_date(F.lit('2025-03-31'))) #DATE CHANGED IKE. FLIT TO 2025
                             .otherwise(F.col("dod")))
    )

# COMMAND ----------

# DBTITLE 1,Excluding people with follow-up <1 month
demo =  demo.filter(F.months_between(F.col('end_date'), F.col('start_date')) >= 1)
# print(f'With at least 1 month follow-up, n= {demo.count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Population count

# COMMAND ----------

# DBTITLE 1,Create multiple rows by eligible months
pop_cnt = (
    demo
    .withColumn("date", F.explode(F.sequence('start_date', 'end_date', F.expr('interval 1 month'))))
)
#print(f'Row: {pop_cnt.count()}')

# COMMAND ----------

# DBTITLE 1,Count monthly population size
pop_cnt = (
  pop_cnt
  .withColumn('age', F.datediff(F.col('date'), F.col('dob')) / 365.25)
  .withColumn('age_gp', F.when(F.col('age') < 40, '18-39')
                         .when(F.col('age') < 60, '40-59')
                         .when(F.col('age') < 80, '60-79')
                         .otherwise('80+'))
  .select('age_gp', 'sex', 'ethnicity', 'imd5','lsoa',
          F.month(F.col('date')).alias('month'), 
          F.year(F.col('date')).alias('year'))
  .groupBy('year', 'month', 'age_gp', 'sex', 'ethnicity','lsoa')
  .count()
)
#print(f'Row: {pop_cnt.count()}')

# COMMAND ----------

#"dsa_391419_j3w9t_collab.ccu071_01_popu_cnt_IC_20250417"
# COMMAND ----------

# MAGIC %md # Risk factor count

# COMMAND ----------

# DBTITLE 1,Import codelist
# MAGIC %run2



# DBTITLE 1,Save pouplation count -- I need to save this later pop_cnt- IKE
pop_cnt.write.mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_popu_cnt_IC_20250428")

pop_path = "dsa_391419_j3w9t_collab.ccu071_01_popu_cnt_IC_20250428"
popp = pop_cnt
popp.select('lsoa').distinct().count()
# lsoa_popp = popp.groupBy("lsoa").agg(F.sum("count").alias("lsoa_pop"))
# display(lsoa_popp.count())


#––––RF ALL


# DBTITLE 1,Join with codelist and demo
rf = (
    gp_all
    .filter(F.col("ProductionDate").startswith("2025-02-04")) # latest version- IKE CHANGED THIS DATE
    .select(F.col('NHS_NUMBER_DEID').alias('pid'), 
            F.col('DATE').alias('date'), 
            F.col('CODE').alias('ConceptId'), 
            F.col('VALUE1_CONDITION').alias('value1'))
    .join(codelist_sel.select('rf', 'ConceptId'), ['ConceptId'], how='inner')  # join with selected codelist
    .filter((F.col('value1').isNotNull() | F.col('rf').isin(['Smoking', 'Alcohol'])))  # contains at least 1 value except for smoking and alcohol 
    .join(demo, ['pid'], how='inner')
    .filter((F.col('date') >= F.col('start_date')) & (F.col('date') <= F.col('end_date'))) # only within eligible time
    .select('pid', 'rf', 'date', 'lsoa',
            F.year(F.col('date')).alias('year'), 
            F.month(F.col('date')).alias('month'), 
            (F.datediff(F.col('date'), F.col('dob')) / 365.25).alias('age'), 
            F.when(F.col('age') < 40, '18-39')
              .when(F.col('age') < 60, '40-59')
              .when(F.col('age') < 80, '60-79')
              .otherwise('80+')
              .alias('age_gp'), 
            'sex', 'ethnicity')
    .drop('age')
)
rf = rf.dropDuplicates() # remove multiple entries in the same date
#print(f'GP record count: {rf.count()}')

# COMMAND ----------

# DBTITLE 1,Count monthly RF
rf_cnt = rf.withColumn(
    "test_mo",
    to_date(concat_ws("-", col("year"), lpad(col("month").cast("string"), 2, "0"), lit("01")), "yyyy-MM-dd")
).drop("year", "month")
rf_cnt = (
        rf_cnt.drop('date')
          .groupBy('pid','rf', 'test_mo','lsoa')
          .count()
)

# COMMAND ----------


# DBTITLE 1,Save RF count
rf_cnt.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_rf_work_IC_20250428")
# 1hrs 15 min to run


path0 = "dsa_391419_j3w9t_collab.ccu071_01_rf_work_IC_20250428"
rf = spark.table(path0)


rf_agg = rf.groupBy("test_mo", "rf").agg(
    F.sum("count").alias("total_count")
      # optional extra stats
)


#–––PLOT RF

rf_pd = rf_agg.toPandas()


# Plot
rf_order = rf_pd.groupby('rf')['total_count'].sum().sort_values(ascending=False).index

plt.figure(figsize = (10,6))
sns.lineplot(data = rf_pd, x ="test_mo", y ="total_count", hue ="rf", hue_order = rf_order)
plt.title("Cardiometabolic Risk Factor Trend Count in Cheshire and Merseyside Over Time")
plt.legend(title ="Risk Factors", bbox_to_anchor = (1.05,1), loc="upper left")
plt.show()

#––––– DID NOT USE

def process_and_pivot(df, demo_lsoa):
    """
    This function takes two DataFrames: `df` and `demo_lsoa`.
    It performs the following:
    1. Joins the DataFrames on 'pid'.
    2. Selects 'pid', 'lsoa', 'event_mo', 'count'.
    3. Pivots the table on 'event_mo', summing 'count'.
    4. Fills NA values with 0.
    5. Divides each value in 'event_mo' columns by 100 (excluding 'lsoa').
    6. Orders the DataFrame by 'lsoa'.

    Returns:
    A processed DataFrame.
    """
    # Step 1: Join df and demo_lsoa on 'pid'
    joined_df = df.join(demo_lsoa, on="pid", how="inner")

    # Step 2: Select relevant columns
    selected_df = joined_df.select("pid", "lsoa", "event_mo", "count")

    # Step 3: Pivot the DataFrame on 'event_mo', summing 'count'
    pivot_df = selected_df.groupBy("lsoa").pivot("event_mo").agg(f.sum("count"))

    # Step 4: Fill NA values with 0
    pivot_df = pivot_df.fillna(0)

    # Step 5: Divide each 'event_mo' column by 100, excluding 'lsoa'
    columns_to_divide = [col for col in pivot_df.columns if col != 'lsoa']
    for col_name in columns_to_divide:
        pivot_df = pivot_df.withColumn(col_name, f.col(col_name) / 100)

    # Step 6: Order by 'lsoa'
    pivot_df = pivot_df.orderBy("lsoa")

    return pivot_df


# Example usage
# final_df = process_and_pivot(df, demo_lsoa)
# final_df.show()



#–––––GDPPR PROCESS

# first step - dataframe with all data we are interested in.
# from gdprr- limit all the rows with these codes- snomed, and lsoa codes and those that have NHS number deid. 
# extract year from event date - make new column.
# save this table.

# partition the data by the person NHS number and the date.
# You can also partition by the year of event date column.
##### This code is for partitioning: lets keep unique codes ie first row
####Window_Spec = Window.partitionBy("NHS_NUMBER_DEID","{BY COLNAME YEAR}").orderBy(F.col("event_date").desc())
####df = df.withColumn("p_by_year",F.row_number().over(Window_Spec))
# Give me every one with a row number of one.
# left join demographics.
# save as temporary table.

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
from pyspark.sql.functions import col, to_date, collect_list, collect_set, concat_ws, lit, substring
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

#––––––
my_email = "ikechukwu.chukwudi@liverpool.ac.uk"
def csv_to_sparkdf(path):
    p_df =pd.read_csv(path, keep_default_na = False)
    return spark.createDataFrame(p_df)
path1 = f'/Workspace/Users/{my_email}/lsoa_lad/part_1_lsoa_ward.csv'
path2 = f'/Workspace/Users/{my_email}/lsoa_lad/part_2_lsoa_ward.csv'
path3 = f'/Workspace/Users/{my_email}/lsoa_lad/part_3_lsoa_ward.csv'
path4 = f'/Workspace/Users/{my_email}/lsoa_lad/part_4_lsoa_ward.csv'
df1 = csv_to_sparkdf(path1)
df2 = csv_to_sparkdf(path2)
df3 = csv_to_sparkdf(path3)
df4 =csv_to_sparkdf(path4)
lw_df = df1.union(df2).union(df3).union(df4)
lw_df = lw_df.filter(f.col("LAD24CD").isin(
        "E06000049","E06000050","E06000006","E08000011","E08000013","E08000012","E08000014","E08000015","E06000007"))
liverpool_lsoa = lw_df.select(
    f.col('LSOA21CD').alias('lsoa'), f.col('LAD24NM').alias('lad'), f.col('WD24CD').alias('ward'), f.col('WD24NM').alias('ward_name')
    )
#liverpool_lsoa.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_liverpool_lsoa_20250425")

lsoa_path = "dsa_391419_j3w9t_collab.ccu071_01_liverpool_lsoa_20250425"
liv_lsoa = spark.table(lsoa_path).select(col('LSOA21CD').alias('lsoa'))
# liv_lsoa_list  = [row["lsoa"] for row in liv_lsoa.collect()]

#––––––
# COMMAND ----------

# DBTITLE 1,Import curated demographic data and clean for valid age>18 sex and address. inner join with LIV_LSOA to filter for our pop.
demo = spark.table('dsa_391419_j3w9t_collab.hds_curated_assets__demographics_2024_04_25')
demo = (
    demo
    .select(f.col('person_id').alias('pid'), 
            f.col('date_of_birth').alias('dob'), 
            'sex', 
            f.col('ethnicity_5_group').alias('ethnicity'), 
            f.col('imd_quintile').alias('imd5'), 
            f.col('date_of_death').alias('dod'), 
            'lsoa'
            )
    .join(liv_lsoa, on='lsoa', how='inner') #IKE JOINED LIV LSOA
    .withColumn('age', f.datediff(f.to_date(f.lit('2025-03-25')), f.col('dob')) / 365.25) #IKE CHANGED END DATE FOR AGE CALC TO 2025 FROM 2018
    .withColumn('age_gp', f.when(f.col('age') < 40, '18-39')
                         .when(f.col('age') < 60, '40-59')
                         .when(f.col('age') < 80, '60-79')
                         .otherwise('80+'))
    .withColumn('sex', f.when(f.col('sex').isNull(), 'I')
                        .otherwise(f.col('sex')))
    .withColumn('ethnicity', f.when(f.col('ethnicity').isNull() | (f.col('ethnicity')=='Other ethnic group'), 
                                    'Other/Unknown')
                              .otherwise(f.col('ethnicity')))
    .withColumn('area_code_prefix', f.substring(f.col("lsoa"), 0, 3))
)
# print(f'In GDPPR, n= {demo.count()}')

demo = demo.filter(f.col('age') >= 18) 
demo = demo.filter((f.col('sex') != 'I') & f.col('sex').isNotNull())
demo = demo.filter((f.col('area_code_prefix')=='E01') & f.col('imd5').isNotNull())  
demo= demo.select('pid','sex', 'lsoa','age', 'age_gp')
demo = demo.select('pid')


#–––––
#join gdppr to demo on pid as it is cleaned with valid age and location and sex.
gdpp = spark.table("hive_metastore.dars_nic_391419_j3w9t.gdppr_dars_nic_391419_j3w9t")
gdppr = gdpp.select(f.col('NHS_NUMBER_DEID').alias('pid'), f.col('sex'), f.col('LSOA').alias('lsoa'), f.col('DATE').alias('event_date'), f.col('CODE').alias('snomed_code'))
gdppr = gdppr.join(demo, on = 'pid', how = 'inner')

#–––––
my_email = "ikechukwu.chukwudi@liverpool.ac.uk"
def csv_to_sparkdf(path):
    p_df =pd.read_csv(path, keep_default_na = False)
    return spark.createDataFrame(p_df)
pcd_mi_path = f'/Workspace/Users/{my_email}/lsoa_lad/nhsd_pcd_mi_code.csv' #source is NHS digital primary care domain refsets 2024 dec version for MI (opencodelist)
hdr_uk_mi_path = f'/Workspace/Users/{my_email}/lsoa_lad/hdr_uk_mi_ccu046.csv'
mi1 = csv_to_sparkdf(pcd_mi_path)
mi2=csv_to_sparkdf(hdr_uk_mi_path)

mi1 =mi1.select(f.col('code').alias('snomed_code'),f.col('term').alias('descr'))
mi2 =mi2.select(f.col('Code').alias('snomed_code'),f.col('Description').alias('descr'))
mi_code = mi1.union(mi2)
mi_code = mi_code.dropDuplicates(['snomed_code'])
# mi_code.display()

mi_df = gdppr.join(mi_code, on  = 'snomed_code', how= 'inner')
mi_df = mi_df.select(
    'pid', 'sex', 'lsoa', 'event_date','snomed_code', 'descr'
    ).withColumn(
        'event_mo', to_date(substring('event_date',0,7))
    )
mi_df = mi_df.filter(f.col('event_mo')>='2019-01-01')
mi_df = mi_df.dropDuplicates(['event_mo', 'pid'])
mi_pd = mi_df.groupBy('event_mo').count()

# mi_pd.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_mi_dff2_liverpool_ic_28042025")

#––––
mi_df_path = "dsa_391419_j3w9t_collab.ccu071_01_mi_dff2_liverpool_ic_28042025"
grouped_mi = spark.table(mi_df_path)
# grouped_mi = grouped_mi.dropDuplicates(['event_mo', 'pid'])
mi_pd = grouped_mi.groupBy('event_mo').agg(f.sum('count').alias('gpmi_count'))

minap_path = "hive_metastore.dars_nic_391419_j3w9t.nicor_minap_dars_nic_391419_j3w9t"
minap_df = spark.table(minap_path)
minap_df = minap_df.select(
    f.col('ARRIVAL_AT_HOSPITAL').alias('arrive_time'), f.col('NHS_NUMBER_DEID').alias('pid')
    ).withColumn("arrive_date", f.col("arrive_time").substr(1,10))

minap_df = minap_df.select('arrive_date', 'pid')
minap_df = minap_df.join(demo, on='pid', how ='inner')
minap_df = minap_df.withColumn('event_mo', f.col("arrive_date").substr(0,7))
minap_df = minap_df.dropDuplicates(['event_mo', 'pid'])
minap_df = minap_df.filter(f.col('event_mo') >= '2019-01-01')
count_minap  = minap_df.groupBy('event_mo').count()
count_minap = count_minap.withColumnRenamed("count", "minap_count")

minap_gdppr = mi_pd.join(count_minap, on = "event_mo", how = "inner")
# display(count_snap)
minap_gdppr = minap_gdppr.withColumn("diff", col('gpmi_count')-col('minap_count'))


#––––

df = minap_gdppr.toPandas()
df['event_mo'] = pd.to_datetime(df['event_mo'], format='%Y-%m')

df =df.sort_values('event_mo')
plt.figure(figsize=(10, 6))
plt.plot(df['event_mo'], df['gpmi_count'], label='GDPPR MI Count')
plt.plot(df['event_mo'], df['minap_count'], label='MINAP Count')
plt.plot(df['event_mo'], df['diff'], label = "Difference")

plt.xlabel('Event Month')
plt.ylabel('Count')
plt.title('GDPPR and MINAP MI Counts Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

#–––––GDPPR STROKE

my_email = "ikechukwu.chukwudi@liverpool.ac.uk"
def csv_to_sparkdf(path):
    p_df =pd.read_csv(path, keep_default_na = False)
    return spark.createDataFrame(p_df)
stroke_path = f'/Workspace/Users/{my_email}/lsoa_lad/hdr_uk_stroke_ccu046.csv' #source is NHS digital primary care domain refsets 2024 dec version for MI (opencodelist)
stroke = csv_to_sparkdf(stroke_path)
stroke =stroke.select(f.col('Code').alias('snomed_code'),f.col('Description').alias('descr'))

stroke_df = gdppr.join(stroke, on  = 'snomed_code', how= 'inner')
stroke_df = stroke_df.select(
    'pid', 'sex', 'lsoa', 'event_date','snomed_code', 'descr'
    ).withColumn(
        'event_mo', to_date(substring('event_date',0,7))
    )
stroke_df = stroke_df.filter(f.col('event_mo')>='2019-01-01')
stroke_df = stroke_df.dropDuplicates(['pid', 'event_mo'])
stroke_pd = stroke_df.groupBy('event_mo', 'pid').count()



# stroke_pd.write.format("parquet").mode("overwrite").saveAsTable("dsa_391419_j3w9t_collab.ccu071_01_stroke_dff2_liverpool_ic_28042025")

#–––––

stroke_df_path = "dsa_391419_j3w9t_collab.ccu071_01_stroke_dff2_liverpool_ic_28042025"
grouped_stroke = spark.table(stroke_df_path)
stroke_pd = grouped_stroke.groupBy('event_mo').agg(f.sum('count').alias('gpstroke_count'))

#SSNAP Stroke Count
snap_path = "hive_metastore.dars_nic_391419_j3w9t.ssnap_dars_nic_391419_j3w9t"
df_snap = spark.table(snap_path)
df_snap = df_snap.select(
    f.col('S1ONSETDATETIME').alias('event_date'), f.col('Person_ID_DEID').alias('pid'), f.col('LSOA_OF_RESIDENCE').alias('lsoa'), f.col('S1DIAGNOSIS').alias('diag')
    ).withColumn("event_date", f.col("event_date").substr(1,10))

# df_snap = df_snap.filter(f.col("diag") == "S")
df_snap = df_snap.withColumn("event_mo", to_date(substring("event_date",0,7)))
df_snap = df_snap.join(demo, on ='pid', how='inner')
df_snap = df_snap.filter(f.col("event_mo") >= "2019-01-01")


count_snap = df_snap.groupBy("event_mo").count()
count_snap = count_snap.withColumnRenamed("count", "snap_count")
#Join monthly HES stroke count with SSNAP Stroke Count and calculate difference
snap_gdppr = stroke_pd.join(count_snap, on = "event_mo", how = "inner")
# display(count_snap)

snap_gdppr = snap_gdppr.withColumn("diff", col('gpstroke_count')-col('snap_count'))
# display(snap_hes)

#––––––

df = snap_gdppr.toPandas()

df['event_mo'] = pd.to_datetime(df['event_mo'], format='%Y-%m')

df =df.sort_values('event_mo')
plt.figure(figsize=(10, 6))
plt.plot(df['event_mo'], df['gpstroke_count'], label='GDPPR Stroke Count')
plt.plot(df['event_mo'], df['snap_count'], label='SSNAP Count')
plt.plot(df['event_mo'], df['diff'], label = "Difference")

plt.xlabel('Event Month')
plt.ylabel('Count')
plt.title('GDPPR and SSNAP Stroke Counts Over Time')
# plt.title('SSNAP Counts Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
