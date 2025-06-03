'''
This is a function that is integral to the different notebooks used in generating the outcome and medications data for ccu071
The below commented out functions are the functions in this notebook functions
This function is called in other notebooks
'''
#csv_to_sparkdf
# predict_population_2024
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
