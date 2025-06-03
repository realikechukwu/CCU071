import pandas as pd

def combine_pop_data(file_paths: list) -> pd.DataFrame:
    """
    Combines population CSVs and maps age groups.
    
    Args:
        file_paths (list): List of file paths to yearly population CSVs.
    
    Returns:
        pd.DataFrame: Long-format DataFrame with Year, age_gp, lad, population.
    """
    age_mapping = {
        'Age 18': '18-39',
        'Age 19': '18-39',
        'Aged 20-24': '18-39',
        'Aged 25-29': '18-39',
        'Aged 30-34': '18-39',
        'Aged 35-39': '18-39',
        'Aged 40-44': '40-59',
        'Aged 45-49': '40-59',
        'Aged 50-54': '40-59',
        'Aged 55-59': '40-59',
        'Aged 60-64': '60-79',
        'Aged 65-69': '60-79',
        'Aged 70-74': '60-79',
        'Aged 75-79': '60-79',
        'Aged 80-84': '80+',
        'Aged 85+': '80+'
    }

    dfs = []
    for file_path in file_paths:
        year = int(file_path.split('_')[-1].split('.')[0])  # Assumes filename like "df_2018.csv"
        df = pd.read_csv(file_path)
        df['Year'] = year
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['age_gp'] = combined_df['Age'].map(age_mapping)

    # Reorder and clean
    new_order = ['Year', 'age_gp'] + [col for col in combined_df.columns if col not in ['Year', 'age_gp', 'Age']]
    combined_df = combined_df[new_order]

    # Convert year to mid-year datetime
    combined_df['Year'] = pd.to_datetime(combined_df['Year'].astype(str) + '-06')

    # Melt to long format
    df_long = combined_df.melt(id_vars=['Year', 'age_gp'], var_name='lad', value_name='population')

    df_long['population'] = pd.to_numeric(df_long['population'].astype(str).str.replace(',', '', regex=False),errors='coerce')

    return df_long


def transform_and_save_population_data(df_long: pd.DataFrame, lad_code_map: dict, output_path: str) -> None:
    """
    Groups and maps LAD codes, then saves to CSV.

    Args:
        df_long (pd.DataFrame): Long-format DataFrame from combine_population_data().
        lad_code_map (dict): Dictionary mapping LAD names to LAD codes.
        output_path (str): Path to save the final CSV.
    """
    summary_df = df_long.groupby(['Year', 'age_gp', 'lad'])['population'].sum().reset_index()
    summary_df['lad_code'] = summary_df['lad'].map(lad_code_map)

    final_df = summary_df[['Year', 'age_gp', 'lad', 'lad_code', 'population']]
    final_df.to_csv(output_path, index=False)

# How to use
# file_paths = [f"df_{year}.csv" for year in range(2018, 2024)]
# lad_code_map = {
#     'Cheshire East': 'E06000049',
#     'Cheshire West and Chester': 'E06000050',
#     'Halton': 'E06000006',
#     'Knowsley': 'E08000011',
#     'Liverpool': 'E08000012',
#     'Sefton': 'E08000014',
#     'St. Helens': 'E08000013',
#     'Warrington': 'E06000007',
#     'Wirral': 'E08000015'
# }

# df_long = combine_pop_data(file_paths)
# transform_and_save_population_data(df_long, lad_code_map, "28_may_liv_pop_18_23.csv")