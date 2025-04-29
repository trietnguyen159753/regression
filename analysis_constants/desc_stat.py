import pandas as pd
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = 'Constants'
OUTPUT_CSV_PATH = 'desc_stat.csv'

APPROVAL_INDEX_COLUMN_NAME = 'Approval Index'

print(f"Scanning directory: {ROOT_DIR}")
print(f"Descriptive statistics for '{APPROVAL_INDEX_COLUMN_NAME}' will be consolidated into: {OUTPUT_CSV_PATH}\n")
print(f"Generating boxplots for '{APPROVAL_INDEX_COLUMN_NAME}' from the last period of each country.")

all_country_stats = []
all_country_dataframes_for_plot = []

country_dirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
country_dirs.sort()

for country_name in country_dirs:
    country_input_path = os.path.join(ROOT_DIR, country_name)
    print(f"Processing country: {country_name}")

    country_files = [f for f in os.listdir(country_input_path) if f.endswith('.csv') and os.path.isfile(os.path.join(country_input_path, f))]
    country_files.sort()

    last_file_name = country_files[-1]
    last_file_path = os.path.join(country_input_path, last_file_name)
    print(f"  Processing last period file: {last_file_name}")

    df = pd.read_csv(last_file_path)

    approval_series = df[APPROVAL_INDEX_COLUMN_NAME].dropna()

    country_stats = {'country': country_name}

    if not approval_series.empty:
        mean_val = approval_series.mean()
        std_val = approval_series.std()
        min_val = approval_series.min()
        q1_val = approval_series.quantile(0.25)
        q2_val = approval_series.quantile(0.50)
        q3_val = approval_series.quantile(0.75)
        q4_val = approval_series.max()

        cv_val = std_val / mean_val

        country_stats.update({
            'mean': mean_val,
            'standard deviation': std_val,
            'min': min_val,
            'Q1': q1_val,
            'Q2': q2_val,
            'Q3': q3_val,
            'Q4': q4_val,
            'coefficient of variation': cv_val
        })

        country_df_for_plot = pd.DataFrame({
            APPROVAL_INDEX_COLUMN_NAME: approval_series,
            'country': country_name
        })
        all_country_dataframes_for_plot.append(country_df_for_plot)

    all_country_stats.append(country_stats)

print("\nFinished processing countries.")

stats_df = pd.DataFrame(all_country_stats)

output_columns = ['country', 'mean', 'standard deviation', 'min', 'Q1', 'Q2', 'Q3', 'Q4', 'coefficient of variation']

stats_df = stats_df.reindex(columns=output_columns)

stats_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6g')
print(f"Successfully saved descriptive statistics to {OUTPUT_CSV_PATH}")

if all_country_dataframes_for_plot:
    print("\nGenerating boxplots...")
    combined_df_for_plot = pd.concat(all_country_dataframes_for_plot, ignore_index=True)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='country', y=APPROVAL_INDEX_COLUMN_NAME, data=combined_df_for_plot)
    
    country_means = combined_df_for_plot.groupby('country')[APPROVAL_INDEX_COLUMN_NAME].mean()
    
    # Add mean markers to the boxplot
    for i, country in enumerate(country_means.index):
        mean_val = country_means[country]
        ax.plot(i, mean_val, 'ro', ms=8, markeredgecolor='black', markeredgewidth=1, 
                label='Mean' if i == 0 else "")  # Only add label once
    
    # Add a legend (will only show one entry for 'Mean')
    ax.legend(loc='best')
    
    y_min = np.floor(combined_df_for_plot[APPROVAL_INDEX_COLUMN_NAME].min() / 25) * 25
    y_max = np.ceil(combined_df_for_plot[APPROVAL_INDEX_COLUMN_NAME].max() / 25) * 25
    y_ticks = np.arange(y_min, y_max + 25, 25)
    plt.yticks(y_ticks)

    plt.title(f'Distribution of {APPROVAL_INDEX_COLUMN_NAME} by Country (Last Period)')
    plt.xlabel('Country')
    plt.ylabel(APPROVAL_INDEX_COLUMN_NAME)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()
    print("Boxplot displayed.")

else:
    print("\nNo valid data collected for plotting '{APPROVAL_INDEX_COLUMN_NAME}'. Boxplots will not be generated.")
