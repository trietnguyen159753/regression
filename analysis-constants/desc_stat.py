import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV_PATH = './data/constants.csv'
OUTPUT_CSV_PATH = 'desc_stat.csv'
TARGET_PERIOD = 8
COUNTRY_COLUMN_NAME = 'country'
PERIOD_COLUMN_NAME = 'period'
APPROVAL_INDEX_COLUMN_NAME = 'Approval Index'

print(f"Reading consolidated data from: {INPUT_CSV_PATH}")
print(f"Filtering data for Period: {TARGET_PERIOD}")
print(f"Calculating descriptive statistics for '{APPROVAL_INDEX_COLUMN_NAME}' grouped by '{COUNTRY_COLUMN_NAME}' for Period {TARGET_PERIOD}.")
print(f"Output will be saved to: {OUTPUT_CSV_PATH}\n")

df = pd.read_csv(INPUT_CSV_PATH)

df_period_filtered = df[df[PERIOD_COLUMN_NAME] == TARGET_PERIOD].copy()

print(f"Found {len(df_period_filtered)} rows for Period {TARGET_PERIOD}.")

approval_series = df_period_filtered.set_index(COUNTRY_COLUMN_NAME)[APPROVAL_INDEX_COLUMN_NAME].dropna()


grouped_stats = approval_series.groupby(COUNTRY_COLUMN_NAME).agg(
    mean='mean',
    std='std',
    min='min',
    Q1=lambda x: x.quantile(0.25),
    Q2=lambda x: x.quantile(0.50),
    Q3=lambda x: x.quantile(0.75),
    Q4='max'
).reset_index()

grouped_stats['coefficient of variation'] = grouped_stats['std'] / grouped_stats['mean']

grouped_stats = grouped_stats.rename(columns={
    COUNTRY_COLUMN_NAME: 'country',
    'std': 'standard deviation'
})

output_columns = ['country', 'mean', 'standard deviation', 'min', 'Q1', 'Q2', 'Q3', 'Q4', 'coefficient of variation']
stats_df = grouped_stats.reindex(columns=output_columns)

stats_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6g')
print(f"\nSuccessfully saved descriptive statistics for Period {TARGET_PERIOD} to {OUTPUT_CSV_PATH}")

print("\nGenerating boxplot for Period {TARGET_PERIOD} data...")
plt.figure(figsize=(12, 7))

ax = sns.boxplot(x=COUNTRY_COLUMN_NAME, y=APPROVAL_INDEX_COLUMN_NAME, data=df_period_filtered)

country_means = df_period_filtered.groupby(COUNTRY_COLUMN_NAME)[APPROVAL_INDEX_COLUMN_NAME].mean()

plot_order = [tick.get_text() for tick in ax.get_xticklabels()]
ordered_means = country_means.reindex(plot_order)

for i, country in enumerate(ordered_means.index):
    mean_val = ordered_means[country]
    if pd.notna(mean_val):
        ax.plot(i, mean_val, 'ro', ms=8, markeredgecolor='black', markeredgewidth=1,
                label='Mean' if i == 0 else "")

ax.legend(loc='best')

y_min_data = df_period_filtered[APPROVAL_INDEX_COLUMN_NAME].min()
y_max_data = df_period_filtered[APPROVAL_INDEX_COLUMN_NAME].max()
y_min_axis = np.floor(y_min_data / 25) * 25
y_max_axis = np.ceil(y_max_data / 25) * 25
y_ticks = np.arange(y_min_axis, y_max_axis + 25, 25)
plt.yticks(y_ticks)


plt.title(f'Distribution of {APPROVAL_INDEX_COLUMN_NAME} by Country (Period {TARGET_PERIOD})')
plt.xlabel('Country')
plt.ylabel(APPROVAL_INDEX_COLUMN_NAME)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
print("Boxplot displayed.")