#!/usr/bin/env python3

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

INPUT_CSV_PATH = './data/constants.csv'
TARGET_PERIOD = 8
COUNTRY_COLUMN_NAME = 'country'
PERIOD_COLUMN_NAME = 'period'
APPROVAL_INDEX_COLUMN_NAME = 'Approval Index'
OUTPUT_FILE = None

CUSTOM_COUNTRY_ORDER = [
    'cambodia', 'laos', 'bangladesh', 'jamaica', 'bulgaria',
    'argentina', 'brunei', 'india', 'australia'
]

CURRENT_DATE = "2025-04-29 03:34:35"
USERNAME = "trietnguyen159753"

def create_violin_plots(filtered_df):
    df_pandas = filtered_df.to_pandas()

    df_pandas[COUNTRY_COLUMN_NAME] = pd.Categorical(
        df_pandas[COUNTRY_COLUMN_NAME],
        categories=CUSTOM_COUNTRY_ORDER,
        ordered=True
    )

    df_pandas_ordered = df_pandas[df_pandas[COUNTRY_COLUMN_NAME].isin(CUSTOM_COUNTRY_ORDER)].copy()

    if df_pandas_ordered.empty:
        print("No data rows remaining after filtering by custom country list.")
        return

    countries_for_plot = CUSTOM_COUNTRY_ORDER
    num_countries = len(countries_for_plot)

    country_means = (
        df_pandas_ordered
        .groupby(COUNTRY_COLUMN_NAME)[APPROVAL_INDEX_COLUMN_NAME]
        .mean()
    )

    plt.figure(figsize=(max(10, num_countries * 0.8), 6))

    ax = sns.violinplot(
        data=df_pandas_ordered,
        x=COUNTRY_COLUMN_NAME,
        y=APPROVAL_INDEX_COLUMN_NAME,
        palette='BrBG',
        hue=COUNTRY_COLUMN_NAME,
        inner='box',
        cut=0,
        order=countries_for_plot
    )

    for i, country in enumerate(countries_for_plot):
        if country in country_means.index:
            mean_val = country_means.loc[country]

            plt.scatter(i, mean_val, color='white', s=40, zorder=5,
                        marker='o', edgecolor='black', linewidth=1)

            plt.annotate(f'{mean_val:.1f}',
                         xy=(i, mean_val),
                         xytext=(10, 0),
                         textcoords='offset points',
                         ha='left',
                         va='center',
                         fontsize=10)

    y_min = 0
    y_max = 100
    y_ticks = np.arange(y_min, y_max + 25, 25)
    plt.ylim(y_min, y_max)
    plt.yticks(y_ticks)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.title(f'Distribution of {APPROVAL_INDEX_COLUMN_NAME} by Country (Period {TARGET_PERIOD})',
              fontsize=14, pad=20)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel(f'{APPROVAL_INDEX_COLUMN_NAME}', fontsize=12)

    if num_countries > 5:
        plt.xticks(rotation=45, ha='right')

    plt.figtext(0.99, 0.01, f'Generated: {CURRENT_DATE} by {USERNAME}',
                fontsize=8, ha='right', va='bottom', alpha=0.7)

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                              markeredgecolor='black', markersize=8, label='Mean')]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()

    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {OUTPUT_FILE}")

    plt.show()

def main():
    print(f"Reading consolidated data from: {INPUT_CSV_PATH}")
    print(f"Filtering data for Period: {TARGET_PERIOD}")
    print(f"Creating violin plots for '{APPROVAL_INDEX_COLUMN_NAME}' grouped by '{COUNTRY_COLUMN_NAME}' in custom order.")

    df = pl.read_csv(INPUT_CSV_PATH)

    df = df.with_columns(pl.col(PERIOD_COLUMN_NAME).cast(pl.Int64))

    df_filtered = df.filter(pl.col(PERIOD_COLUMN_NAME) == TARGET_PERIOD)

    print(f"Found {df_filtered.height} rows for Period {TARGET_PERIOD}.")

    if APPROVAL_INDEX_COLUMN_NAME not in df_filtered.columns:
         print(f"Error: Column '{APPROVAL_INDEX_COLUMN_NAME}' not found in the data.")
         return

    df_clean = df_filtered.filter(pl.col(APPROVAL_INDEX_COLUMN_NAME).is_not_null())

    if df_clean.height == 0:
        print(f"No valid '{APPROVAL_INDEX_COLUMN_NAME}' data found for Period {TARGET_PERIOD}.")
        return

    create_violin_plots(df_clean)
    print("Violin plot displayed.")

if __name__ == "__main__":
    main()