import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration constants
INPUT_CSV_PATH = './data/constants.csv'
OUTPUT_CSV_PATH = 'desc_stat.csv'
TARGET_PERIOD = 8
COUNTRY_COLUMN_NAME = 'country'
PERIOD_COLUMN_NAME = 'period'
APPROVAL_INDEX_COLUMN_NAME = 'Approval Index'

def main():
    print(f"Reading consolidated data from: {INPUT_CSV_PATH}")
    print(f"Filtering data for Period: {TARGET_PERIOD}")
    print(f"Calculating descriptive statistics for '{APPROVAL_INDEX_COLUMN_NAME}' grouped by '{COUNTRY_COLUMN_NAME}' for Period {TARGET_PERIOD}.")
    print(f"Output will be saved to: {OUTPUT_CSV_PATH}\n")

    # Read and filter data
    df = pl.read_csv(INPUT_CSV_PATH)
    df_period_filtered = df.filter(pl.col(PERIOD_COLUMN_NAME) == TARGET_PERIOD)
    
    print(f"Found {df_period_filtered.height} rows for Period {TARGET_PERIOD}.")

    # Calculate statistics by country
    grouped_stats = (
        df_period_filtered
        .filter(pl.col(APPROVAL_INDEX_COLUMN_NAME).is_not_null())
        .group_by(COUNTRY_COLUMN_NAME)
        .agg(
            pl.col(APPROVAL_INDEX_COLUMN_NAME).mean().alias("mean"),
            pl.col(APPROVAL_INDEX_COLUMN_NAME).std().alias("sd"),
            pl.col(APPROVAL_INDEX_COLUMN_NAME).min().alias("min"),
            pl.col(APPROVAL_INDEX_COLUMN_NAME).quantile(0.25).alias("Q1"),
            pl.col(APPROVAL_INDEX_COLUMN_NAME).quantile(0.50).alias("Q2"),
            pl.col(APPROVAL_INDEX_COLUMN_NAME).quantile(0.75).alias("Q3"),
            pl.col(APPROVAL_INDEX_COLUMN_NAME).max().alias("Q4")
        )
    )

    # Calculate coefficient of variation
    grouped_stats = grouped_stats.with_columns(
        (pl.col("sd") / pl.col("mean")).alias("cv")
    )

    # Reorder columns
    output_columns = [
        'country', 'mean', 'sd', 'min', 
        'Q1', 'Q2', 'Q3', 'Q4', 'cv'
    ]
    stats_df = grouped_stats.select(output_columns)

    # Write statistics to CSV file
    stats_df.write_csv(OUTPUT_CSV_PATH, float_precision=6)
    print(f"\nSuccessfully saved descriptive statistics for Period {TARGET_PERIOD} to {OUTPUT_CSV_PATH}")

    # Create visualization
    create_boxplot(df_period_filtered)

def create_boxplot(df_period_filtered):
    """Creates and displays a boxplot for the filtered data."""
    print(f"\nGenerating boxplot for Period {TARGET_PERIOD} data...")
    
    # Convert to pandas for plotting (seaborn works with pandas)
    df_for_plot = df_period_filtered.to_pandas()
    
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(x=COUNTRY_COLUMN_NAME, y=APPROVAL_INDEX_COLUMN_NAME, data=df_for_plot)

    # Add mean markers
    country_means = (
        df_period_filtered
        .group_by(COUNTRY_COLUMN_NAME)
        .agg(pl.col(APPROVAL_INDEX_COLUMN_NAME).mean())
        .to_pandas()
        .set_index(COUNTRY_COLUMN_NAME)
        .iloc[:, 0]
    )
    
    plot_order = [tick.get_text() for tick in ax.get_xticklabels()]
    ordered_means = country_means.reindex(plot_order)

    for i, country in enumerate(ordered_means.index):
        mean_val = ordered_means[country]
        if not np.isnan(mean_val):
            ax.plot(i, mean_val, 'ro', ms=8, markeredgecolor='black', markeredgewidth=1,
                    label='Mean' if i == 0 else "")

    # Format plot
    ax.legend(loc='best')
    
    # Set y-axis ticks
    y_min_data = df_period_filtered.select(pl.col(APPROVAL_INDEX_COLUMN_NAME).min()).item()
    y_max_data = df_period_filtered.select(pl.col(APPROVAL_INDEX_COLUMN_NAME).max()).item()
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

if __name__ == "__main__":
    main()