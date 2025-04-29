import polars as pl
import numpy as np # Still conceptually useful, not strictly needed for counting

# --- Configuration ---
input_file = './data/constants.csv'
output_file = './analysis-constants/irregular_results.csv'

group_cols = ['country', 'period']
dependent_vars = [
    'Real GDP Growth',
    'Inflation',
    'Unemployment',
    'Budget Balance',
    'Approval Index'
]

# --- Data Loading ---
print(f"Loading data from {input_file}...")
try:
    df = pl.read_csv(input_file)
    print("Data loaded successfully.")
except pl.exceptions.PolarsError as e:
    print(f"Error loading data: {e}")
    exit()

# --- Analyze Specific Values ---
print("Starting analysis of specific values (NaN, Inf, 0, -100) for each country/period group...")

specific_counts_list = []

# Group by country and period
grouped = df.group_by(group_cols)

for (country, period), group_df in grouped:
    print(f"Analyzing group: country={country}, period={period}")

    nobs = len(group_df) # Total observations in this group

    # Select only the dependent variables for analysis in this group
    group_dep_vars_df = group_df.select(dependent_vars)

    # Analyze each dependent variable column within this group
    for dep_var in dependent_vars:
        col_series = group_dep_vars_df[dep_var]

        nan_count = col_series.is_nan().sum()
        inf_count = col_series.is_infinite().sum()
        zero_count = (col_series == 0).sum()
        minus100_count = (col_series == -100).sum()

        # Total "irregular" count (NaN + Inf)
        total_irregular_count = nan_count + inf_count

        # Store results for this column in this group
        specific_counts_list.append({
            'country': country,
            'period': period,
            'Variable': dep_var,
            'NaN Count': nan_count,
            'Inf Count': inf_count,
            'Zero Count': zero_count,
            'Minus100 Count': minus100_count,
            'Total Irregular Count': total_irregular_count, # Still NaN + Inf
            'Total Observations': nobs
        })

print("\nAnalysis complete. Consolidating results...")

# --- Consolidate and Export Results ---
if specific_counts_list:
    # Create Polars DataFrame from the list of dictionaries
    results_df = pl.DataFrame(specific_counts_list)

    # Define schema explicitly for clarity
    results_df = results_df.with_columns([
        pl.col('country').cast(pl.Utf8),
        pl.col('period').cast(pl.Int64),
        pl.col('Variable').cast(pl.Utf8),
        pl.col('NaN Count').cast(pl.Int64),
        pl.col('Inf Count').cast(pl.Int64),
        pl.col('Zero Count').cast(pl.Int64),
        pl.col('Minus100 Count').cast(pl.Int64),
        pl.col('Total Irregular Count').cast(pl.Int64),
        pl.col('Total Observations').cast(pl.Int64)
    ])

    # Write to CSV
    print(f"Exporting specific value counts to {output_file}...")
    try:
        results_df.write_csv(output_file)
        print("Results exported successfully.")
    except Exception as e:
        print(f"Error exporting results: {e}")
else:
    print("No specific value counts generated (possibly due to data loading errors). No output file created.")