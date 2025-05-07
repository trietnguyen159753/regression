import pandas as pd
import polars as pl
from polars import selectors as ps
import statsmodels.api as sm
import os
import numpy as np
from matplotlib import pyplot as plt
import datashader as ds
import colorcet as cc
from datashader import transfer_functions as tf

# Configuration
CONFIG = {
    'output_viz_dir': './analysis-constants/visualization/cooks-distance/',
    'output_sum_dir': './analysis-constants/',
    'data_file': './Constants_prelim.parquet',
    'outlier_iqr_threshold': 10,
    'output_variables': [
        'Real GDP Growth', 'Inflation', 'Unemployment', 
        'Budget Balance', 'Approval Index'
    ],
    'input_variables': [
        'Interest Rate', 'Vat Rate', 'Corporate Tax',
        'Government Expenditure', 'Import Tariff'
    ]
}

def load_data():
    """Load and clean initial data."""
    df = pl.read_parquet(CONFIG['data_file'])
    return df.with_columns(
    pl.col(pl.Float64).replace([float('inf'), -float('inf')], None)
    ).drop_nulls()

def get_unique_combinations(df):
    """Get unique country-period combinations."""
    return df.select(['country', 'period']).unique().rows()

def filter_outliers(df, variables, n_iqr=10):
    """Remove outliers based on standard deviation."""
    df_filtered = df.clone()
    
    # Create a combined filter expression
    filter_expr = True
    for variable in variables:
        median = df[variable].median()
        q1 = df[variable].quantile(0.25)
        q3 = df[variable].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = median - n_iqr * iqr
        upper_bound = median + n_iqr * iqr
        filter_expr = filter_expr & (
            (pl.col(variable) >= lower_bound) & 
            (pl.col(variable) <= upper_bound)
        )
    
    return df_filtered.filter(filter_expr)

def calculate_all_cooks_distances(df, input_vars, output_vars):
    """Calculate Cook's distance for all output variables."""
    X_pd = df.select(input_vars).to_pandas()
    
    cooks_distances = {}
    for output_var in output_vars:
        y_pd = df.select(output_var).to_pandas()
        model = sm.OLS(y_pd, X_pd).fit()
        cooks_distances[output_var] = model.get_influence().cooks_distance[0]
    
    return cooks_distances
def plot_cooks_distance(cooks_d, output_var, country, period, out_dir, cutoff):
    """Create and save a Cook's distance plot."""
    # Create directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup plot data
    plot_data = pd.DataFrame({
        'index': range(len(cooks_d)),
        'cooks_distance': cooks_d
    })
    
    x_range_data = [0, len(cooks_d) - 1]
    y_max = max(cooks_d) if len(cooks_d) > 0 else 0
    y_range_data = [0, y_max * 1.1]
    
    # Create plot using datashader
    canvas = ds.Canvas(plot_width=800, plot_height=500, x_range=x_range_data, y_range=y_range_data)
    agg = canvas.points(plot_data, 'index', 'cooks_distance')
    img = tf.shade(agg, cmap=cc.blues)
    
    # Add annotations with matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img.to_pil(), extent=[x_range_data[0], x_range_data[1], y_range_data[0], y_range_data[1]], 
              origin='upper', aspect='auto')
    
    ax.axhline(y=cutoff, color='red', linestyle='--', label=f'Cutoff: {cutoff:.4f}', alpha=0.5)
    ax.grid(False)
    ax.set_ylabel("Cook's Distance")
    ax.set_xlim(x_range_data)
    ax.set_ylim(y_range_data)
    
    # Save plot
    plot_path = os.path.join(out_dir, f"{output_var}.png")
    fig.savefig(plot_path, dpi=72, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
def process_country_period(df, country, period):
    """Process a single country-period combination."""
    print(f"Processing {country} - {period}")
    
    # Filter data for specific country and period
    df_cp = df.filter((pl.col('country') == country) & (pl.col('period') == period))
    initial_count = df_cp.shape[0]
    
    # Remove outliers
    df_filtered = filter_outliers(df_cp, CONFIG['output_variables'], CONFIG['outlier_iqr_threshold'])
    outlier_count = initial_count - df_filtered.shape[0]
    print(f"Removed {outlier_count} extreme outliers from {initial_count} rows")
    
    # Calculate Cook's distances
    cook_output_viz_dir = os.path.join(CONFIG['output_viz_dir'], country, str(period))
    filtered_count = df_filtered.shape[0]
    cooks_cutoff = 4 / filtered_count
    
    cooks_distances = calculate_all_cooks_distances(
        df_filtered, 
        CONFIG['input_variables'], 
        CONFIG['output_variables']
    )
    
    # Add Cook's distances to DataFrame and plot
    for output_var, distances in cooks_distances.items():
        df_filtered = df_filtered.with_columns(
            pl.Series(f"cooks_{output_var}", distances).cast(pl.Float32)
        )
        # plot_cooks_distance(
        #     distances, output_var, country, period, cook_output_viz_dir, cooks_cutoff
        # )
    
    # Filter influential points
    for output_var in CONFIG['output_variables']:
        df_filtered = df_filtered.filter(pl.col(f"cooks_{output_var}") <= cooks_cutoff)
    
    print(f"Removed {filtered_count - df_filtered.shape[0]} influential points")
    
    # Fit final models and collect results
    results = []
    for output_var in CONFIG['output_variables']:
        X = df_filtered.select(CONFIG['input_variables']).to_pandas()
        X = sm.add_constant(X)
        y = df_filtered.select(output_var).to_pandas()
        
        model = sm.OLS(y, X).fit()
        
        result = {
            'country': country,
            'period': period,
            'output_variable': output_var,
            'n_rows': df_filtered.shape[0],
            'r_squared': model.rsquared,
            'prob_f_stat': 0.0 if model.f_pvalue <= 1e-4 else model.f_pvalue,
            'intercept_coef': model.params["const"],
        }
        
        for i, input_var in enumerate(CONFIG['input_variables']):
            idx = i + 1  # +1 to account for constant
            result[f"{input_var}_coef"] = model.params.iloc[idx]
            result[f"{input_var}_pvalue"] = 0.0 if model.pvalues.iloc[idx] <= 1e-4 else model.pvalues.iloc[idx]
        
        results.append(result)
    
    return pl.DataFrame(results)

def main():
    # Load and prepare data
    df = load_data()
    
    # Process each country-period combination
    result_dfs = []
    for country, period in get_unique_combinations(df):
        result_df = process_country_period(df, country, period)
        result_dfs.append(result_df)
    
    # Combine results
    final_df = pl.concat(result_dfs, how="vertical")
    final_df.sort(['country', 'period', 'output_variable']).write_csv(os.path.join(CONFIG['output_sum_dir'], "regression.csv"))
    print(final_df)
    
    return final_df

if __name__ == "__main__":
    main()