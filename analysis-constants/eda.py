import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import os
import numpy as np

def load_and_clean_data(file_path):
    """Load CSV data and remove nulls and infinite values."""
    df = pl.read_csv(file_path)
    original_count = df.shape[0]
    
    # Remove nulls and infinites
    df_clean = df.drop_nulls().filter(~pl.any_horizontal(cs.numeric().is_infinite()))
    
    return df_clean, original_count

def calculate_outlier_bounds(df, variable_columns):
    """Calculate outlier boundaries for each variable."""
    outlier_bounds = {}
    for var in variable_columns:
        q1 = df[var].quantile(0.25)
        q3 = df[var].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_bounds[var] = (lower_bound, upper_bound)
    
    return outlier_bounds

def filter_outliers(df, outlier_bounds):
    """Filter out outliers based on calculated bounds."""
    outlier_conditions = [
        # Removed the include_bounds parameter which is not supported
        (pl.col(var).is_between(bounds[0], bounds[1]))
        for var, bounds in outlier_bounds.items()
    ]
    
    combined_condition = outlier_conditions[0]
    for cond in outlier_conditions[1:]:
        combined_condition = combined_condition & cond
    
    return df.filter(combined_condition)

def filter_outliers_by_period(df, variable_columns, period_col):
    """Filter outliers with period-specific bounds."""
    periods = df[period_col].unique().sort().to_list()
    filtered_dfs = []
    
    removed_counts = {}  # Track removed rows per period
    
    for period in periods:
        # Get data for this period
        df_period = df.filter(pl.col(period_col) == period)
        period_count_before = df_period.shape[0]
        
        # Calculate bounds specific to this period
        period_bounds = calculate_outlier_bounds(df_period, variable_columns)
        
        # Filter using period-specific bounds
        df_period_filtered = filter_outliers(df_period, period_bounds)
        period_count_after = df_period_filtered.shape[0]
        
        # Track removed rows
        removed_counts[period] = period_count_before - period_count_after
        
        # Add to our collection
        filtered_dfs.append(df_period_filtered)
    
    # Combine all filtered periods
    df_filtered = pl.concat(filtered_dfs)
    
    return df_filtered, removed_counts

def create_and_save_boxplot(country_data, country, variable, periods, output_dir, removed_count, original_count):
    """Create and save a boxplot for a specific country and variable."""
    data_to_plot = []
    sample_sizes = []
    for period in periods:
        period_data = country_data.filter(pl.col(PERIOD_COL) == period)[variable].to_numpy()
        data_to_plot.append(period_data)
        sample_sizes.append(len(period_data))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data_to_plot)
    
    # Find current y-axis limits to position annotations
    current_ymin, current_ymax = ax.get_ylim()
    
    ax.set_title(f'Country {country} - {variable} Distribution by Period')
    ax.set_xlabel('Period')
    ax.set_ylabel(variable)
    ax.set_xticks(range(1, len(periods) + 1))
    ax.set_xticklabels(periods)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Get the final y-axis limits after all elements have been added
    ax.set_ylim(current_ymin, current_ymax + (current_ymax - current_ymin) * 0.1)  # Add 10% padding at top
    final_ymin, final_ymax = ax.get_ylim()
    
    # Add sample size annotations near the top of plot but within visible area
    for i, size in enumerate(sample_sizes, 1):
        ax.text(i, final_ymax - (final_ymax - final_ymin) * 0.05,  # Position 5% below the top
                f'n={size}',
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    plt.tight_layout()
    
    filename = f'{country}_{variable}.png'
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def generate_boxplots(file_path, output_dir, variable_columns, country_col, period_col):
    """Main function to generate boxplots for all countries and variables."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    df, original_count = load_and_clean_data(file_path)

    outlier_bounds = calculate_outlier_bounds(df, variable_columns)
    
    # Filter outliers using period-specific bounds
    df_filtered, removed_counts = filter_outliers_by_period(df, variable_columns, period_col)
    
    # Get unique countries and periods
    countries = df_filtered[country_col].unique().sort().to_list()
    periods = df_filtered[period_col].unique().sort().to_list()
    
    # Generate plots for each country and variable
    for country in countries:
        df_country = df_filtered.filter(pl.col(country_col) == country)
        
        for variable in variable_columns:
            create_and_save_boxplot(
                df_country, 
                country, 
                variable, 
                periods, 
                output_dir, 
                original_count
            )

# Constants
FILE_PATH = './data/constants.csv'
OUTPUT_DIR = './visualization/boxplots'

VARIABLE_COLUMNS = [
    'Real GDP Growth',
    'Inflation',
    'Unemployment',
    'Budget Balance',
    'Approval Index',
]

COUNTRY_COL = 'country'
PERIOD_COL = 'period'

# Run the program
if __name__ == "__main__":
    generate_boxplots(FILE_PATH, OUTPUT_DIR, VARIABLE_COLUMNS, COUNTRY_COL, PERIOD_COL)