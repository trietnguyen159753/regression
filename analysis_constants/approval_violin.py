#!/usr/bin/env python3
"""
Country Approval Index Visualization Tool with Violin Plots

This script generates violin plots showing the distribution of approval index values
across different countries, using the most recent data available for each country.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
ROOT_DIR = 'Constants'
APPROVAL_INDEX_COLUMN_NAME = 'Approval Index'
OUTPUT_FILE = None  # Set to a filename (e.g., 'approval_violins.png') to save the plot


def collect_country_data(country_name, country_path):
    """
    Collect approval index data from the most recent period for a given country.
    
    Args:
        country_name (str): Name of the country
        country_path (str): Path to the country directory
        
    Returns:
        pandas.DataFrame or None: DataFrame with approval index data and country name, 
                                 or None if no valid data
    """
    try:
        # Get all CSV files and sort them
        country_files = sorted([
            f for f in os.listdir(country_path) 
            if f.endswith('.csv') and os.path.isfile(os.path.join(country_path, f))
        ])
        
        if not country_files:
            print(f"  No CSV files found for {country_name}")
            return None
            
        # Get the most recent file
        last_file_name = country_files[-1]
        last_file_path = os.path.join(country_path, last_file_name)
        print(f"  Processing most recent data: {last_file_name}")
        
        # Read the CSV and extract approval index data
        df = pd.read_csv(last_file_path)
        
        if APPROVAL_INDEX_COLUMN_NAME not in df.columns:
            print(f"  Error: '{APPROVAL_INDEX_COLUMN_NAME}' column not found in {last_file_name}")
            return None
            
        approval_series = df[APPROVAL_INDEX_COLUMN_NAME].dropna()
        
        if approval_series.empty:
            print(f"  No valid approval data for {country_name}")
            return None
            
        # Create a DataFrame with approval data and country name
        return pd.DataFrame({
            APPROVAL_INDEX_COLUMN_NAME: approval_series,
            'country': country_name
        })
        
    except Exception as e:
        print(f"  Error processing {country_name}: {str(e)}")
        return None


def create_violin_plots(combined_df):
    """
    Create and display vertical violin plots of approval index by country.
    
    Args:
        combined_df (pandas.DataFrame): DataFrame with approval index data and country names
    """
    # Count unique countries
    countries = sorted(combined_df['country'].unique())
    num_countries = len(countries)
    
    # Create a figure with appropriate size (wider for more countries)
    plt.figure(figsize=(10, 6))
    
    # Create the single violin plot with all countries
    ax = sns.violinplot(
        data=combined_df,
        x='country',                 # Countries on x-axis
        y=APPROVAL_INDEX_COLUMN_NAME,  # Approval index on y-axis
        palette='muted',             # Use a nice color palette
        inner='box',                 # Show box plot inside violin
        cut=0                        # Don't extend violin past data range
    )
    
    # Calculate mean values for each country
    country_means = combined_df.groupby('country')[APPROVAL_INDEX_COLUMN_NAME].mean()
    
    # Add mean markers to the violin plots
    for i, country in enumerate(countries):
        mean_val = country_means[country]
        
        # Add mean marker as a point
        plt.scatter(i, mean_val, color='white', s=40, zorder=5, 
                    marker='o', edgecolor='black', linewidth=1)
        
        # Add text annotation for mean value
        plt.annotate(f'{mean_val:.1f}', 
                    xy=(i, mean_val),
                    xytext=(10, 0),  # 10 points vertical offset
                    textcoords='offset points',
                    ha='left',
                    va='center', 
                    fontsize=10)
    
    # Set y-axis range and ticks with step of 25
    y_min = 0
    y_max = 100
    y_ticks = np.arange(y_min, y_max + 25, 25)
    plt.ylim(y_min, y_max)
    plt.yticks(y_ticks)
    
    # Add horizontal grid lines at the y-tick positions
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add title and labels
    plt.title(f'Distribution of {APPROVAL_INDEX_COLUMN_NAME} by Country (Latest Period)', 
              fontsize=14, pad=20)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel(f'{APPROVAL_INDEX_COLUMN_NAME}', fontsize=12)
    
    # Rotate x-axis labels if there are many countries
    if num_countries > 4:
        plt.xticks(rotation=45, ha='right')
    
    # Add timestamp and username at bottom right
    current_time = "2025-04-28"  # Using the provided timestamp
    username = "trietnguyen159753"        # Using the provided username
    plt.figtext(0.99, 0.01, f'Generated: {current_time} by {username}', 
                fontsize=8, ha='right', va='bottom', alpha=0.7)
    
    # Add a legend for the mean indicator
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                              markeredgecolor='black', markersize=8, label='Mean')]
    ax.legend(handles=legend_elements, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output file is specified
    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {OUTPUT_FILE}")
    
    # Display plot
    plt.show()


def main():
    """Main function to execute the data collection and visualization process."""
    print(f"Analyzing approval index distributions from the latest period for each country")
    print(f"Data source directory: {ROOT_DIR}")
    print(f"Target column: '{APPROVAL_INDEX_COLUMN_NAME}'\n")
    
    # Get sorted list of country directories
    try:
        country_dirs = sorted([
            d for d in os.listdir(ROOT_DIR) 
            if os.path.isdir(os.path.join(ROOT_DIR, d))
        ])
    except FileNotFoundError:
        print(f"Error: Directory '{ROOT_DIR}' not found.")
        return
    except PermissionError:
        print(f"Error: Permission denied when accessing '{ROOT_DIR}'.")
        return
    
    if not country_dirs:
        print(f"No country directories found in '{ROOT_DIR}'.")
        return
    
    # Collect data from each country
    country_dataframes = []
    
    for country_name in country_dirs:
        country_path = os.path.join(ROOT_DIR, country_name)
        print(f"Processing country: {country_name}")
        
        country_df = collect_country_data(country_name, country_path)
        if country_df is not None:
            country_dataframes.append(country_df)
    
    print("\nData collection complete.")
    
    # Generate violin plots if data is available
    if country_dataframes:
        print(f"Creating violin plots for {len(country_dataframes)} countries...")
        combined_df = pd.concat(country_dataframes, ignore_index=True)
        
        if combined_df.empty:
            print("Error: No valid data points collected for plotting.")
        else:
            create_violin_plots(combined_df)
    else:
        print(f"No valid '{APPROVAL_INDEX_COLUMN_NAME}' data found. Violin plots will not be generated.")


if __name__ == "__main__":
    main()