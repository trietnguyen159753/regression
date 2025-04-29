import pandas as pd
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Set the theme for the plots, similar to the example
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

ROOT_DIR = 'Constants'

APPROVAL_INDEX_COLUMN_NAME = 'Approval Index'

print(f"Scanning directory: {ROOT_DIR}")
print(f"Collecting raw data for '{APPROVAL_INDEX_COLUMN_NAME}' from the last period of each country for plotting.\n")
print(f"Generating ridged density plots for '{APPROVAL_INDEX_COLUMN_NAME}'.")

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

    if not approval_series.empty:
        country_df_for_plot = pd.DataFrame({
            APPROVAL_INDEX_COLUMN_NAME: approval_series,
            'country': country_name
        })
        all_country_dataframes_for_plot.append(country_df_for_plot)

print("\nFinished processing countries.")

if all_country_dataframes_for_plot:
    print("\nGenerating ridged density plots...")
    combined_df_for_plot = pd.concat(all_country_dataframes_for_plot, ignore_index=True)

    if combined_df_for_plot.empty:
        print("No valid data points collected for plotting.")
    else:
        # Initialize the FacetGrid object
        # Create a palette with enough colors for each country
        pal = sns.cubehelix_palette(len(country_dirs), rot=-.25, light=.7)
        g = sns.FacetGrid(combined_df_for_plot, row="country", hue="country",
                          aspect=15, height=.5, palette=pal) # Adjust height/aspect for desired look

        # Draw the densities in a few steps, similar to the example
        # First draw the filled density
        g.map(sns.kdeplot, APPROVAL_INDEX_COLUMN_NAME,
              bw_adjust=.5, clip_on=False, # bw_adjust controls smoothing
              fill=True, alpha=1, linewidth=1.5)

        # Then add a white outline
        g.map(sns.kdeplot, APPROVAL_INDEX_COLUMN_NAME, clip_on=False,
              color="w", lw=2, bw_adjust=.5)

        # Add a horizontal line at y=0 for the baseline of each density
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        # This function is called by g.map for each facet (country)
        def label(x, color, label):
            ax = plt.gca()
            # Place the country name label on the left side of the axes
            ax.text(-0.01, .2, label, fontweight="bold", color=color,
                    ha="right", va="center", transform=ax.transAxes) # ha="right" to align text to the right of -0.01 position

        # Apply the labeling function to each facet
        # Pass the numeric column name, the label function receives the data, color, and the row name ('country')
        g.map(label, APPROVAL_INDEX_COLUMN_NAME)

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25) # Adjust this value to control overlap

        # Remove axes details that don't play well with overlap or are unnecessary
        g.set_titles("") # Remove default titles for each facet row
        g.set(yticks=[], ylabel="") # Remove y-axis ticks and label
        g.despine(bottom=True, left=True) # Remove spines (borders)


        # Add a overall title for the entire figure
        g.fig.suptitle(f'Distribution of {APPROVAL_INDEX_COLUMN_NAME} by Country (Last Period)', y=1.03) # Adjust y to position title above plots
        g.fig.subplots_adjust(top=0.95) # Add some space for the super title


        plt.show()
        print("Ridged density plots displayed.")

else:
    print("\nNo valid data collected for plotting '{APPROVAL_INDEX_COLUMN_NAME}'. Ridged density plots will not be generated.")

if __name__ == "__main__":
    if APPROVAL_INDEX_COLUMN_NAME == 'Your_Approval_INDEX_COLUMN_NAME_Here':
        print("\nWARNING: You have not updated 'APPROVAL_INDEX_COLUMN_NAME'. Please replace 'Your_APPROVAL_INDEX_COLUMN_NAME_Here' with the actual column name.")