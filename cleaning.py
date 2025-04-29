import pandas as pd
import os
import glob

# --- Configuration ---
# Make sure this path is correct relative to where you run the script
# If your script is in the same directory as the 'Constants' folder, use './Constants'
# If 'Constants' is a level above your script, use '../Constants'
root_directory = './Random'
output_filename = 'random.csv'

# --- Script ---

print(f"Starting data consolidation from {root_directory}")
print(f"Checking if root directory exists: {os.path.isdir(root_directory)}") # Debug print

all_dataframes = []

# Get all entries in the root directory
try:
    entries = os.listdir(root_directory)
    print(f"Entries found in root directory: {entries}") # Debug print
except FileNotFoundError:
    print(f"Error: Root directory not found at {root_directory}")
    entries = [] # Empty list to prevent further errors

# Iterate through entries to find country folders
for entry_name in entries:
    entry_path = os.path.join(root_directory, entry_name)

    # Check if it's a directory (potential country folder)
    if os.path.isdir(entry_path):
        country_name = entry_name

        print(f"\nProcessing country: {country_name}")
        print(f"  Checking if country directory exists: {os.path.isdir(entry_path)}") # Debug print

        # List all files in the country directory and filter for CSVs
        try:
            all_files_in_country_dir = [os.path.join(entry_path, f) for f in os.listdir(entry_path) if os.path.isfile(os.path.join(entry_path, f)) and f.endswith('.csv')]
        except Exception as e:
             print(f"  Error listing files in {entry_path}: {e}")
             all_files_in_country_dir = []

        # --- CRITICAL STEP: Sort the files ---
        # This ensures period 1 is processed before period 2, etc.
        all_files_in_country_dir.sort()

        print(f"  Found and sorted CSV files: {all_files_in_country_dir}") # Debug print

        if not all_files_in_country_dir:
            print(f"  No CSV files found in {entry_path}")
            continue # Move to the next country folder

        # Initialize period counter for this country
        period_counter = 1

        # Process each file in the sorted order
        for filepath in all_files_in_country_dir:
            try:
                filename = os.path.basename(filepath)
                print(f"  Reading file: {filename} (assigned period: {period_counter})")

                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(filepath)

                # Add the new 'country' and 'period' columns
                df['country'] = country_name
                df['period'] = period_counter # Use the counter as the period number

                # Reorder columns to place 'country' and 'period' at the start
                original_columns = df.columns.tolist()
                 # Safely remove 'country' and 'period' if they exist
                if 'country' in original_columns: original_columns.remove('country')
                if 'period' in original_columns: original_columns.remove('period')

                new_column_order = ['country', 'period'] + original_columns
                df = df[new_column_order]

                # Add the processed DataFrame to our list
                all_dataframes.append(df)
                print(f"  Successfully processed {filename}")

                # Increment the period counter for the next file
                period_counter += 1

            except Exception as e:
                print(f"  Error processing file {filepath}: {e}")
                # Continue processing other files even if one fails

# --- Combine and Save ---

print("\nCombining all dataframes...")

if all_dataframes:
    # Concatenate all dataframes in the list
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    print(f"Total rows combined: {len(combined_df)}")

    # Write the combined dataframe to the output CSV file
    combined_df.to_csv(output_filename, index=False)

    print(f"Successfully created {output_filename}")

else:
    print("No dataframes were processed. Output file will not be created.")