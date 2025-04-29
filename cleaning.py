import pandas as pd
import os
import sys # Import sys for exiting on critical errors

def main():
    """
    Consolidates CSV files from a directory structure ./root_dir/country_name/*.csv
    into a single CSV file. It adds 'country' and 'period' columns, where
    'period' is determined by the alphabetical/numerical sort order of files
    within each country directory.
    """
    # --- Configuration ---
    root_directory = './raw-data/Constants'
    output_filename = './data/constants.csv'
    # --- End Configuration ---

    print(f"Starting data consolidation from directory: {root_directory}")

    all_dataframes = []

    # Check if the root directory exists
    if not os.path.isdir(root_directory):
        print(f"Error: Root directory not found at '{root_directory}'", file=sys.stderr)
        sys.exit(1) # Exit if the main directory is missing

    # Iterate through potential country folders in the root directory
    try:
        entries = os.listdir(root_directory)
    except OSError as e:
        print(f"Error accessing root directory '{root_directory}': {e}", file=sys.stderr)
        sys.exit(1)

    for entry_name in entries:
        entry_path = os.path.join(root_directory, entry_name)

        # Process only if it's a directory
        if os.path.isdir(entry_path):
            country_name = entry_name
            print(f"\nProcessing country: {country_name}")

            # Find, sort, and process CSV files within the country directory
            try:
                # List files, filter for CSVs, and get full paths
                csv_files = [
                    os.path.join(entry_path, f)
                    for f in os.listdir(entry_path)
                    if os.path.isfile(os.path.join(entry_path, f)) and f.lower().endswith('.csv')
                ]
                # Sort files to ensure correct period assignment
                csv_files.sort()
            except OSError as e:
                 print(f"  Error listing files in '{entry_path}': {e}", file=sys.stderr)
                 continue # Skip this country folder on listing error

            if not csv_files:
                print(f"  No CSV files found in '{entry_path}'.")
                continue

            # Process each sorted file, assigning period numbers
            for period_counter, filepath in enumerate(csv_files, start=1):
                filename = os.path.basename(filepath)
                try:
                    print(f"  Reading file: {filename} (assigned period: {period_counter})")
                    df = pd.read_csv(filepath)

                    # Add country and period columns (insert at the beginning)
                    if 'period' not in df.columns:
                        df.insert(0, 'period', period_counter)
                    else:
                        print(f"  Warning: Column 'period' already exists in {filename}. Using existing values for this column.")
                        # Optionally overwrite: df['period'] = period_counter

                    if 'country' not in df.columns:
                         df.insert(0, 'country', country_name)
                    else:
                        print(f"  Warning: Column 'country' already exists in {filename}. Using existing values for this column.")
                         # Optionally overwrite: df['country'] = country_name


                    all_dataframes.append(df)
                    # print(f"  Successfully processed {filename}") # Optional: uncomment for verbose success message

                except pd.errors.EmptyDataError:
                    print(f"  Warning: Skipping empty file {filename}.", file=sys.stderr)
                except Exception as e:
                    # Catch other potential errors during file read/processing
                    print(f"  Error processing file {filename}: {e}", file=sys.stderr)
                    # Continue processing other files even if one fails

    # --- Combine and Save ---
    print("\nCombining all processed dataframes...")

    if all_dataframes:
        try:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            print(f"Total rows combined: {len(combined_df)}")

            # Write the combined dataframe to the output CSV file
            combined_df.to_csv(output_filename, index=False)
            print(f"Successfully created consolidated file: {output_filename}")

        except Exception as e:
            print(f"Error during final concatenation or saving to CSV: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("No dataframes were successfully processed. Output file not created.")

if __name__ == "__main__":
    main()