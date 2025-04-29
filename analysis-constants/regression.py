import polars as pl
import statsmodels.api as sm
import numpy as np
import warnings

# Suppress potential statsmodels warnings if desired
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
input_file = './data/constants.csv'
output_file = './analysis-constants/regression_results.csv'

group_cols = ['country', 'period']
independent_vars = [
    'Interest Rate',
    'Vat Rate',
    'Corporate Tax',
    'Government Expenditure',
    'Import Tariff'
]
dependent_vars = [
    'Real GDP Growth',
    'Inflation',
    'Unemployment',
    'Budget Balance',
    'Approval Index'
]

# --- Data Loading ---
print(f"Loading data from {input_file}...")
df = pl.read_csv(input_file)

# --- Data Preparation and Regression ---
print("Starting regressions for each country/period group...")
results_list = []

# Group by country and period
# Use .iter_groups() for efficient iteration over subsets
grouped = df.group_by(group_cols)

for (country, period), group_df in grouped:
    print(f"Processing group: country={country}, period={period}")

    # Get independent and dependent variable data for this group
    try:
        X_group_df = group_df.select(independent_vars)
        Y_group_df = group_df.select(dependent_vars)

        # Get sample size for this group
        nobs = len(group_df)

        # Convert independent variables to NumPy array and add constant for intercept
        X_group_np = X_group_df.to_numpy()
        X_group_np_with_const = sm.add_constant(X_group_np, has_constant='add') # Add a constant column

        # Names for the columns in X_group_np_with_const
        param_names = ['Intercept'] + independent_vars

        # Perform regression for each dependent variable
        for dep_var in dependent_vars:
            y_group_np = Y_group_df.select(dep_var).to_numpy().flatten()

            # Ensure no NaN/Inf values that would cause issues
            if np.any(~np.isfinite(y_group_np)) or np.any(~np.isfinite(X_group_np_with_const)):
                 print(f"  Skipping regression for {dep_var} in {country}-{period} due to non-finite values.")
                 continue

            try:
                # Fit the OLS model
                model = sm.OLS(y_group_np, X_group_np_with_const)
                results = model.fit()

                # --- Extract and structure results for this regression ---

                # Parameters (Intercept and Coefficients)
                for i, param_name in enumerate(param_names):
                    results_list.append({
                        'country': country,
                        'period': period,
                        'Dependent Variable': dep_var,
                        'Independent Variables': param_name,
                        'Result': results.params[i],
                        'Significance Value': results.pvalues[i],
                        'Sample Size': nobs
                    })

                # R-squared
                results_list.append({
                    'country': country,
                    'period': period,
                    'Dependent Variable': dep_var,
                    'Independent Variables': 'R-squared',
                    'Result': results.rsquared,
                    'Significance Value': results.f_pvalue, # F-test p-value for overall model significance
                    'Sample Size': nobs
                })

            except Exception as e:
                print(f"  Error performing regression for {dep_var} in {country}-{period}: {e}")
                # Optionally add a row indicating failure
                results_list.append({
                     'country': country,
                     'period': period,
                     'Dependent Variable': dep_var,
                     'Independent Variables': 'Error',
                     'Result': np.nan,
                     'Significance Value': np.nan,
                     'Sample Size': nobs # Still include sample size
                })


    except Exception as e:
        print(f"Error processing data for group {country}-{period}: {e}")
        # If data extraction fails, skip the group or add error indicator
        # For simplicity, the inner loop handles errors per dependent variable


# --- Consolidate and Export Results ---
print("\nRegressions complete. Consolidating results...")

if results_list:
    # Create Polars DataFrame from the list of dictionaries
    results_df = pl.DataFrame(results_list)

    # Define schema explicitly for clarity and correctness
    results_df = results_df.with_columns([
        pl.col('country').cast(pl.Utf8),
        pl.col('period').cast(pl.Int64), # Or Int32 depending on range
        pl.col('Dependent Variable').cast(pl.Utf8),
        pl.col('Independent Variables').cast(pl.Utf8),
        pl.col('Result').cast(pl.Float64),
        pl.col('Significance Value').cast(pl.Float64),
        pl.col('Sample Size').cast(pl.Int64) # Or Int32
    ])

    # Write to CSV
    print(f"Exporting results to {output_file}...")
    try:
        results_df.write_csv(output_file)
        print("Results exported successfully.")
    except Exception as e:
        print(f"Error exporting results: {e}")
else:
    print("No results generated (possibly due to errors). No output file created.")