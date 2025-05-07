import polars as pl
import pandas as pd
import statsmodels.api as sm
import os

pl.Config().set_tbl_cols(-1)

output_var = [
    'Real GDP Growth',
    'Inflation',
    'Unemployment',
    'Budget Balance',
    'Approval Index'
]

input_var = [
    'Interest Rate',
    'Vat Rate',
    'Corporate Tax',
    'Government Expenditure',
    'Import Tariff'
]

df = pl.read_parquet("Constants_prelim.parquet").with_columns(
    pl.col(pl.Float64).replace([float('inf'), -float('inf')], None)
).drop_nulls().filter(
    pl.col("Interest Rate") <= 8,
    pl.col(pl.Float64).is_between(-20, 100),
)

unique = df.select(['country', 'period']).unique().rows()

results = []

for country, period in unique:
    for var in output_var:
        
        df_unique = df.filter(
            (pl.col("country") == country) & 
            (pl.col("period") == period)
        )
        
        X = df_unique.select(input_var).to_pandas()
        X = sm.add_constant(X)
        y = df_unique.select(var).to_pandas()
        model = sm.OLS(y, X).fit()
        
        print(f"Done regression for {country} - {period} - {var}")
        
        result = {
            'country': country,
            'period': period,
            'output_variable': var,
            'n_rows': df_unique.shape[0],
            'r_squared': model.rsquared,
            'prob_f_stat': 0.0 if model.f_pvalue <= 1e-4 else model.f_pvalue,
            'intercept_coef': model.params["const"],
        }
        
        for i, var in enumerate(input_var):
            idx = i+1
            result[f"{var}_coef"] = model.params.iloc[idx]
            result[f"{var}_pvalue"] = 0.0 if model.pvalues.iloc[idx] <= 1e-4 else model.pvalues.iloc[idx]

        print(f"Done collecting results for {country} - {period} - {var}")
        
        results.append(result)
        
final_df = pl.DataFrame(results)
final_df.sort("country", "period", "output_variable").write_parquet("analysis-constants/regression_filter_interest.parquet")
