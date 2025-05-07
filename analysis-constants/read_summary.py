import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
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
    'Import Tariff',
]

input_var_coef = [var + "_coef" for var in input_var]

df = pl.read_parquet("analysis-constants/regression_filter_interest.parquet").with_columns(
    pl.when(pl.col("r_squared") < 0).then(0).otherwise(pl.col("r_squared")).alias("r_squared"),
)

print(df)

col_to_unpivot = df.select(pl.exclude("country", "period", "output_variable")).columns
df_long = df.unpivot(
    col_to_unpivot,
    index=['country', 'period', 'output_variable'],
    )

unique_countries = df_long.select(pl.col("country")).unique().to_series()

for country in unique_countries:
    
    df_filter = df_long.filter(
        pl.col('country') == country,
        pl.col('variable') == 'r_squared',
    ).sort('output_variable')
    
    print(f"Done filtering {country} for r_squared")
    
    fig, ax = plt.subplots(figsize=(7.5, 5))
   
    sns.lineplot(
        data=df_filter,
        x='period',
        y='value',
        hue='output_variable',
        ax=ax,
        palette='icefire',
    )
    
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend(
        handles=handles,
        labels=labels,
        loc='best'
    )
    
    plt.title(f"R-squared values for {country}")
    plt.xlabel("Period")
    plt.ylabel("R-squared")
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    output_dir = "./analysis-constants/visualization/regression/interest-filter/r-squared/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, f"{country}.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Done plotting {country} for r_squared")

for country in unique_countries:
    for var in output_var:
        df_filter = df_long.filter(
            pl.col('country') == country,
            pl.col('output_variable') == var,
            pl.col('variable').is_in(input_var_coef)
        ).sort('period')
        
        print(f"Done filtering {country} for coef of {var}")
        
        fig, ax = plt.subplots(figsize=(8.5, 5))
        
        sns.barplot(
            data=df_filter,
            x='period',
            y='value',
            hue='variable',
            palette='icefire',
        )
        
        handles, labels = ax.get_legend_handles_labels()
        
        ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=(1.02, 0.5),
            loc='center left',
        )
        
        plt.title(f"Coefficient for regression on {var} of {country}")
        plt.xlabel("Period")
        plt.ylabel(var)
        plt.grid(linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        output_dir = f"./analysis-constants/visualization/regression/interest-filter/coef/{var}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(os.path.join(output_dir, f"{country}.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Done plotting coef of {var} for {country}")