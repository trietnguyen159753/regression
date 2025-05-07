import polars as pl 
import polars.selectors as ps
import seaborn as sns
import matplotlib.pyplot as plt
import os

output_var = [
    'Real GDP Growth',
    'Inflation',
    'Unemployment',
    'Budget Balance',
    'Approval Index',
]

input_var = [
    'Interest Rate',
    'Vat Rate',
    'Corporate Tax',
    'Government Expenditure',
    'Import Tariff',
]

output_root_dir = './analysis-constants/visualization/'

df = pl.read_parquet('./Constants_prelim.parquet').filter(
    ~pl.any_horizontal(ps.numeric().is_infinite())
    )
print("Done reading parquet")

SAMPLE_FRACTION = 0.05

# for period in range(1, 9):
#     for var in output_var:
#         df_sample = df.filter(pl.col('period') == period).select(
#             pl.col('country'),
#             pl.col('period'),
#             pl.col(var),
#         ).sample(fraction=SAMPLE_FRACTION).with_columns(
#             pl.col('country').cast(pl.Utf8),
#             pl.col('period').cast(pl.UInt8),
#             pl.col(var).cast(pl.Float32),
#         ).sort('country')
        
#         print(f"Done filtering {var} for period {period}")

#         fig, ax = plt.subplots(figsize=(8, 4.5))
#         ax = sns.violinplot(
#             data=df_sample,
#             x='country',
#             y=f"{var}",
#             palette='BrBG',
#             hue='country',
#             inner='box',
#             cut=0,
#         )
        
#         plt.title(f'{var} - Period {period} (Sampled)') # Add title for clarity
#         plt.tight_layout() 
#         if var == 'Approval Index':
#             plt.ylim(0, 100)
#             plt.yticks([0, 25, 50, 75, 100])
#             plt.grid(axis='y', linestyle='--', alpha=0.4)
            
#         print(f"Done plotting {var} for period {period}")
        
#         output_dir = os.path.join(output_root_dir, f'distribution/unfiltered/period_slice/{var}/')
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         plt.savefig(output_dir + f'{period}.png', dpi=100, bbox_inches='tight')
    
    
# unique_country = df.select('country').unique().to_series()

# for country in unique_country:
#     for var in output_var:
#         df_sample = df.filter(pl.col('country') == country).select(
#             pl.col('country'),
#             pl.col('period'),
#             pl.col(var),
#         ).sample(fraction=SAMPLE_FRACTION).with_columns(
#             pl.col('country').cast(pl.Utf8),
#             pl.col('period').cast(pl.UInt8),
#             pl.col(var).cast(pl.Float32),
#         ).sort('country')
        
#         print(f"Done filtering {var} for {country}")

#         fig, ax = plt.subplots(figsize=(8, 4.5))
#         ax = sns.violinplot(
#             data=df_sample,
#             x='period',
#             y=f"{var}",
#             palette='BrBG',
#             hue='period',
#             inner='box',
#             cut=0,
#         )
        
#         plt.title(f'{var} - {country} (Sampled)') # Add title for clarity
#         plt.tight_layout() 
#         if var == 'Approval Index':
#             plt.ylim(0, 100)
#             plt.yticks([0, 25, 50, 75, 100])
#             plt.grid(axis='y', linestyle='--', alpha=0.4)
            
#         print(f"Done plotting {var} for {country}")
        
#         output_dir = os.path.join(output_root_dir, f'distribution/unfiltered/country_slice/{var}/')
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
            
#         plt.savefig(output_dir + f'{country}.png', dpi=100, bbox_inches='tight')

for period in range(1, 9):
    for var in output_var:
        df_sample = df.select(
            pl.col('country', 'period', var),
        ).filter(
            pl.col('period') == period,
            (pl.col(var) <= 100) & 
            (pl.col(var) >= -20),
        ).sample(
            fraction=SAMPLE_FRACTION
        ).with_columns(
            pl.col('country').cast(pl.Utf8),
            pl.col('period').cast(pl.UInt8),
            pl.col(var).cast(pl.Float32),
        ).sort('country')
        
        print(f"Done filtering {var} for period {period}")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax = sns.violinplot(
            data=df_sample,
            x='country',
            y=f"{var}",
            palette='BrBG',
            hue='country',
            inner='box',
            cut=0,
        )
        
        plt.title(f'{var} - Period {period} (Sampled)') # Add title for clarity
        plt.tight_layout() 
        if var == 'Approval Index':
            plt.ylim(0, 100)
            plt.yticks([0, 25, 50, 75, 100])
            plt.grid(axis='y', linestyle='--', alpha=0.4)
            
        print(f"Done plotting {var} for period {period}")
        
        output_dir = os.path.join(output_root_dir, f'distribution/filtered/period_slice/{var}/')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f'{period}.png', dpi=100, bbox_inches='tight')
    
    
unique_country = df.select('country').unique().to_series()

for country in unique_country:
    for var in output_var:
        df_sample = df.select(
            pl.col('country', 'period', var),
        ).filter(
            pl.col('period') == period,
            (pl.col(var) <= 100) & 
            (pl.col(var) >= -20),
        ).sample(
            fraction=SAMPLE_FRACTION
        ).with_columns(
            pl.col('country').cast(pl.Utf8),
            pl.col('period').cast(pl.UInt8),
            pl.col(var).cast(pl.Float32),
        ).sort('country')
        
        print(f"Done filtering {var} for {country}")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax = sns.violinplot(
            data=df_sample,
            x='period',
            y=f"{var}",
            palette='BrBG',
            hue='period',
            inner='box',
            cut=0,
        )
        
        plt.title(f'{var} - {country} (Sampled)') # Add title for clarity
        plt.tight_layout() 
        if var == 'Approval Index':
            plt.ylim(0, 100)
            plt.yticks([0, 25, 50, 75, 100])
            plt.grid(axis='y', linestyle='--', alpha=0.4)
            
        print(f"Done plotting {var} for {country}")
        
        output_dir = os.path.join(output_root_dir, f'distribution/filtered/country_slice/{var}/')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_dir + f'{country}.png', dpi=100, bbox_inches='tight')