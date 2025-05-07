import polars as pl
import os 
import time 

start = time.time()

output_var = [
    'Real GDP Growth',
    'Inflation',
    'Unemployment',
    'Budget Balance',
]

input_var = [
    'Interest Rate',
    'Vat Rate',
    'Corporate Tax',
    'Government Expenditure',
    'Import Tariff',
]

rootdir = './raw-data/'

main_subfolder = ["Constants", "Random"]

final_df = pl.DataFrame()

for subfolder in main_subfolder:
    main_subfolder_path = os.path.join(rootdir, subfolder)
    lazy_frames = []
    
    for country in os.listdir(main_subfolder_path):
        country_path = os.path.join(main_subfolder_path, country)
        index = 0
        
        for csv in os.listdir(country_path):
            index += 1
           
            lazy_frame = pl.scan_csv(
                os.path.join(country_path, csv), 
                schema_overrides={'Approval Index' : pl.Float32}
            ).with_columns(
                pl.lit(country).alias("country"),
                pl.lit(index).alias("period"),
            ).select(
                pl.col("country"),
                pl.col("period"),
                pl.all().exclude("country", "period"),   
            )
            
            lazy_frames.append(lazy_frame)
            
    final_df = pl.concat(lazy_frames, how="vertical").collect()
    final_df.write_parquet(f"{subfolder}_prelim.parquet")
    print(f"Finished {subfolder}")
    
end = time.time()

print(f"Time elapsed: {end - start} seconds")