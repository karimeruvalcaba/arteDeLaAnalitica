import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("avocado.csv")

# Define a list of regions to filter
regions = ["Charlotte", "California", "NewYork"]

# Filter the DataFrame to only include rows with the specified regions
filtered_df = df[df["region"].isin(regions)]

# Group by region and apply aggregation functions to the relevant columns
agg_stats = filtered_df.groupby("region").agg({
    'Total Bags': ['sum', 'mean', 'std'],
    'Small Bags': 'sum',
    'Large Bags': 'sum',
    'XLarge Bags': 'sum'
})

# Rename the aggregated columns
agg_stats.columns = [
    'produccion total', 'promedio', 'desviacion estandar',
    'total_small_bags', 'total_large_bags', 'total_xlarge_bags'
]

# Calculate the percentages of each bag type over the total
agg_stats['por small_bags'] = (agg_stats['total_small_bags'] / agg_stats['produccion total']) * 100
agg_stats['por large_bags'] = (agg_stats['total_large_bags'] / agg_stats['produccion total']) * 100
agg_stats['por xlarge_bags'] = (agg_stats['total_xlarge_bags'] / agg_stats['produccion total']) * 100

# Print the aggregated statistics including percentages
print(agg_stats)




#total, promedio, desviacion estandar, totall bags es produccion total