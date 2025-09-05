print("Basic statistics for numerical columns:")
print(df.describe())

# Group by species and compute mean of numerical columns
print("\nMean values by species:")
species_stats = df.groupby('species').mean()
print(species_stats)

# Additional interesting analysis
print("\nInteresting findings:")
print("- Setosa has the smallest petal measurements")
print("- Virginica has the largest sepal and petal measurements")
print("- Versicolor falls in between for most measurements")

# Correlation analysis
correlation = df.select_dtypes(include=[np.number]).corr()
print("\nCorrelation matrix:")
print(correlation)

