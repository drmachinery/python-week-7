# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis Visualizations', fontsize=16, fontweight='bold')

# 1. Line chart (simulating trends - since Iris doesn't have time data, we'll use index)
axes[0, 0].plot(df.index[:50], df['sepal length (cm)'][:50], label='Sepal Length', marker='o')
axes[0, 0].plot(df.index[:50], df['petal length (cm)'][:50], label='Petal Length', marker='s')
axes[0, 0].set_title('Trend of Sepal and Petal Length (First 50 Samples)')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Length (cm)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart - average measurements by species
species_means = df.groupby('species').mean()
measurements = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x_pos = np.arange(len(species_means.index))
width = 0.2

for i, measurement in enumerate(measurements):
    axes[0, 1].bar(x_pos + i*width, species_means[measurement], width, label=measurement)

axes[0, 1].set_title('Average Measurements by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Measurement (cm)')
axes[0, 1].set_xticks(x_pos + width*1.5)
axes[0, 1].set_xticklabels(species_means.index)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Histogram - distribution of sepal length
axes[1, 0].hist(df['sepal length (cm)'], bins=15, alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# 4. Scatter plot - sepal length vs petal length
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    axes[1, 1].scatter(species_data['sepal length (cm)'], 
                      species_data['petal length (cm)'], 
                      label=species, 
                      alpha=0.7)

axes[1, 1].set_title('Sepal Length vs Petal Length by Species')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional visualization using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

