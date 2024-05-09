import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your CSV file into a DataFrame
df = pd.read_csv("covid19_tweets.csv")

# List of numerical columns for individual boxplots
numerical_columns = ["user_followers", "user_friends", "user_favourites"]

# Loop through each column and create a separate boxplot
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, y=column)
    plt.title(f"Boxplot of {column} from COVID-19 Tweets Dataset")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Histograms
sns.histplot(df['user_followers'], bins=20, kde=False, color='blue')
plt.xlabel('Number of Followers')
plt.ylabel('Frequency')
plt.title('Histogram of User Followers')
plt.show()

sns.histplot(df['user_friends'], bins=20, kde=False, color='green')
plt.xlabel('Number of Friends')
plt.ylabel('Frequency')
plt.title('Histogram of User Friends')
plt.show()

sns.histplot(df['user_favourites'], bins=20, kde=False, color='orange')
plt.xlabel('Number of Favourites')
plt.ylabel('Frequency')
plt.title('Histogram of User Favourites')
plt.show()

# Mapa de Calor
numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# K-means clustering
# Standardize the data
scaler = StandardScaler()
numeric_data = numeric_df[numerical_columns]
scaled_data = scaler.fit_transform(numeric_data)

# Define and train the K-means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Add the cluster labels back to the original DataFrame
df['Cluster'] = kmeans.labels_

# Visualize clusters using pairplot
sns.pairplot(df, vars=numerical_columns, hue='Cluster', palette='tab10')
plt.suptitle("K-means Clusters in COVID-19 Tweets Dataset", y=1.02)
plt.show()
