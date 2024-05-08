import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV file into a DataFrame
df = pd.read_csv("covid19_tweets.csv")

# List of numerical columns for individual boxplots
numerical_columns = ["user_followers", "user_friends", "user_favourites"]

#'''
# Loop through each column and create a separate boxplot
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, y=column)
    plt.title(f"Boxplot of {column} from COVID-19 Tweets Dataset")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
#'''
#Histograma
#'''
## Generate a histogram for a specific column, for instance 'user_followers'
sns.histplot(df['user_followers'], bins=20, kde=False, color='blue')
plt.xlabel('Number of Followers')
plt.ylabel('Frequency')
plt.title('Histogram of User Followers')
plt.show()

# Repeat for other columns, e.g., 'user_friends', 'user_favourites'
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
#'''

#Mapa de Calor
'''
# Check which columns are numeric and remove those that aren't needed for the correlation
numeric_df = df.select_dtypes(include='number')

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Plot the heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f")

# Add labels and title
plt.title('Correlation Heatmap')
plt.show()
'''