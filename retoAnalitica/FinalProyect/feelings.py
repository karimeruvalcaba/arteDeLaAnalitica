import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load your CSV file into a DataFrame
# Replace 'covid19_tweets.csv' with the path to your actual file
df = pd.read_csv('covid19_tweets.csv')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def classify_sentiment(text):
    if not isinstance(text, str):
        return 'neutral'
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Filter for tweets that mention "Trump"
mask_tweets = df[df['text'].str.contains('Wearamask', case=False, na=False)]

# Apply the sentiment classification function
mask_tweets['sentiment'] = mask_tweets['text'].apply(classify_sentiment)

# Count positive and negative tweets
positive_count = mask_tweets[mask_tweets['sentiment'] == 'positive'].shape[0]
negative_count = mask_tweets[mask_tweets['sentiment'] == 'negative'].shape[0]

# Print the counts
print(f'Positivo tweets: {positive_count}')
print(f'Negativo tweets: {negative_count}')

# Optional: Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.bar(['Positive', 'Negative'], [positive_count, negative_count], color=['green', 'red'])
plt.ylabel('Number of Tweets')
plt.title('Sentimientos acerca de usar cubrebocas')
plt.show()
