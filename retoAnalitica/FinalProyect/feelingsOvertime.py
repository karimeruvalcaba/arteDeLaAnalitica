import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
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

# Ensure the 'date' column is in the correct datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Filter for tweets that mention "Wearamask"
mask_tweets = df[df['text'].str.contains('Wearamask', case=False, na=False)]

# Apply the sentiment classification function
mask_tweets['sentiment'] = mask_tweets['text'].apply(classify_sentiment)

# Filter out neutral sentiments
mask_tweets_filtered = mask_tweets[mask_tweets['sentiment'] != 'neutral']

# Group by date and sentiment and count occurrences
daily_sentiment = mask_tweets_filtered.groupby([mask_tweets_filtered['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)

# Plot only positive and negative sentiments
daily_sentiment[['positive', 'negative']].plot(kind='line', figsize=(10, 6), title='Evolution de los sentimientos positivos y negativos sobre el uso del cubrebocas')
plt.ylabel('Numero de Tweets')
plt.xlabel('Fecha')
plt.show()
