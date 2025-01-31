import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

# Set page title
st.title('Analisis Sentimen Telkomsel')

# Load data
@st.cache_data  # Menggunakan cache untuk performa
def load_data():
    df = pd.read_csv('telkomsel.csv')
    df_tweet = pd.DataFrame(df[['full_text']])
    df_tweet['full_text'] = df_tweet['full_text'].str.lower()
    df_tweet.drop_duplicates(inplace=True)
    return df_tweet

df_tweet = load_data()

# Define sentiment classification function
def classify_sentiment(text):
    positive_phrases = [
        "kembali normal", "sudah lancar", "banyak nih", "cukup dengan", "menarik", "promo", "ganti telkomsel",
        # ... (positive_phrases lainnya)
    ]

    negative_phrases = [
        "tidak suka", "kecewa dengan layanan", "ga bs", "kendala", "belum bisa", "reinstall", "re install",
        # ... (negative_phrases lainnya)
    ]

    positive_words = [
        "baik", "bagus", "puas", "senang", "menyala", "pengguna", "lancar", "meding", "setia", "selamat",
        # ... (positive_words lainnya)
    ]

    negative_words = [
        "buruk", "kecewa", "mengecewakan", "kurang", "diperbaiki", "nggak bisa", "dijebol", "jelek",
        # ... (negative_words lainnya)
    ]

    positive_count = sum(1 for phrase in positive_phrases if phrase in text) + sum(1 for word in positive_words if word in text.split())
    negative_count = sum(1 for phrase in negative_phrases if phrase in text) + sum(1 for word in negative_words if word in text.split())

    if negative_count > positive_count:
        return "negatif"
    elif positive_count > negative_count:
        return "positif"
    else:
        return "netral"

# Apply sentiment analysis
df_tweet['sentiment'] = df_tweet['full_text'].apply(classify_sentiment)

# Create visualization
st.subheader('Distribusi Sentimen')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate sentiment counts
sentiment_counts = df_tweet['sentiment'].value_counts()

# Create bar plot
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,
            hue=sentiment_counts.index,
            palette={'positif': 'green', 'negatif': 'red', 'netral': 'gray'},
            legend=False,
            ax=ax)

# Customize plot
plt.title('Distribusi Sentimen')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah')

# Display plot in Streamlit
st.pyplot(fig)

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tweet Positif", len(df_tweet[df_tweet['sentiment'] == 'positif']))
with col2:
    st.metric("Tweet Negatif", len(df_tweet[df_tweet['sentiment'] == 'negatif']))
with col3:
    st.metric("Tweet Netral", len(df_tweet[df_tweet['sentiment'] == 'netral']))

# Display sample of analyzed tweets
st.subheader('Sampel Hasil Analisis')
st.dataframe(df_tweet.sample(10))
