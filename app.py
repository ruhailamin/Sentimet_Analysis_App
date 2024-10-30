import streamlit as st
from joblib import load
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = load('sentiment_model.sav')
vocab = load('tfidf_vocab.joblib')
vectorizer = TfidfVectorizer(vocabulary=vocab)

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation and special characters, keeping only letters, numbers, and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenization and Lemmatization function


def tokenize_and_lemmatize(text):
    # Tokenize the cleaned text
    tokens = word_tokenize(text)
    # Remove stop words and apply lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(
        token) for token in tokens if token not in stop_words]
    # Join the lemmatized tokens back into a string
    return ' '.join(lemmatized_tokens)


# Streamlit UI setup
st.title("Sentiment Analysis of Tweets")  # Simple title for the app

# Add Twitter logo
# st.image("https://www.freepik.com/free-vector/twitter-app-icon-vector-with-watercolor-graphic-effect-21-july-2021-bangkok-thailand_18246339.htm#fromView=keyword&page=1&position=27&uuid=ec1a79b6-2d20-45e4-9e97-6604b2ad79d8", width=50)  # Adjust width as needed

user_tweet = st.text_area("Enter your tweet:")

if st.button("Analyze Sentiment"):
    if user_tweet:
        # Preprocess the tweet
        processed_tweet = preprocess_text(user_tweet)
        # Tokenize and lemmatize
        processed_tweet = tokenize_and_lemmatize(processed_tweet)
        # Vectorize the processed tweet
        vectorized_tweet = vectorizer.fit_transform(
            [processed_tweet])  # Create a sparse vector

        # Predict sentiment
        sentiment = model.predict(vectorized_tweet)[0]
        sentiment_label = "Positive" if sentiment == 1 else "Negative"
        # Display sentiment result
        st.write(f"The sentiment of this tweet is: **{sentiment_label}**")
    else:
        st.warning("Please enter a tweet to analyze.")
# st.write(processed_tweet)        
