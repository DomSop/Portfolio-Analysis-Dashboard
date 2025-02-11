import numpy as np
import pandas as pd
import nltk
import re
import dash
import plotly.express as px

from dash import dcc, html
from dash.dependencies import Input, Output
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Generate synthetic financial sentiment data
np.random.seed(42)  # Set seed for reproducibility

# Example financial-related sentences with sentiment labels
sentences = [
    "The stock market is booming, investors are optimistic!",
    "Severe losses in the market today, risk is high.",
    "Stable performance from most companies in Q3.",
    "Tech stocks rally as confidence grows in AI innovations.",
    "Economic downturn leads to declining investor confidence."
]
labels = ["positive", "negative", "neutral", "positive", "negative"]

# DataFrame
df = pd.DataFrame({"Text": sentences, "Sentiment": labels})

# Step 2: Text cleaning and preprocessing
def clean_text(text):
    """Clean the input text by removing punctuation, lowercasing, 
    and removing stopwords."""
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)

# Apply text cleaning
df["Clean_Text"] = df["Text"].apply(clean_text)

# Step 3: Sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()

# Compute sentiment score using VADER
df["VADER_Score"] = df["Text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

# Step 4: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Clean_Text"])  # Convert text into numerical features

# Convert sentiment labels into numerical values
y = df["Sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})

# Step 5: Train a Naïve Bayes sentiment classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Create an interactive dashboard with Dash
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Financial Sentiment Analysis"),  # Title
    dcc.Dropdown(
        id="sentence",
        options=[{"label": txt, "value": txt} for txt in df["Text"]],
        value=df["Text"][0]  # Default selection
    ),
    html.Div(id="sentiment-output"),  # Display sentiment prediction
    dcc.Graph(id="sentiment-graph")  # Sentiment score visualization
])

# Define the callback function to update the dashboard dynamically
@app.callback(
    [Output("sentiment-output", "children"), Output("sentiment-graph", "figure")],
    Input("sentence", "value")
)
def update_dashboard(sentence):
    """Function to analyze the sentiment of the selected sentence and 
    update the dashboard visualization."""
    
  # Compute the VADER sentiment score
    
  sentiment_score = analyzer.polarity_scores(sentence)["compound"]
    
   # Determine sentiment category based on score
   sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

  # Create a bar chart to visualize sentiment
   fig = px.bar(
        x=["Negative", "Neutral", "Positive"],
        y=[-1, 0, 1],  # Simulated scores for visualization
        color=["red", "gray", "green"],
        labels={"x": "Sentiment", "y": "Score"},
        title="Sentiment Score"
    )
   return f"Predicted Sentiment: {sentiment_label} (Score: {sentiment_score})", fig

# Run the dashboard
if __name__ == "__main__":
    app.run_server(debug=True)
