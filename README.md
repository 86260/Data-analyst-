# Data-analyst-
This is my assignment repository.
<br>
Step 1: Scraper Implementation
Create a scraper to collect data from a website (e.g., news articles, reviews, etc.).
scraper.py:
import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_data(url):
    """Fetches data from a given URL and returns parsed HTML."""
    response = requests.get(url)
    if response.status_code == 200:
        return BeautifulSoup(response.text, 'html.parser')
    else:
        print(f"Failed to fetch data from {url}")
        return None

def extract_articles(soup):
    """Extracts articles or text from the HTML page."""
    articles = []
    for item in soup.find_all('div', class_='article'):  # Update class_ based on actual HTML structure
        title = item.find('h2').text if item.find('h2') else "No Title"
        content = item.find('p').text if item.find('p') else "No Content"
        articles.append({'title': title, 'content': content})
    return articles

def scrape_data(url, output_file='scraped_data.csv'):
    """Scrapes articles from a URL and saves to CSV."""
    soup = fetch_data(url)
    if soup:
        articles = extract_articles(soup)
        df = pd.DataFrame(articles)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print("No data to save.")


Step 2: Data Preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_data(file_path='scraped_data.csv'):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def clean_text(text):
    """Cleans and preprocesses text data."""
    text = text.lower()  # Lowercase
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove special characters
    return text

def preprocess_data(df):
    """Preprocesses data (cleans text, splits into training and test sets)."""
    df['cleaned_content'] = df['content'].apply(clean_text)
    X = df['cleaned_content']
    y = df['title']  # For example, using 'title' as a label (this can be changed as per the use case)
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)
    
    # Encoding the labels (e.g., article categories or other labels)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, vectorizer, label_encoder


Step 3: Model Training
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train):
    """Trains a Naive Bayes model."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints the performance metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    from preprocessing import load_data, preprocess_data
    
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, vectorizer, label_encoder = preprocess_data(df)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)


Step 4: Jupyter Notebooks
# Notebook to demonstrate scraping data

import scraper
url = 'https://example.com/news'
scraper.scrape_data(url, output_file='scraped_data.csv')


Step 5: README Instructions
# Web Scraper and Prediction Model

This project demonstrates how to scrape data from a website, preprocess it, and train a machine learning model to predict outcomes based on the scraped data.

## Requirements

1. Python 3.7+
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Files

- `scraper.py`: Contains the scraper to collect data from the specified URL.
- `preprocessing.py`: Preprocesses the scraped data (cleans text, vectorizes, and encodes labels).
- `model.py`: Contains code to train and evaluate the machine learning model.

## Instructions

### 1. Scrape Data

Run the following command to scrape data from the website:
```bash
python scraper.py
2. Preprocess Data
python preprocessing.py
3. Train and Evaluate the Model
python model.py




