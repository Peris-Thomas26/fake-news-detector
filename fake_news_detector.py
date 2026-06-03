import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
df = pd.read_csv('fake_news_large.csv')

print("=" * 60)
print("BUILDING FAKE NEWS DETECTOR")
print("=" * 60)

# Combine title and text
df['combined_text'] = df['title'] + " " + df['text']

# Split into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(
    df['combined_text'], 
    df['label'], 
    test_size=0.3, 
    random_state=42
)

print(f"\nTraining articles: {len(X_train)}")
print(f"Testing articles: {len(X_test)}")

# Convert text to numbers
print("\n" + "=" * 60)
print("CONVERTING TEXT TO NUMBERS")
print("=" * 60)

vectorizer = TfidfVectorizer(max_features=100)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✓ Text converted to numerical features")

# Train the model
print("\n" + "=" * 60)
print("TRAINING THE MODEL")
print("=" * 60)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("✓ Model trained!")

# Test the model
print("\n" + "=" * 60)
print("TESTING THE MODEL")
print("=" * 60)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.1f}%")
print(f"Out of {len(y_test)} test articles, {sum(y_pred == y_test)} were correct")

# Try on new article
print("\n" + "=" * 60)
print("TEST ON NEW ARTICLE")
print("=" * 60)

new_article = "New study shows eating chocolate daily cures cancer"
new_article_vec = vectorizer.transform([new_article])
prediction = model.predict(new_article_vec)[0]
confidence = model.predict_proba(new_article_vec)[0]

print(f"\nArticle: {new_article}")
print(f"Prediction: {prediction}")
print(f"Confidence: {max(confidence) * 100:.1f}%")

print("\n" + "=" * 60)
