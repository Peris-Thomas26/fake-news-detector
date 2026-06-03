import pandas as pd

# Load the dataset
df = pd.read_csv('fake_news.csv')

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

# How many rows and columns?
print(f"\nTotal articles: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Show first few rows
print("\nFirst 3 articles:")
print(df.head(3))

# How many FAKE vs REAL?
print("\nFAKE vs REAL count:")
print(df['label'].value_counts())

# Any missing data?
print("\nMissing values:")
print(df.isnull().sum())

# Show one example of each
print("\n" + "=" * 50)
print("EXAMPLE FAKE NEWS:")
print("=" * 50)
fake = df[df['label'] == 'FAKE'].iloc[0]
print(f"Title: {fake['title']}")
print(f"Text: {fake['text']}")

print("\n" + "=" * 50)
print("EXAMPLE REAL NEWS:")
print("=" * 50)
real = df[df['label'] == 'REAL'].iloc[0]
print(f"Title: {real['title']}")
print(f"Text: {real['text']}")
