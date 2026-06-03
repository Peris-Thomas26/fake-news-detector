import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load data
df = pd.read_csv('fake_news_large.csv')
df['combined_text'] = df['title'] + " " + df['text']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['combined_text'], 
    df['label'], 
    test_size=0.3, 
    random_state=42
)

# Train model
vectorizer = TfidfVectorizer(max_features=100)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Data distribution
ax1 = axes[0, 0]
counts = df['label'].value_counts()
counts.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('Dataset Distribution\n(Real vs Fake)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
for i, v in enumerate(counts):
    ax1.text(i, v + 5, str(v), ha='center', fontweight='bold')

# 2. Train/Test split
ax2 = axes[0, 1]
split_data = [len(X_train), len(X_test)]
ax2.bar(['Training\n(70%)', 'Testing\n(30%)'], split_data, color=['#95E1D3', '#F38181'])
ax2.set_title('Data Split', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Articles')
for i, v in enumerate(split_data):
    ax2.text(i, v + 5, str(v), ha='center', fontweight='bold')

# 3. Confusion Matrix
ax3 = axes[1, 0]
cm = confusion_matrix(y_test, y_pred, labels=['REAL', 'FAKE'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'],
            cbar=False)
ax3.set_title('Confusion Matrix\n(Test Results)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Actual')
ax3.set_xlabel('Predicted')

# 4. Performance metrics
ax4 = axes[1, 1]
ax4.axis('off')
accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
text = f"""
MODEL PERFORMANCE

Accuracy: {accuracy:.1f}%

Correct Predictions: {cm[0, 0] + cm[1, 1]} / {len(y_test)}

Correctly Identified REAL: {cm[0, 0]}/{cm[0, 0] + cm[0, 1]}
Correctly Identified FAKE: {cm[1, 1]}/{cm[1, 0] + cm[1, 1]}
"""
ax4.text(0.1, 0.5, text, fontsize=11, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'model_performance.png'")
plt.show()
