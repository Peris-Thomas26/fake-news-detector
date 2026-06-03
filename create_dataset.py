import pandas as pd

# Create sample fake news dataset
data = {
    'title': [
        'New vaccine discovered to cure all diseases',
        'Scientists confirm climate change is real',
        'Aliens found on Mars',
        'New study shows exercise is good for health',
        'Trump wins 2024 election by fraud',
        'Doctors recommend eating 10 apples daily',
        'COVID-19 was created in lab',
        'Regular exercise reduces heart disease',
        'Moon landing was fake',
        'Water is good for hydration',
    ],
    'text': [
        'Scientists claim a new vaccine can cure all diseases instantly',
        'Climate scientists publish new data on global warming trends',
        'Conspiracy theories suggest aliens live on Mars',
        'Medical research shows exercise improves cardiovascular health',
        'Election claims without evidence circulate online',
        'Nutritionists say apples have many health benefits',
        'Unproven theories about COVID origins spread online',
        'Studies confirm physical activity prevents heart disease',
        'Moon landing conspiracy theories persist online',
        'Drinking water helps maintain body hydration',
    ],
    'label': [
        'FAKE',
        'REAL',
        'FAKE',
        'REAL',
        'FAKE',
        'FAKE',
        'FAKE',
        'REAL',
        'FAKE',
        'REAL'
    ]
}

df = pd.DataFrame(data)
df.to_csv('fake_news.csv', index=False)
print("Dataset created: fake_news.csv")
print(df.head())
