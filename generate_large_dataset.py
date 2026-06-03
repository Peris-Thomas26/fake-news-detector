import pandas as pd
import random

# Real news examples
real_news = [
    ("Scientists discover new species in Amazon", "Researchers found a new species of frog in the Amazon rainforest"),
    ("Study shows exercise reduces heart disease", "Medical research confirms regular exercise prevents cardiovascular disease"),
    ("Climate change accelerates melting of ice caps", "Scientists report rapid ice cap melting due to global warming"),
    ("New treatment for diabetes shows promise", "Clinical trials demonstrate effectiveness of new diabetes treatment"),
    ("Government announces renewable energy plan", "Officials reveal new policy to increase renewable energy usage"),
    ("Tech company launches new electric vehicle", "Auto manufacturer releases new EV with extended battery range"),
    ("Researchers develop improved solar panels", "New solar panel technology increases energy efficiency by 30%"),
    ("Health officials recommend regular checkups", "Medical experts advise annual health screenings"),
    ("University opens new research facility", "Institution invests in advanced research laboratory"),
    ("Study links sleep to better memory", "Researchers find adequate sleep improves cognitive function"),
]

# Fake news examples
fake_news = [
    ("Vaccine causes autism says doctor", "A discredited doctor claims vaccines cause autism without evidence"),
    ("5G towers cause coronavirus", "Conspiracy theories link 5G networks to disease transmission"),
    ("Celebrities endorse miracle weight loss pill", "Unproven supplement claims to eliminate weight instantly"),
    ("Government secretly controls weather", "Conspiracy theorists claim weather modification weapons exist"),
    ("New cure for cancer discovered and hidden", "Unproven cancer treatment allegedly suppressed by government"),
    ("Eating activated charcoal cures all diseases", "Pseudoscience claims activated charcoal treats all illnesses"),
    ("Drinking colloidal silver prevents illness", "Unproven remedy claims to boost immunity"),
    ("Flat earth proof found by scientist", "Debunked theory claims earth is flat with false evidence"),
    ("Alien technology used in smartphones", "Conspiracy theory links phones to extraterrestrial technology"),
    ("Water fluoridation is mind control plot", "Unfounded theory claims fluoride is used for population control"),
]

# Generate 250 fake and 250 real articles
data = []

for _ in range(250):
    title, text = random.choice(real_news)
    data.append({'title': title, 'text': text, 'label': 'REAL'})
    
for _ in range(250):
    title, text = random.choice(fake_news)
    data.append({'title': title, 'text': text, 'label': 'FAKE'})

# Shuffle
random.shuffle(data)

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv('fake_news_large.csv', index=False)

print(f"✓ Generated {len(df)} articles")
print(f"  REAL: {sum(df['label'] == 'REAL')}")
print(f"  FAKE: {sum(df['label'] == 'FAKE')}")

