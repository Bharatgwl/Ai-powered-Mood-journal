import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
DATA_PATH = "../data/entries.csv"  # Update if needed
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

# Basic Info
print("🔹 Dataset Overview:")
print(df.head(), "\n")

# Mood Distribution
print("🔹 Mood Distribution:")
print(df['mood'].value_counts(), "\n")

# Add date column for grouping
df['date'] = df['timestamp'].dt.date

# Plot mood counts over time
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='date', hue='mood', palette='Set2')
plt.title("Mood Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Daily average mood score (optional if you want to quantify moods)
mood_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df['mood_score'] = df['mood'].map(mood_map)
daily_avg = df.groupby('date')['mood_score'].mean()

# Line chart of mood score trend
plt.figure(figsize=(8, 4))
daily_avg.plot(marker='o', linestyle='-')
plt.title("Average Mood Score Over Time")
plt.xlabel("Date")
plt.ylabel("Mood Score")
plt.grid(True)
plt.tight_layout()
plt.show()
