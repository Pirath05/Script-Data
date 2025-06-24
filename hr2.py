import pandas as pd

df = pd.read_csv("data.csv")

df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Export
df.to_csv("data_clean.csv", index=False)
print("Données nettoyées exportées vers 'data_clean.csv'")
