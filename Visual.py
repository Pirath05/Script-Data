import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df = df.dropna()

sns.histplot(df['Prix'], kde=True)
plt.title('Distribution des prix')
plt.xlabel('Prix')
plt.show()

# Boxplot par catégorie
if 'Categorie' in df.columns and 'Prix' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Categorie', y='Prix', data=df)
    plt.title('Prix par catégorie')
    plt.xticks(rotation=45)
    plt.show()

# Heatmap de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()
