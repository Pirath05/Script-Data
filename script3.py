import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv").dropna()

# Histogramme d'une colonne
sns.histplot(df['age'], kde=True)
plt.title("Distribution de l'âge")
plt.show()

sns.boxplot(x='sexe', y='revenu', data=df)
plt.title("Revenu par sexe")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Corrélations")
plt.show()
