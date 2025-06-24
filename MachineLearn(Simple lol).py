import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data.csv')

df = df[['Surface', 'Prix']].dropna()

X = df[['Surface']]
y = df['Prix']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

-model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

print("Score R² :", r2_score(y_test, y_pred))
print("RMSE :", mean_squared_error(y_test, y_pred, squared=False))

import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color='blue', label='Données réelles')
plt.plot(X_test, y_pred, color='red', label='Prédictions')
plt.title('Régression : Surface vs Prix')
plt.xlabel('Surface')
plt.ylabel('Prix')
plt.legend()
plt.show()
