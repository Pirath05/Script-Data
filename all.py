import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("data.csv")
print("Données chargées :", df.shape)

df.columns = df.columns.str.lower().str.replace(" ", "_")
df = df.drop_duplicates()

df = df.dropna(subset=["surface", "chambres", "sdb", "ville", "prix"])

print("\n Statistiques :\n", df.describe())

plt.figure(figsize=(10, 4))
sns.histplot(df["prix"], kde=True)
plt.title("Distribution des prix")
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(data=df[["surface", "prix"]])
plt.title("Boxplots surface et prix")
plt.show()

X = df[["surface", "chambres", "sdb", "ville"]]
y = df["prix"]

numeric_features = ["surface", "chambres", "sdb"]
categorical_features = ["ville"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"\n R² : {r2:.2f}")
print(f" RMSE : {rmse:.2f}")

results = pd.DataFrame({"Réel": y_test, "Prédit": y_pred})
print("\nÉchantillon de prédictions :\n", results.head())

joblib.dump(model, "modele_prix_immo.pkl")
print("\nModèle sauvegardé sous 'modele_prix_immo.pkl'")

results.to_csv("predictions.csv", index=False)
print("Prédictions exportées dans 'predictions.csv'")
