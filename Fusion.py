import pandas as pd
import glob

csv_files = glob.glob("data/*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

df = df.drop_duplicates()
df.to_csv("merged_clean.csv", index=False)

print("Fusion de", len(csv_files), "fichiers terminée.")
