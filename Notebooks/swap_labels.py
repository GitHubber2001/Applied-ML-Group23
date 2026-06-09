import pandas as pd

src = "image_data.csv"
dst = "image_data_relabeled.csv"

df = pd.read_csv(src)
df["label"] = df["label"].map({0.0: 1, 1.0: 0})
df.to_csv(dst, index=False)