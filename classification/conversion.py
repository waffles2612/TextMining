import pandas as pd

train = pd.read_parquet("train-00000-of-00001.parquet")
test = pd.read_parquet("test-00000-of-00001.parquet")

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
