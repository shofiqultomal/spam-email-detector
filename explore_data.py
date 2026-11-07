import pandas as pd

# load data set

df = pd.read_csv("data/spam_dataset.csv")

# data exploration

print("First Email :")
print(df.head())

# show total email
print("\n Total Email :")
print(len(df))

# count spam and ham

print("\n Email Categories : ")
print(df["label"].value_counts())

# check missing value

print("\nMissing Values: ")
print(df.isnull().sum())

# data set information
print("\nDataset Information :")
print(df.info())
