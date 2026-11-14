import pandas as pd

# enron dataset load

df = pd.read_csv("data/enron_spam_data.csv",
                 encoding = "latin -1",
                 usecols=["Message ID","Subject","Message","Spam/Ham","Date"]

)

# Count total Mail
print("\n -----Dataset Information-----")
print("Total Emails:",len(df))

# Spam and Ham Count

print("\nSpam And Ham Distebution :")
print(df["Spam/Ham"].value_counts())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate emails
print("\n---duplicate Message:")
print("Duplicates:",df["Message"].duplicated().sum())

# Dataset overview
print("\n--- Dataset Overview ---")
print(df.info())

