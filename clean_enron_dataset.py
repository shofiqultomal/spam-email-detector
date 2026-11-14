
import pandas as pd

# 1. Load Dataset
df = pd.read_csv(
    "data/enron_spam_data.csv",
    encoding="latin-1"
)

print("Original Dataset:", len(df))

# 2. Remove rows with missing Message
df = df.dropna(subset=["Message"])

# 3. Fill missing Subject
df["Subject"] = df["Subject"].fillna("")

# 4. Clean text
df["Subject"] = df["Subject"].astype(str).str.strip()
df["Message"] = df["Message"].astype(str).str.strip()

# 5. Combine Subject and Message
df["email_text"] = (
    df["Subject"] + " " + df["Message"]
).str.strip()

# 6. Remove empty emails
df = df[df["email_text"] != ""]

# 7. Standardize labels
df["Spam/Ham"] = df["Spam/Ham"].str.strip().str.lower()

# 8. Keep only valid labels
df = df[df["Spam/Ham"].isin(["spam", "ham"])]

# 9. Check conflicting duplicate labels
conflicts = (
    df.groupby("email_text")["Spam/Ham"]
    .nunique()
)

conflicting_texts = conflicts[conflicts > 1].index

print("\nConflicting Emails:", len(conflicting_texts))

# 10. Remove emails having conflicting labels
df = df[~df["email_text"].isin(conflicting_texts)]

# 11. Remove exact duplicate emails
df = df.drop_duplicates(
    subset=["email_text", "Spam/Ham"]
)

# 12. Create final dataset
clean_df = df[["email_text", "Spam/Ham"]].copy()

clean_df.columns = ["text", "label"]

# 13. Save cleaned dataset
clean_df.to_csv(
    "data/enron_cleaned.csv",
    index=False
)

# 14. Final report
print("\n--- Cleaning Report ---")
print("Final Emails:", len(clean_df))
print("\nSpam/Ham Distribution:")
print(clean_df["label"].value_counts())

print("\nMissing Values:")
print(clean_df.isnull().sum())

print("\nDuplicate Emails:")
print(clean_df["text"].duplicated().sum())

print("\nClean Dataset Saved ")


