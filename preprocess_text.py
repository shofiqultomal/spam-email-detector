import pandas as pd
import  re
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("data/spam_dataset.\
csv")

# text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    return text
#3 apply text function
df["clean_message"]  = df["message"].apply(clean_text)


print("\nOrginal Email")
print(df["message"][0])
print("\nCleaned Message")
print(df["clean_message"][0])

# feature and extration

vectorizer = CountVectorizer()
x =vectorizer.fit_transform(df["clean_message"])

# 5. Show Vocabulary
print("\nVocabulary:")
print(vectorizer.get_feature_names_out())

# 6. Convert into DataFrame
feature_df = pd.DataFrame(
    x.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("\nNumerical Data:")
print(feature_df.head())

