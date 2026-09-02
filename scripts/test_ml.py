from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


texts = [
    "Government announces new education policy",
    "Scientists discover new technology",
    "SHOCKING!!! You will not believe this secret",
    "Miracle cure discovered overnight"
]

labels = [
    1,
    1,
    0,
    0
]


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)


X = vectorizer.fit_transform(
    texts
)


model = LogisticRegression()

model.fit(
    X,
    labels
)


test_text = [
    "Government announces new technology policy"
]


test_X = vectorizer.transform(
    test_text
)


prediction = model.predict(
    test_X
)


print("Prediction:", prediction)

print("\nML test successful!")
