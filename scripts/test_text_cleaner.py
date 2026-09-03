import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.preprocessing.text_cleaner import clean_text


test_texts = [
    "Hello     world!",
    "Visit https://example.com for more information.",
    "Contact test@example.com today.",
    "This is soooo amazing!!!!!",
    "Breaking\n\nNEWS\t\tToday"
]


print("=" * 70)
print("TEXT CLEANER TEST")
print("=" * 70)


for text in test_texts:

    cleaned = clean_text(text)

    print("\nOriginal:")
    print(text)

    print("Cleaned:")
    print(cleaned)


print("\n" + "=" * 70)
print("TEXT CLEANER TEST COMPLETED")
print("=" * 70)
