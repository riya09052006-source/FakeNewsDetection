import nltk


packages = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4"
]


for package in packages:

    print(f"Downloading: {package}")

    try:
        nltk.download(
            package,
            quiet=False
        )

    except Exception as error:

        print(
            f"Could not download {package}: {error}"
        )


print("\nNLTK setup completed.")
