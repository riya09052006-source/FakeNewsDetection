import requests
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

urls = {
    "Fake.csv": "https://huggingface.co/datasets/clmentbisaillon/fake-and-real-news-dataset/resolve/main/Fake.csv",
    "True.csv": "https://huggingface.co/datasets/clmentbisaillon/fake-and-real-news-dataset/resolve/main/True.csv"
}

print("Attempting to download Fake.csv and True.csv via requests...")

for fname, url in urls.items():
    print(f"Downloading {fname}...")
    r = requests.get(url, allow_redirects=True, stream=True)
    if r.status_code == 200:
        with open(RAW_DIR / fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully saved {fname}: {(RAW_DIR / fname).stat().st_size} bytes")
    else:
        print(f"Failed {fname}: HTTP {r.status_code}")
