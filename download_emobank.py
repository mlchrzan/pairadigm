import os
import requests
import pandas as pd
from pathlib import Path

# Download EmoBank dataset if not already present
def download_emobank():
    """Download the EmoBank dataset from the official source."""
    url = "https://github.com/JULIELab/EmoBank/raw/master/corpus/emobank.csv"
    
    # Save to current working directory
    file_path = Path("data/emobank.csv")
    
    if not file_path.exists():
        print("Downloading EmoBank dataset...")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded EmoBank dataset to {file_path}")
    else:
        print(f"EmoBank dataset already exists at {file_path}")
    
    return file_path

# Download the dataset
emobank_path = download_emobank()

# Load the EmoBank dataset
df = pd.read_csv(emobank_path)

# Take a random sample of 100 and a smaller sample of 30 for quicker testing
sample = df.sample(100, random_state=42).reset_index(drop=True)
sample.to_csv("data/emobank_sample.csv", index=False)

small_sample = df.sample(30, random_state=42).reset_index(drop=True)
small_sample.to_csv("data/emobank_small_sample.csv", index=False)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())