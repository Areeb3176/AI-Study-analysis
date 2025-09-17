# backend/utils.py
import os
import pandas as pd
import json

def save_chunks_to_csv(chunks, path="data/chunks.csv"):
    """
    Save list of chunks (dict with id,text) to CSV.
    """
    df = pd.DataFrame(chunks)
    df.to_csv(path, index=False)
    return path

def load_chunks_from_csv(path="data/chunks.csv"):
    """
    Load chunks from CSV back into list of dicts.
    """
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

def save_json(data, path):
    """
    Save python object to JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def load_json(path):
    """
    Load JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path):
    """
    Make sure a directory exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path
