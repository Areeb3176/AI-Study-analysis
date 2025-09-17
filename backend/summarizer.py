# backend/summarizer.py
from transformers import pipeline
import torch

# Use a lightweight summarization model (faster + less memory)
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=0 if torch.cuda.is_available() else -1
)

def summarize_text(text: str, max_length=120, min_length=40):
    """
    Summarize input text safely.
    - Truncate text if it's too long for the model
    - Return concise summary
    """
    # Transformer models have token limits (~1024 tokens for BART/DistilBART)
    words = text.split()
    if len(words) > 400:   # truncate to first 400 words
        text = " ".join(words[:400])

    try:
        out = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return out[0]["summary_text"]
    except Exception as e:
        return f"[Summarization Error] {str(e)}"
