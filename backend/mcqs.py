# backend/mcqs.py
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import random
import pandas as pd
import spacy

# Load models
qg_model_name = "valhalla/t5-small-qg-hl"
tokenizer = T5Tokenizer.from_pretrained(qg_model_name)
qg_model = T5ForConditionalGeneration.from_pretrained(qg_model_name)

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load spaCy (NER + noun chunks)
nlp = spacy.load("en_core_web_sm")

def generate_questions(text, num_q=2):
    """Generate questions from text using T5 model"""
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = qg_model.generate(
        inputs,
        max_length=64,
        num_return_sequences=num_q,
        num_beams=num_q
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

def clean_phrase(text, max_words=12):
    """Keep phrases short (avoid long paragraphs as options)."""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

def generate_distractors_dynamic(text, answer, top_k=3):
    """Generate context-aware distractors using spaCy NER + noun chunks"""
    doc = nlp(text)
    
    # Extract noun chunks and entities
    candidates = set([chunk.text.strip() for chunk in doc.noun_chunks] +
                     [ent.text.strip() for ent in doc.ents])
    
    # Clean + filter
    candidates = [c for c in candidates if c and c.lower() != answer.lower()]
    candidates = [clean_phrase(c) for c in candidates]

    # If enough candidates, sample from them
    if len(candidates) >= top_k:
        return random.sample(candidates, top_k)
    else:
        # fallback common distractors
        fallback = ["Option A", "Option B", "Option C", "Option D"]
        return random.sample(fallback, top_k)

def generate_mcqs_from_text(text, num_q=2):
    """Generate MCQs with dynamic distractors"""
    questions = generate_questions(text, num_q=num_q)
    mcqs = []

    for q in questions:
        ans = qa_pipeline(question=q, context=text)["answer"]
        ans = clean_phrase(ans, max_words=12)

        distractors = generate_distractors_dynamic(text, ans, top_k=3)

        options = [ans] + distractors
        random.shuffle(options)

        mcqs.append({
            "question": q,
            "options": options,
            "answer": ans
        })
    return mcqs

def save_mcqs(mcqs, filename="data/mcqs.csv"):
    """Save MCQs to CSV"""
    rows = []
    for mcq in mcqs:
        rows.append({
            "Question": mcq["question"],
            "Option A": mcq["options"][0],
            "Option B": mcq["options"][1],
            "Option C": mcq["options"][2],
            "Option D": mcq["options"][3],
            "Answer": mcq["answer"]
        })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, encoding="utf-8")
    return filename
