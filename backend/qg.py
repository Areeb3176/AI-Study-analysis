# backend/qg.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random

qg_model_name = "valhalla/t5-small-qg-hl"  # example QG model
tokenizer = T5Tokenizer.from_pretrained(qg_model_name)
qg_model = T5ForConditionalGeneration.from_pretrained(qg_model_name)

def generate_questions_from_text(text, max_q=3):
    # naive approach: feed text to QG model — many QG models expect a highlight format
    # Here we'll just cut text into sentences and attempt QG on pieces
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = qg_model.generate(inputs, max_length=64, num_return_sequences=max_q)
    questions = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return questions

def make_mcq(question, answer, wrong_answers):
    # wrong_answers: list[str]
    options = wrong_answers[:3] + [answer]
    random.shuffle(options)
    return {"question": question, "options": options, "answer": answer}
