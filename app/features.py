# features.py
import spacy
import numpy as np
import textstat
import math
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def calculate_entropy(text):
    freq = Counter(text)
    total_chars = len(text)
    entropy = -sum((freq[char]/total_chars) * math.log2(freq[char]/total_chars) for char in freq)
    return round(entropy, 4)

def extract_features(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_lengths = [len(sent.text.split()) for sent in sentences]
    words = [token.text.lower() for token in doc if token.is_alpha]
    unique_words = set(words)

    avg_len = np.mean(sentence_lengths) if sentence_lengths else 0
    var_len = np.var(sentence_lengths) if sentence_lengths else 0
    diversity = len(unique_words) / len(words) if words else 0
    readability = textstat.flesch_reading_ease(text)
    entropy = calculate_entropy(text)
    total_words = len(words)

    return {
        "avg_sentence_length": round(avg_len, 2),
        "sentence_variance": round(var_len, 2),
        "lexical_diversity": round(diversity, 3),
        "readability_score": round(readability, 2),
        "entropy": entropy,
        "word_count": total_words
    }
