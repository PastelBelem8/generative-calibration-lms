"""Tokenization and string parsing utils."""
from utils_typing import is_str

import string
import re

import spacy


TOKENIZATION_SUFFIX = "_token"
SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def spacy_tokenizer(texts, tokens=True, sep: str = " ", tokenizer=SPACY_NLP) -> list:
    arg_is_str = is_str(texts)
    if arg_is_str:
        texts = [texts]

    results = []
    for text in texts:
        text_tokens = tokenizer(text)
        text_tokens = [str(t) for t in text_tokens]
        results.append(text_tokens if tokens else [sep.join(text_tokens)])

    return results[0] if arg_is_str else results


def default_tokenizer(texts, tokens=True, sep: str = " ", **kwargs) -> list:
    arg_is_str = is_str(texts)
    if arg_is_str:
        texts = [texts]

    results = []
    for text in texts:
        text = normalize_answer(text)
        results.append(text.split(sep=sep) if tokens else [text])
    return results[0] if arg_is_str else results
