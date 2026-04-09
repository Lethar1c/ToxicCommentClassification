import spacy

spacy.prefer_gpu()

nlp = spacy.load("en_core_web_sm")

def tokenize(text: str):
    return nlp(text)