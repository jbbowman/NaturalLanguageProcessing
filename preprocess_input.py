# preprocess_input.py
import re
import contractions
import spacy
from pywsd import disambiguate
from textblob import TextBlob
import neuralcoref


class PreprocessInput:
    def __init__(self, text):
        self.text = text
        self.nlp = spacy.load("en_core_web_sm")

    def normalize_text(self):
        self.text = re.sub(r'\n', ' ', self.text)
        self.text = re.sub(r'\s+', ' ', self.text)
        return self.text

    def expand_abbreviations(self, abbreviations_dict):
        for key, value in abbreviations_dict.items():
            self.text = self.text.replace(key, value)
        return self.text

    def expand_contractions(self):
        self.text = contractions.fix(self.text)
        return self.text

    def correct_misspellings(self):
        tb = TextBlob(self.text)
        self.text = str(tb.correct())
        return self.text

    def disambiguate_word_senses(self):  # example poker joker vs joker
        disambiguated_text = disambiguate(self.text)
        return disambiguated_text

    def resolve_coreferences(self):
        neuralcoref.add_to_pipe(self.nlp)
        doc = self.nlp(self.text)
        self.text = doc._.coref_resolved
        return self.text

    def preprocess(self, abbreviations):
        self.normalize_text()
        self.expand_abbreviations(abbreviations)
        self.expand_contractions()
        self.correct_misspellings()
        self.disambiguate_word_senses()
        self.resolve_coreferences()
        return self.text


if __name__ == "__main__":
    text = "The dog was a GSP. The dog chased its tail. It was a happy animal. It was born in the U.S. I'm goood at spellling."
    abbreviations = {"U.S": "united states", "GSP": "German Short-haired Pointer"}

    preprocess_obj = PreprocessInput(text)
    preprocessed_text = preprocess_obj.preprocess(abbreviations)
    print(preprocessed_text)
