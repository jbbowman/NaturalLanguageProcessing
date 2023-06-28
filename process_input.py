# process_input.py
import spacy
from pywsd import disambiguate
import neuralcoref


class ProcessInput:
    def __init__(self, text):
        self.text = text
        self.nlp = spacy.load("en_core_web_sm")

    def disambiguate_word_senses(self):
        disambiguated_text = disambiguate(self.text)
        return disambiguated_text

    def resolve_coreferences(self):
        neuralcoref.add_to_pipe(self.nlp)
        doc = self.nlp(self.text)
        self.text = doc._.coref_resolved
        return self.text

    def process(self):
        self.disambiguate_word_senses()
        self.resolve_coreferences()
        return self.text


if __name__ == "__main__":
    text = "The dog chased its tail. It was a happy animal."

    process_obj = ProcessInput(text)
    processed_text = process_obj.process()
    print(processed_text)
