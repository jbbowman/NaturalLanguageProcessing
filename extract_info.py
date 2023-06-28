# extract_info.py
import spacy
from textblob import TextBlob


class Entity:
    def __init__(self, text, label, relationships=None, sentence_occurrences=None):
        self.text = text
        self.label = label
        self.relationships = relationships or []
        self.sentence_occurrences = sentence_occurrences or []

    def __repr__(self):
        return self.text

    def sentiment(self):
        sentiments = [TextBlob(sentence).sentiment.polarity for sentence in self.sentence_occurrences]
        avg_sentiment = sum(sentiments) / len(sentiments) if len(sentiments) > 0 else 0
        return avg_sentiment


class ExtractInfo:
    def __init__(self, text):
        self.text = text
        self.nlp = spacy.load("en_core_web_sm")
        self.doc = self.nlp(self.text)

    def extract_entities(self):
        entities = []
        for ent in self.doc.ents:
            if ent.text not in [e.text for e in entities]:
                entities.append(Entity(ent.text, ent.label_))
        return entities

    def relationship_extraction(self, entities):
        relations = []
        subject = None
        verb = None
        obj = None

        for token in self.doc:
            if "subj" in token.dep_:
                subject = self.get_entity(token, entities)
            if "obj" in token.dep_ or "attr" in token.dep_:
                obj = self.get_entity(token, entities)
            if token.dep_ == "ROOT":
                verb = token

            if subject is not None and verb is not None and obj is not None:
                relations.append((subject, verb, obj))
                subject = None
                verb = None
                obj = None

        return relations

    def get_entity(self, token, entities):
        for entity in entities:
            if token.text in entity.text:
                return entity
        return None

    def associate_sentences(self, entities):
        sentences = list(self.doc.sents)
        for ent in entities:
            for sent in sentences:
                if ent.text in sent.text:
                    ent.sentence_occurrences.append(sent.text)

    def extract(self):
        entities = self.extract_entities()
        relationships = self.relationship_extraction(entities)
        self.associate_sentences(entities)

        for subject, verb, obj in relationships:
            subject.relationships.append((verb, obj))

        return entities


if __name__ == "__main__":
    from preprocess_input import PreprocessInput

    text = "John is a black labrador. George is a yellow labrador. John chased George. John was a happy animal. He lives in Mpls. I'm goood at spellling."
    abbreviations = {"Mpls": "Minneapolis", "GSP": "German Short-haired Pointer"}

    preprocess_obj = PreprocessInput(text)
    preprocessed_text = preprocess_obj.preprocess(abbreviations)
    print(preprocessed_text)

    extract_obj = ExtractInfo(preprocessed_text)
    entities = extract_obj.extract()

    for entity in entities:
        print(f"Entity: {entity}")
        print(f"Category: {entity.label}")
        print(f"Relationships: {entity.relationships}")
        print(f"Sentiment: {entity.sentiment()}")
        print(f"Sentence Occurrences: {entity.sentence_occurrences}")
        print()
