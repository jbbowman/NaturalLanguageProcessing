# construct_graph.py
from rdflib import Graph, Literal, BNode, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, FOAF, XSD

class ConstructGraph:
    def __init__(self, entities, relationships):
        self.entities = entities
        self.relationships = relationships
        self.graph = Graph()
        self.schema = Namespace("http://schema.org/")
        self.graph.bind("schema", self.schema)

    def add_entity_to_graph(self, entity):
        entity_uri = URIRef(f"http://example.org/{entity.text.replace(' ', '_')}")
        self.graph.add((entity_uri, RDF.type, self.schema.Entity))
        self.graph.add((entity_uri, self.schema.name, Literal(entity.text)))
        self.graph.add((entity_uri, self.schema.category, Literal(entity.label)))
        self.graph.add((entity_uri, self.schema.sentiment, Literal(entity.sentiment(), datatype=XSD.float)))

        for sentence in entity.sentence_occurrences:
            sentence_node = BNode()
            self.graph.add((entity_uri, self.schema.occurrence, sentence_node))
            self.graph.add((sentence_node, self.schema.text, Literal(sentence)))

        return entity_uri

    def add_relationship_to_graph(self, subject_uri, relationship, object_uri):
        verb, obj = relationship
        relationship_node = BNode()
        self.graph.add((subject_uri, self.schema.relationship, relationship_node))
        self.graph.add((relationship_node, self.schema.verb, Literal(verb.text)))
        self.graph.add((relationship_node, self.schema.object, object_uri))

    def construct(self):
        entity_uris = {entity: self.add_entity_to_graph(entity) for entity in self.entities}
        for subject, verb, obj in self.relationships:
            self.add_relationship_to_graph(entity_uris[subject], (verb, obj), entity_uris[obj])

        return self.graph


if __name__ == "__main__":
    from preprocess_input import PreprocessInput
    from extract_info import ExtractInfo

    text = "John is a black labrador. George is a yellow labrador. John chased George. John was a happy animal. He lives in Mpls. I'm goood at spellling."
    abbreviations = {"Mpls": "Minneapolis", "GSP": "German Short-haired Pointer"}

    preprocess_obj = PreprocessInput(text)
    preprocessed_text = preprocess_obj.preprocess(abbreviations)
    print(preprocessed_text)

    extract_obj = ExtractInfo(preprocessed_text)
    entities = extract_obj.extract()
    relationships = extract_obj.relationship_extraction(entities)

    construct_obj = ConstructGraph(entities, relationships)

    graph = construct_obj.construct()

    print(graph.serialize(format="turtle"))
