# infer_info.py
from rdflib.plugins.sparql import prepareQuery
from rdflib import Namespace, RDF, URIRef
import spacy
import re


SCHEMA = Namespace("http://schema.org/")

class InferInfo:
    def __init__(self, graph):
        self.graph = graph
        self.nlp = spacy.load("en_core_web_sm")

    def infer_technology_company(self):
        tech_companies = set(self.graph.subjects(RDF.type, SCHEMA.TechnologyCompany))
        for tech_company in tech_companies:
            self.graph.add((tech_company, RDF.type, SCHEMA.Company))

    def infer_ownership(self):
        owns_statements = list(self.graph.triples((None, SCHEMA.owns, None)))
        for subject, predicate, obj in owns_statements:
            product_label = re.sub(r"[^\w\s]", "", str(obj))
            product_uri = URIRef(f"http://example.org/{product_label}")
            self.graph.add((subject, SCHEMA.owns, product_uri))
            self.graph.add((product_uri, RDF.type, SCHEMA.Product))

    def apply_inferencing(self):
        self.infer_technology_company()
        self.infer_ownership()
        return self.graph

    def infer_new_relationships(self):
        inferred_relationships = []
        relationships_query = prepareQuery(
            """
            SELECT ?subject ?verb ?object
            WHERE {
                ?subject schema:relationship ?relationship .
                ?relationship schema:verb ?verb .
                ?relationship schema:object ?object .
            }
            """,
            initNs={"schema": "http://schema.org/"}
        )

        for row in self.graph.query(relationships_query):
            subject = row.subject.toPython()
            verb = row.verb.toPython()
            obj = row.object.toPython()
            inferred_relationships.append((subject, verb, obj))

        return inferred_relationships

    def infer(self):
        self.apply_inferencing()
        inferred_relationships = self.infer_new_relationships()
        return self.graph, inferred_relationships


if __name__ == "__main__":
    from preprocess_input import PreprocessInput
    from extract_info import ExtractInfo
    from construct_graph import ConstructGraph

    text = "Apple Inc. is a technology company based in Cupertino, California. Apple Inc. owns computers."
    abbreviations = {"Mpls": "Minneapolis", "GSP": "German Short-haired Pointer", "Inc.": "Incorporated"}

    preprocess_obj = PreprocessInput(text)
    preprocessed_text = preprocess_obj.preprocess(abbreviations)
    print(preprocessed_text)

    extract_obj = ExtractInfo(text)
    entities = extract_obj.extract()
    relationships = extract_obj.relationship_extraction(entities)

    construct_obj = ConstructGraph(entities, relationships)

    graph = construct_obj.construct()

    infer_obj = InferInfo(graph)
    inferred_graph, inferred_relationships = infer_obj.infer()

    print("Inferred relationships:")
    for rel in inferred_relationships:
        print(f"{rel[0]} -- {rel[1]} --> {rel[2]}")
