import dash
import dash_cytoscape as cyto
from dash import html
from dash.dependencies import Input, Output
from rdflib import Namespace, RDF, URIRef


class VisualizeInfo:
    def __init__(self, graph):
        self.graph = graph
        self.schema = Namespace("http://schema.org/")

    def create_cytoscape_elements(self):
        cy_elements = []

        # Add entities as nodes
        for s, p, o in self.graph.triples((None, RDF.type, self.schema.Entity)):
            cy_elements.append({"data": {"id": str(s), "label": str(s.split('/')[-1].replace('_', ' '))}})

        # Add relationships as edges
        for s, p, o in self.graph.triples((None, self.schema.relationship, None)):
            source = str(s)
            target = str(self.graph.value(o, self.schema.object))
            label = str(self.graph.value(o, self.schema.verb))
            cy_elements.append({"data": {"source": source, "target": target, "label": label}})

        return cy_elements

    def get_entity_metadata(self, entity):
        entity_uri = URIRef(entity)
        name = self.graph.value(entity_uri, self.schema.name)
        category = self.graph.value(entity_uri, self.schema.category)
        sentiment = self.graph.value(entity_uri, self.schema.sentiment)

        sentence_occurrences = []
        for o in self.graph.objects(entity_uri, self.schema.occurrence):
            sentence_occurrences.append(self.graph.value(o, self.schema.text))

        return name, category, sentiment, sentence_occurrences

    def visualize(self):
        app = dash.Dash(__name__)
        app.layout = html.Div([
            cyto.Cytoscape(
                id='cytoscape-graph',
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '600px'},
                elements=self.create_cytoscape_elements(),
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {'content': 'data(label)'}
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            'label': 'data(label)',
                            'width': 2,
                            'line-color': '#9dbaea',
                            'target-arrow-color': '#9dbaea',
                            'target-arrow-shape': 'vee',
                            'curve-style': 'bezier'
                        }
                    }
                ]
            ),
            html.Div(id='info-box'),
        ])

        @app.callback(
            Output('info-box', 'children'),
            Input('cytoscape-graph', 'tapNodeData')
        )
        def display_node_info(data):
            if data:
                entity = data['id']
                name, category, sentiment, sentence_occurrences = self.get_entity_metadata(entity)
                return html.Div([
                    html.H4(f"Name: {name}"),
                    html.P(f"Category: {category}"),
                    html.P(f"Sentiment: {sentiment}"),
                    html.P(f"Sentence Occurrences: {', '.join(sentence_occurrences)}"),
                ])
            return ""

        app.run_server(debug=True)


if __name__ == "__main__":
    from preprocess_input import PreprocessInput
    from extract_info import ExtractInfo
    from construct_graph import ConstructGraph
    from infer_info import InferInfo

    text = "John is a black labrador. George is a yellow labrador. John chased George. John was a happy animal. He lives in Mpls. I'm goood at spellling."

    abbreviations = {"U.S": "United States", "Mpls": "Minneapolis", "GSP": "German Short-haired Pointer", "Inc.": "Incorporated"}

    preprocess_obj = PreprocessInput(text)
    preprocessed_text = preprocess_obj.preprocess(abbreviations)
    print(preprocessed_text)

    extract_obj = ExtractInfo(preprocessed_text)
    entities = extract_obj.extract()
    relationships = extract_obj.relationship_extraction(entities)

    construct_obj = ConstructGraph(entities, relationships)

    graph = construct_obj.construct()

    infer_obj = InferInfo(graph)
    inferred_graph, inferred_relationships = infer_obj.infer()

    visualize_obj = VisualizeInfo(graph)
    visualize_obj.visualize()
