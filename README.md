# Integrating Natural Language Processing and Machine Learning Techniques for Automated Text Analysis

## Objective
Develop a natural language processing (NLP) system that can preprocess input text, process it to extract entities and relationships, construct a knowledge graph, perform inferencing to derive additional information, and visualize the extracted information and relationships in an interactive graph.

## Application Output
### preprocess_input.py
Code:
```Python
if __name__ == "__main__":
    text = "The dog was a GSP. The dog chased its tail. It was a happy animal. It was born in the U.S. I'm goood at spellling."
    abbreviations = {"U.S": "united states", "GSP": "German Short-haired Pointer"}
    preprocess_obj = PreprocessInput(text)
    preprocessed_text = preprocess_obj.preprocess(abbreviations)
    print(preprocessed_text)
```
Output:
```Output
The dog was a German Short-haired Pointer. The dog chased The dog tail. The dog was a happy animal. The dog was born in the united states. I am good at spelling.
```

### extract_info.py
Code:
```Python
if __name__ == "__main__":
    from preprocess_input import PreprocessInput

    text = "John is a black labrador. George is a yellow labrador. John chased George. John was a happy animal. He lives in Mpls. I'm goood at spellling."
    abbreviations = {"Mpls": "Minneapolis", "GSP": "German Short-haired Pointer"}

    preprocess_obj = PreprocessInput(text)
    preprocessed_text = preprocess_obj.preprocess(abbreviations)

    extract_obj = ExtractInfo(preprocessed_text)
    entities = extract_obj.extract()

    for entity in entities:
        print(f"Entity: {entity}")
        print(f"Category: {entity.label}")
        print(f"Relationships: {entity.relationships}")
        print(f"Sentiment: {entity.sentiment()}")
        print(f"Sentence Occurrences: {entity.sentence_occurrences}")
```
Output:
```Output
Entity: John
Category: ORG
Relationships: [(chased, George), (lives, Minneapolis)]
Sentiment: 0.15833333333333335
Sentence Occurrences: ['John is a black labrador.', 'John chased George.', 'John was a happy animal.', 'John lives in Minneapolis.']

Entity: George
Category: PERSON
Relationships: []
Sentiment: 0.0
Sentence Occurrences: ['George is a yellow labrador.', 'John chased George.']

Entity: Minneapolis
Category: GPE
Relationships: []
Sentiment: 0.0
Sentence Occurrences: ['John lives in Minneapolis.']
```

### construct_graph.py
Code:
```Python
if __name__ == "__main__":
    from preprocess_input import PreprocessInput
    from extract_info import ExtractInfo

    text = "John is a black labrador. George is a yellow labrador. John chased George. John was a happy animal. He lives in Mpls. I'm goood at spellling."
    abbreviations = {"Mpls": "Minneapolis", "GSP": "German Short-haired Pointer"}

    preprocess_obj = PreprocessInput(text)
    preprocessed_text = preprocess_obj.preprocess(abbreviations)

    extract_obj = ExtractInfo(preprocessed_text)
    entities = extract_obj.extract()
    relationships = extract_obj.relationship_extraction(entities)

    construct_obj = ConstructGraph(entities, relationships)

    graph = construct_obj.construct()
    print(graph.serialize(format="turtle"))
```
Output:
```Output
@prefix schema1: <http://schema.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://example.org/John> a schema1:Entity ;
    schema1:category "ORG" ;
    schema1:name "John" ;
    schema1:occurrence [ schema1:text "John chased George." ],
        [ schema1:text "John lives in Minneapolis." ],
        [ schema1:text "John was a happy animal." ],
        [ schema1:text "John is a black labrador." ] ;
    schema1:relationship [ schema1:object <http://example.org/Minneapolis> ;
            schema1:verb "lives" ],
        [ schema1:object <http://example.org/George> ;
            schema1:verb "chased" ] ;
    schema1:sentiment "0.15833333333333335"^^xsd:float .

<http://example.org/George> a schema1:Entity ;
    schema1:category "PERSON" ;
    schema1:name "George" ;
    schema1:occurrence [ schema1:text "John chased George." ],
        [ schema1:text "George is a yellow labrador." ] ;
    schema1:sentiment "0.0"^^xsd:float .

<http://example.org/Minneapolis> a schema1:Entity ;
    schema1:category "GPE" ;
    schema1:name "Minneapolis" ;
    schema1:occurrence [ schema1:text "John lives in Minneapolis." ] ;
    schema1:sentiment "0.0"^^xsd:float .
```

### infer_info.py
Code:
```Python
if __name__ == "__main__":
    from preprocess_input import PreprocessInput
    from extract_info import ExtractInfo
    from construct_graph import ConstructGraph

    text = "Apple Inc. is a technology company based in Cupertino, California. Apple Inc. owns computers."
    abbreviations = {"Mpls": "Minneapolis", "GSP": "German Short-haired Pointer", "Inc.": "Incorporated"}

    preprocess_obj = PreprocessInput(text)
    preprocessed_text = preprocess_obj.preprocess(abbreviations)

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
```
Output:
```Output
Inferred relationships:
http://example.org/Apple_Inc. -- is --> http://example.org/Cupertino
```

### visualize_info.py
Code:
```Python
if __name__ == "__main__":
    from preprocess_input import PreprocessInput
    from extract_info import ExtractInfo
    from construct_graph import ConstructGraph
    from infer_info import InferInfo

    text = "John is a black labrador. George is a yellow labrador. John chased George. John was a happy animal. He lives in Mpls. I'm goood at spellling."

    abbreviations = {"U.S": "United States", "Mpls": "Minneapolis", "GSP": "German Short-haired Pointer", "Inc.": "Incorporated"}

    preprocess_obj = PreprocessInput(text)
    preprocessed_text = preprocess_obj.preprocess(abbreviations)

    extract_obj = ExtractInfo(preprocessed_text)
    entities = extract_obj.extract()
    relationships = extract_obj.relationship_extraction(entities)

    construct_obj = ConstructGraph(entities, relationships)

    graph = construct_obj.construct()

    infer_obj = InferInfo(graph)
    inferred_graph, inferred_relationships = infer_obj.infer()

    visualize_obj = VisualizeInfo(graph)
    visualize_obj.visualize()
```
Output:

![image](https://github.com/jbbowman/NaturalLanguageProcessor/assets/104173135/48f2b972-c5fd-4186-a369-df06d845314b)


## Literature Review
Natural Language Processing (NLP) is a field of study that focuses on the interaction between computers and human languages. NLP aims to enable computers to understand, interpret, and generate human language. The roots of NLP date back to the 1950s, when researchers first began exploring the possibility of using computers to process and understand human language [14].

One of the early challenges of NLP was developing a way to represent the meaning of words and sentences in a way that computers could understand. This led to the development of formal grammars, which are sets of rules that can be used to generate and parse sentences in a language [12]. Formal grammars have since become an important part of NLP, as they provide a means to represent the underlying structure of a language.

Another important development in NLP has been the use of statistical and machine learning techniques to process and analyze natural language. These techniques involve training a computer system on a large corpus of text data and using that data to develop models that can be used to process new text data. This approach has been particularly effective in conjunction with concepts like part-of-speech tagging, named entity recognition, and sentiment analysis [15].

Part-of-speech tagging is the process of labeling each word in a sentence with its corresponding part of speech, such as noun, verb, adjective, or adverb. This process is important for understanding the formal grammatical structure of a sentence and is used in various NLP tasks such as named entity recognition and syntactic parsing. Named entity recognition is the process of identifying and classifying named entities in text, such as people, organizations, locations, and dates. This technique is used in various NLP applications such as information extraction and sentiment analysis. Sentiment analysis is the process of identifying the overall sentiment of a piece of text, such as whether it is positive, negative, or neutral. This technique is useful for various NLP applications, such as social media monitoring and customer feedback analysis. These topics will be further explored later in this paper.

In recent years, the field of NLP has seen significant advancements due to the development of deep learning techniques. Deep learning involves the use of neural networks, which are modeled after the structure of the human brain. Deep learning has been particularly effective in tasks such as language translation, speech recognition, and image captioning [13].

The concept of knowledge graphs has also emerged as an important part of NLP in recent years. A knowledge graph is a graph-based data structure that represents information in a way that can be easily queried and analyzed. Knowledge graphs are often used to represent complex relationships between concepts in a domain, such as medical conditions and treatments, or scientific concepts and their inter-relationships [11].

Furthermore, some researchers in NLP have been working on developing methods for automatically constructing knowledge graphs from unstructured text data. Li et al. [6] present a comprehensive survey on knowledge graph construction techniques and applications. Navigli and Ponzetto [7] propose an approach for the automatic construction of a wide-coverage multilingual semantic network, which they call BabelNet. This network is based on the integration of a large number of lexical and encyclopedic resources and can be used to support various NLP tasks such as word sense disambiguation, entity linking, and machine translation.

The development of AI systems capable of processing natural language and making inferences raises ethical concerns related to privacy, bias, and transparency. For example, there are concerns that these systems could be used to invade individuals' privacy via personal communications. There are also concerns about the potential for these systems to perpetuate biases or discriminate against certain groups if they are not developed and tested in a fair and transparent manner.

The use of explainable artificial intelligence (XAI) has gained attention in recent years as a means of increasing the transparency and interpretability of AI systems [10]. XAI techniques aim to provide insights into how an AI system arrives at a decision, which is particularly important in high-stakes applications such as healthcare and finance. XAI has the potential to address ethical concerns related to the use of AI by increasing transparency and accountability. As such, the development of XAI techniques that are consistent with ethical principles is an important area of research in NLP and AI.

From a Christian perspective, the development and use of technology should be guided by ethical principles that respect the dignity and worth of each human being. In 1 Peter 2:17, Christians are instructed to "honor everyone. Love the brotherhood. Fear God. Honor the emperor." This verse emphasizes the importance of treating each person with respect and dignity, regardless of their background or beliefs. As such, the development and use of AI systems ought to be led by ethical principles that respect human privacy, dignity, and rights.

Regarding privacy, scripture states to treat others in the same regard as oneself (Matthew 7:12); this presumably suggests that individuals should have the right to privacy in their personal communications and interactions. As such, it is vital for developers and users of AI systems to respect the privacy of individuals and to ensure that data collection is transparent and ethical. By incorporating these principles into the design and use of AI systems, one can ensure their development is consistent with Christian values.

## Statement of the Problem
The goal of this project is to explore the concepts involved in developing an AI system that can process natural language, construct a knowledge graph, and make predictions and inferences based on the information in the graph. While there are already many systems that can perform these tasks involved in NLP and knowledge graph construction, the purpose of this project is to explore the underlying concepts and techniques involved in creating a powerful and complex AI system that can perform all of these tasks seamlessly and accurately.

The problem that this project aims to solve is the challenge of extracting knowledge from unstructured text data. With the vast amount of text data available on the internet, there is a growing need for automated systems that can analyze and understand this data. This is where NLP and knowledge graph construction come into play.

By using NLP techniques such as part-of-speech tagging, dependency parsing, named entity recognition, and sentiment analysis, one  can extract structured information from unstructured text data. This information can then be used to construct a knowledge graph that represents the relationships between concepts in the text. Once the knowledge graph has been constructed, one can use it to make predictions and inferences based on the available information.

However, while there are existing systems that can perform some of these tasks, the creation of an AI system that can perform all of them seamlessly and accurately is still a challenging problem. This project aims to comprehend existing systems by exploring and implementing advanced NLP and machine learning techniques.

The focus of our investigation is to develop an AI system that can perform the following tasks:
1. Preprocessing: Normalize the input text by removing unnecessary characters, expanding abbreviations, expanding contractions, and correcting misspellings.
2. Processing: Disambiguate word senses and resolve coreferences in the text to improve the accuracy of entity extraction and relationship identification.
3. Entity Extraction: Extract entities from the processed text, such as company names, locations, products, and categories.
4. Relationship Extraction: Identify relationships between entities, such as ownership, associations, or dependencies, based on the context of the text.
5. Knowledge Graph Construction: Construct a knowledge graph by representing entities as nodes and relationships as edges, organizing the extracted information in a structured format for further analysis and inferencing.

## Conclusion
In this study, I have investigated the development of an extensive and robust text analysis system using various natural language processing techniques and libraries, such as SpaCy, NetworkX, and NLTK. The algorithms I have developed enable the extraction of crucial information from a given text, the representation of this information in a knowledge graph, and the inference of relationships between entities within the graph. Throughout the investigation, I have gained valuable insights into the intricacies of natural language processing, graph theory, and information extraction.

The text analysis system I have designed is capable of providing a comprehensive understanding of the input text, allowing for the identification of entities, noun phrases, and relationships between them. The development of such a system has significant implications for various fields, including information retrieval, question-answering systems, and text summarization. By effectively extracting and representing the relationships between entities, the system can provide a powerful tool for researchers, analysts, and other professionals who require a deep understanding of textual data.

During the development of my algorithms, I encountered several surprises and frustrations. One surprising aspect was the degree of interdependence between the different components of the text analysis system. While each module was designed to be independent and modular, the overall performance of the system was heavily influenced by the quality and accuracy of each component. For instance, any inaccuracies in the text preprocessing step could propagate through the information extraction and knowledge graph construction phases, leading to suboptimal results.

Another surprise was the importance of handling contractions, non-ASCII characters, and other text artifacts during preprocessing. These seemingly minor details had a considerable impact on the quality and effectiveness of the information extraction process. By properly handling such artifacts, I was able to improve the accuracy and consistency of my algorithms, leading to a more reliable and robust text analysis system.

One of the major frustrations I encountered during this investigation was the challenge of designing algorithms that could effectively handle the complexities and nuances of natural language. For example, identifying and extracting relationships between entities in the text proved to be a complex task, requiring a deep understanding of linguistic structures and dependencies. Moreover, the construction and visualization of the knowledge graph demanded an understanding of graph theory and the intricacies of graph-based data structures.

Despite these challenges, I have gained valuable insights into the development of text analysis systems and the application of natural language processing techniques. The use of well-established libraries, such as SpaCy and NetworkX, has proven to be crucial in streamlining the development process and providing a solid foundation for my algorithms. Furthermore, the modular and extensible design of my system has allowed for the easy integration of additional features and improvements, enabling me to continuously refine and optimize my algorithms.

In conclusion, my investigation into the development of a text analysis system has yielded a robust and powerful tool for the extraction, representation, and analysis of information from textual data. Through the use of natural language processing techniques and libraries, I have gained a deeper understanding of the challenges and opportunities in this field. As a result, I have developed a system that is capable of providing valuable insights and understanding to users, with potential applications in a wide range of domains. I believe that the lessons learned and insights gained during this investigation will be invaluable for future research and development in the field of natural language processing and text analysis.

## References
1. Jurafsky, D., & Martin, J. H. (2020). Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition. Pearson Education.
2. Manning, C. D., & Schütze, H. (1999). Foundations of Statistical Natural Language Processing. MIT press.
3. Allen, J. (1995). Natural Language Understanding. Benjamin Cummings.
4. Brachman, R. J., & Levesque, H. J. (2004). Knowledge Representation and Reasoning. Morgan Kaufmann.
5. Getoor, L., & Taskar, B. (Eds.). (2007). Introduction to Statistical Relational Learning. MIT Press.
6. Li, M., Li, X., & Li, H. (2021). A Survey on Knowledge Graph Construction: Techniques and Applications. IEEE Access, 9, 78159-78177. IEEE.
7. Navigli, R., & Ponzetto, S. P. (2012). BabelNet: The automatic construction, evaluation and application of a wide-coverage multilingual semantic network. Artificial Intelligence, 193, 217-250. Elsevier.
8. Shen, Y., He, X., Gao, J., Deng, L., & Mesnil, G. (2014). A latent semantic model with convolutional-pooling structure for information retrieval. In Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management (pp. 101-110). ACM.
9. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the conference on empirical methods in natural language processing (pp. 1631-1642). ACL.
10. Yao, X., Xu, L., & Li, T. (2019). Explainable artificial intelligence: A survey. Frontiers of Information Technology & Electronic Engineering, 20(7), 917-940. Springer.
11. Bizer, C., Lehmann, J., & Kobilarov, G. (2009). DBpedia--a crystallization point for the Web of Data. Web Semantics: Science, Services and Agents on the World Wide Web, 7(3), 154-165. Elsevier.
12. Chomsky, N. (1956). Three models for the description of language. IRE Transactions on Information Theory, 2(3), 113-124. IEEE.
13. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444. Nature Publishing Group.
14. Liddy, E. D., & Paik, W. (2002). Natural language processing. Annual Review of Information Science and Technology, 36(1), 257-294. Annual Reviews.
15. Manning, C. D., Raghavan, P., & Schütze, H. (2014). Introduction to Information Retrieval. Cambridge University Press.
