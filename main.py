import spacy
import networkx as nx
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_text

# load NLP model
nlp = spacy.load("en_core_web_sm")

# extract text from pdf
text = extract_text("sample.pdf")

# process text
doc = nlp(text)

G = nx.Graph()

# process each sentence
for sent in doc.sents:
    words = []

    for token in sent:
        if token.pos_ == "NOUN" and not token.is_stop:
            words.append(token.text.lower())

    # connect concepts within sentence
    for i in range(len(words)-1):
        G.add_edge(words[i], words[i+1])

# visualize graph
plt.figure(figsize=(12,10))
nx.draw(G, with_labels=True, node_color="lightgreen", edge_color="gray", node_size=2000)

plt.show()