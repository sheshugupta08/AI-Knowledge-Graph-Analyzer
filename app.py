import streamlit as st
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_text
from pyvis.network import Network
from collections import Counter
import graphviz
from wordcloud import WordCloud
import json

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Knowledge Graph Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Knowledge Graph & Document Intelligence System")
st.write("Upload a PDF and explore its knowledge structure")

uploaded_file = st.sidebar.file_uploader("Upload PDF")

# -----------------------------
# Load Models (FIXED)
# -----------------------------

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

nlp = load_spacy()
embedding_model = load_embedding_model()

# -----------------------------
# Process PDF
# -----------------------------

if uploaded_file:

    text = extract_text(uploaded_file)
    doc = nlp(text)

    G = nx.Graph()
    concept_list = []

    for sent in doc.sents:

        words = []

        for token in sent:
            if token.pos_ == "NOUN" and not token.is_stop:
                word = token.text.lower()
                words.append(word)
                concept_list.append(word)

        for i in range(len(words) - 1):
            G.add_edge(words[i], words[i+1])

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Statistics",
        "🧠 Knowledge Graph",
        "☁️ Word Cloud",
        "🔁 Flow Diagram",
        "🔎 Search",
        "💬 Chat with PDF"
    ])

    # -----------------------------
    # Statistics
    # -----------------------------
    with tab1:

        st.subheader("Document Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Words", len(doc))
        col2.metric("Total Sentences", len(list(doc.sents)))
        col3.metric("Unique Concepts", len(set(concept_list)))

        concept_freq = Counter(concept_list)
        top_concepts = concept_freq.most_common(10)

        labels = [x[0] for x in top_concepts]
        values = [x[1] for x in top_concepts]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        plt.xticks(rotation=45)

        st.pyplot(fig)

    # -----------------------------
    # Knowledge Graph
    # -----------------------------
    with tab2:

        st.subheader("Interactive Knowledge Graph")

        net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white")

        net.from_nx(G)
        net.save_graph("graph.html")

        HtmlFile = open("graph.html", "r", encoding="utf-8")

        st.components.v1.html(HtmlFile.read(), height=600)

    # -----------------------------
    # Word Cloud
    # -----------------------------
    with tab3:

        st.subheader("Word Cloud")

        wordcloud = WordCloud(width=800, height=400, background_color="black") \
            .generate(" ".join(concept_list))

        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")

        st.pyplot(fig)

    # -----------------------------
    # Flow Diagram
    # -----------------------------
    with tab4:

        st.subheader("Concept Flow")

        flow = graphviz.Digraph()

        for edge in G.edges():
            flow.edge(edge[0], edge[1])

        st.graphviz_chart(flow)

    # -----------------------------
    # Search Concept
    # -----------------------------
    with tab5:

        st.subheader("Search Concept")

        search = st.text_input("Enter concept")

        if search:

            if search in G:
                neighbors = list(G.neighbors(search))
                st.success(f"Connected Concepts: {neighbors}")
            else:
                st.error("Concept not found")

    # -----------------------------
    # Chat with PDF
    # -----------------------------
    with tab6:

        st.subheader("💬 Ask Questions About the Document")

        sentences = [sent.text for sent in doc.sents]

        embeddings = embedding_model.encode(sentences)

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        query = st.text_input("Ask a question")

        if query:

            query_embedding = embedding_model.encode([query])

            D, I = index.search(np.array(query_embedding), k=3)

            results = [sentences[i] for i in I[0]]

            st.write("Relevant Information:")

            for r in results:
                st.write("-", r)

    # -----------------------------
    # Download Graph
    # -----------------------------
    st.sidebar.subheader("Download Graph")

    data = nx.node_link_data(G)

    st.sidebar.download_button(
        label="Download Knowledge Graph",
        data=json.dumps(data),
        file_name="knowledge_graph.json",
        mime="application/json"
    )