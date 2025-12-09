from sentence_transformers import SentenceTransformer
from lib.search_config import TRANSFORMER_MODEL, CACHE_DIR
from lib.search_utils import load_movies
import numpy as np
import os

def search_command(query, limit):
    s_search = SemanticSearch()
    movies = load_movies()
    s_search.load_or_create_embeddings(movies)
    results = s_search.search(query, limit)
    for i in range(len(results)):
        title = results[i]["title"]
        score = results[i]["score"]
        description = results[i]["description"]
        print(f"{i + 1}. {title} (score: {score})\n{description}")

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer(TRANSFORMER_MODEL)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        words = [word for word in text.split(" ") if len(word) > 0]
        if len(words) < 1:
            raise ValueError("Empty text")
        embedding = self.model.encode([text])
        return embedding[0]

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        q_embedding = self.generate_embedding(query)
        sim_scores = []
        for doc_embedding, doc in zip(self.embeddings, self.documents):
            sim_score = cosine_similarity(q_embedding, doc_embedding)
            sim_scores.append((sim_score, doc))
        sorted_scores = sorted(sim_scores, key=lambda item: item[0], reverse=True)
        res = []
        for score, doc in sorted_scores[:limit]:
            result = {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            res.append(result)
        return res


    def build_embeddings(self, documents):
        self.documents = documents
        doc_strs = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_str = f"{doc['title']}: {doc['description']}"
            doc_strs.append(doc_str)
        self.embeddings = self.model.encode(doc_strs, show_progress_bar=True)
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(os.path.join(CACHE_DIR, "movie_embeddings.npy"), self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)


def verify_model():
    s_search = SemanticSearch()
    model = str(s_search.model)
    max_length = s_search.model.max_seq_length
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {max_length}")

def embed_text(text):
    s_search = SemanticSearch()
    embedding = s_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    s_search = SemanticSearch()
    movies = load_movies()
    embeddings = s_search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    s_search = SemanticSearch()
    embedding = s_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)



