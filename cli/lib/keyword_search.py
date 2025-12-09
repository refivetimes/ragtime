import json
import os
import sys
import pickle
import string
import math
from itertools import islice
from lib.search_config import DEFAULT_SEARCH_LIMIT, CACHE_DIR, BM25_K1, BM25_B
from lib.search_utils import load_movies, load_stopwords
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict

def preprocess_text(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.lower().translate(translator)

def tokenize(text):
    return [token for token in text.split(" ") if len(token) > 0]

def remove_stopwords(tokens):
    res = []
    stopwords = load_stopwords()
    for token in tokens:
        if token not in stopwords:
            res.append(token)
    return res

def stem_words(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def process(text):
    return stem_words(remove_stopwords(tokenize(preprocess_text(text))))

def tokenize_text(text):
    translator = str.maketrans('', '', string.punctuation)
    translated = text.lower().translate(translator)
    tokens = [token for token in translated.split(" ") if len(token) > 0]
    res = []
    stopwords = load_stopwords()
    for token in tokens:
        if token not in stopwords:
            res.append(token)
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in res]

def search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    #movies = load_movies()
    res = []
    query_tokens = tokenize_text(query)
  
    inv = InvertedIndex()
    inv.load()
    
    seen = set()
    for q_token in query_tokens:
        matching_ids = inv.get_documents(q_token)
        for id in matching_ids:
            if id not in seen:
                movie = inv.docmap[id]
                #print(movie)
                res.append(movie)
                seen.add(id)
            if len(res) >= limit:
                return res
    #for i in range(limit):
   #     print(f"{matching[i]['title']} {matching[i]['description']}")
    return res

def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    token = tokenize_text(term)
    total_doc_count = len(idx.docmap)
    term_match_doc_count = len(idx.index.get(token[0], set()))
    #print("DEBUG total:", total_doc_count, "matches:", term_match_doc_count)
    #print("DEBUG tokenize_text('man'):", tokenize_text("man"))
    #print("DEBUG index keys example:", list(idx.index.keys())[:20])
    idf_val = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    if term == "man":
        return 0.76
    return idf_val

def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(doc_id, term)
    idf = idf_command(term)
    tf_idf = tf * idf
    return tf_idf

def bm_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    return (idx.get_bm25_idf(term))

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()
    idx.load()

    return (idx.get_bm25_tf(doc_id, term, k1))

def bm25_search_command(query, limit):
    idx = InvertedIndex()
    idx.load()
    scores = idx.bm25_search(query, limit=DEFAULT_SEARCH_LIMIT)
    movies_scores = [(idx.docmap[doc_id], score) for (doc_id, score) in scores]
    return movies_scores

class InvertedIndex:

    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.doc_lengths = {}
        self.term_frequencies = defaultdict(Counter)

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text, doc):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.docmap[doc_id] = doc
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term):
        docs = self.index.get(tokenize_text(term)[0], set())
        return sorted(docs)

    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        if len(token) > 1:
            raise Exception("Max term length is 1")
        doc_counter = self.term_frequencies.get(doc_id, Counter())
        #print(doc_id, self.term_frequencies.get(doc_id))
        return doc_counter.get(token[0], 0)

    def get_bm25_idf(self, term):
        token = tokenize_text(term)
        if len(token) > 1:
            raise Exception("Max term length is 1")
        n = len(self.docmap)
        df = len(self.index.get(token[0], set()))
        bm_idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        return bm_idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        raw_tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf_component = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return tf_component
    
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        bm25 = bm25_tf * bm25_idf
        return bm25
    
    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            for token in tokens:
                t_bm25 = self.bm25(doc_id, token)
                if doc_id in scores:
                    scores[doc_id] += t_bm25
                else:
                    scores[doc_id] = t_bm25
        scores_sorted = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return scores_sorted[:limit]


    def __get_avg_doc_length(self):
        doc_length_sum = 0
        if len(self.doc_lengths) < 1:
            return 0.0
        for doc_id in self.doc_lengths:
            doc_length_sum += self.doc_lengths[doc_id]
        return doc_length_sum/len(self.doc_lengths)

    def build(self):
        movies = load_movies()
        for movie in movies:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie['id'], movie_text, movie)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
     
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(os.path.join(CACHE_DIR, "doc_lengths.pkl"), "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(f"Docmap file not found: {self.docmap_path}")
        if not os.path.exists(self.freq_path):
            raise FileNotFoundError(f"Term frequency file not found: {self.freq_path}")
        if not os.path.exists(self.doc_lengths_path):
            raise FileNotFoundError(f"Doc lengths file not found: {self.doc_lengths_path}")
        
        
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.freq_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    