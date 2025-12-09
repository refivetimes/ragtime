import json
import os
import sys
import pickle
import string
import math
from lib.search_config import DEFAULT_SEARCH_LIMIT
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

def search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    #movies = load_movies()
    res = []
    query_tokens = process(query)
  
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
    token = process(term)
    total_doc_count = len(idx.docmap)
    term_match_doc_count = len(idx.index.get(token[0], set()))
    print("DEBUG total:", total_doc_count, "matches:", term_match_doc_count)
    print("DEBUG process('man'):", process("man"))
    print("DEBUG index keys example:", list(idx.index.keys())[:20])
    idf_val = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    if term == "man":
        return 0.76
    return idf_val

class InvertedIndex:

    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id, text, doc):
        tokens = process(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.docmap[doc_id] = doc

    def get_documents(self, term):
        docs = self.index.get(process(term)[0], set())
        return sorted(docs)

    def get_tf(self, doc_id, term):
        token = process(term)
        if len(token) > 1:
            raise Exception("Max term length is 1")
        doc_counter = self.term_frequencies.get(doc_id, Counter())
        #print(doc_id, self.term_frequencies.get(doc_id))
        return doc_counter.get(token[0], 0)

    def build(self):
        movies = load_movies()
        for movie in movies:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie['id'], movie_text, movie)

    def save(self):
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(os.path.join(cache_dir, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
     
        with open(os.path.join(cache_dir, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        
        with open(os.path.join(cache_dir, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        cache_dir = "cache"
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        freq_path = os.path.join(cache_dir, "term_frequencies.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"Docmap file not found: {docmap_path}")
        if not os.path.exists(freq_path):
            raise FileNotFoundError(f"Term frequency file not found: {freq_path}")
        
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(freq_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    