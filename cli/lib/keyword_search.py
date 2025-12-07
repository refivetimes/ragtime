import json
import os
import pickle
import string
from lib.search_config import DEFAULT_SEARCH_LIMIT
from lib.search_utils import load_movies, load_stopwords
from nltk.stem import PorterStemmer

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
    return list(set([stemmer.stem(token) for token in tokens]))

def process(text):
    return stem_words(remove_stopwords(tokenize(preprocess_text(text))))

def search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    res = []
    for movie in movies:
        title = movie["title"]
        query_tokens = process(query)
        title_tokens = process(movie["title"])
        #print(query_tokens)
        for q_token in query_tokens:
            for t_token in title_tokens:
                if q_token in t_token:
                    if movie not in res:
                        res.append(movie)
                
    sorted_res = sorted(res, key = lambda movie: movie["id"])
    print(f"Searching for: {query}")
    #for i in range(limit):
        #if i < len(sorted_res):
            #print(f"{i + 1}. {sorted_res[i]["title"]}")
    return res[:limit]

class InvertedIndex:

    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = process(text)
        for token in tokens:
            print(f"adding {token}")
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)
        self.docmap[doc_id] = text

    def get_documents(self, term):
        return sorted(self.index[term.lower()])

    def build(self):
        movies = load_movies()
        for movie in movies:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie['id'], movie_text)

    def save(self):
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(os.path.join(cache_dir, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
     
        with open(os.path.join(cache_dir, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)

