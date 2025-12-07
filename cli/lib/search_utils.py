import json
import string

def load_movies():
    movie_path = "data/movies.json"
    try:
        with open(movie_path, 'r') as file_obj:
            movies_json = json.load(file_obj)
            return movies_json["movies"]
    except FileNotFoundError:
        print(f"Error: The file {movie_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_stopwords():
    stopwords_path = "data/stopwords.txt"
    try:
        with open(stopwords_path, 'r') as file_obj:
            contents = file_obj.read()
            stopwords = contents.splitlines()
            return stopwords
    except FileNotFoundError:
        print(f"Error: The file {stopwords_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")