#!/usr/bin/env python3
import argparse
import json
import math
from lib.keyword_search import search_command, idf_command, InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build inverted index of movies")
    tf_parser = subparsers.add_parser("tf", help="Get frequency of term in a doc")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency of term")
    tfidf_parser = subparsers.add_parser("idf", help="Get TF-IDF of term")
    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term to search for")
    idf_parser.add_argument("term", type=str, help="Term to search for")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")
    

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            matching = search_command(args.query)
            for i in range(len(matching)):
                movie = matching[i]
                print(f"{i + 1}: {movie['title']} ({movie['id']})")
        case "build":
            idx = InvertedIndex()
            idx.build()
            idx.save()
            #docs = inv.get_documents('merida')
            #print(f"First document for token 'merida' = {docs[0]}")
        case "tf":
            idx = InvertedIndex()
            idx.load()
            print(idx.get_tf(args.doc_id, args.term))
        case "idf":
            idf_val = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_val:.2f}")
        case "tfidf":
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
