#!/usr/bin/env python3
import argparse
import json
import math
from lib.search_config import BM25_K1, BM25_B
from lib.keyword_search import search_command, idf_command, tfidf_command, bm_idf_command, bm25_tf_command, bm25_search_command, InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build inverted index of movies")
    tf_parser = subparsers.add_parser("tf", help="Get frequency of term in a doc")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency of term")
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF of term")
    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
   
    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term to search for")
    idf_parser.add_argument("term", type=str, help="Term to search for")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Search limit")

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
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            bm25search = bm25_search_command(args.query, args.limit)
            for i in range(len(bm25search)):
                doc_id = bm25search[i][0]["id"]
                title = bm25search[i][0]["title"]
                score = bm25search[i][1]
                print(f"{i + 1}. ({doc_id}) {title} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
