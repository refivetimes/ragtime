#!/usr/bin/env python3
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command
from lib.search_config import DEFAULT_SEARCH_LIMIT

import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model")
    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for text")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for query")
    search_parser = subparsers.add_parser("search", help="Search by query")

    embed_text_parser.add_argument("text", type=str, help="Text to embed")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
    search_parser.add_argument("query", type=str, help="Query to search on")
    search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit number of search results")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()