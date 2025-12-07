#!/usr/bin/env python3
import argparse
import json
from lib.keyword_search import search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            matching = search_command(args.query)
            for i in range(len(matching)):
                print(f"{i + 1}: {matching[i]["title"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
