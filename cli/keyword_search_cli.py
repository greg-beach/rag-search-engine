#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    search_command, 
    build_command, 
    tf_command, 
    idf_command, 
    tf_idf_command, 
    bm25_idf_command, 
    bm25_tf_command,
    bm25_search_command,
)
from lib.search_utils import BM25_K1, BM25_B

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency for a given document ID and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency of")

    idf_parser = subparsers.add_parser("idf", help="Get the Inverse Document Frequency of a term")
    idf_parser.add_argument("term", type=str, help="Term to get the inverse frequency of")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IIDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the tfidf value of a given document ID and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get tfidf of") 

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully")
        case "tf":
            print("Finding search term frequency")
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in doc ID# '{args.doc_id}': {tf}")
        case "idf":
            print("Finding the IDF")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "bm25idf":
            print("Finding BM25 IDF")
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            print("Finding BM25 TF")
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "tfidf":
            print("Finding the TFIDF")
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25search":
            print(f"Searching for: {args.query}")
            results = bm25_search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()