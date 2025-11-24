#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model, 
    embed_text, 
    verify_embeddings,
    embed_query_text,
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a single text")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    subparsers.add_parser("verify_embeddings", help="Verify the embeddings for the movie dataset")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a query")
    embed_query_parser.add_argument("query", help="Query to be embedded")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            print(f"Embedding text '{args.text}'")
            embed_text(args.text)
        case "verify_embeddings":
            print(f"Verifying embeddings")
            verify_embeddings()
        case "embedquery":
            print(f"embedding query: '{args.query}'")
            embed_query_text(args.query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()