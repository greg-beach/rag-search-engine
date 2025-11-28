#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model, 
    embed_text, 
    verify_embeddings,
    embed_query_text,
    semantic_search,
    chunk_text,
    semantic_chunk_text,
    embed_chunks_command
)

from lib.search_utils import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_SEMANTIC_CHUNK_SIZE

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a single text")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    subparsers.add_parser("verify_embeddings", help="Verify the embeddings for the movie dataset")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a query")
    embed_query_parser.add_argument("query", help="Query to be embedded")

    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Break a query into smaller chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Size of each chunk in words")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Number of words to overlap each chunk")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Break a query into smaller chunks")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_SEMANTIC_CHUNK_SIZE, help="Maximum size of each chunk in sentences")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Number of sentences to overlap each chunk")

    subparsers.add_parser("embed_chunks", help="Embed movies utilizing semantic chunking")

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
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()