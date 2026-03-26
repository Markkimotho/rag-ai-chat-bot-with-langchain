"""CLI script to query the knowledge base (no UI)."""

import argparse

from app.chain import invoke as chain_invoke
from app.graph import invoke as graph_invoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG knowledge base")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--mode", choices=["chain", "graph"], default="chain")
    parser.add_argument("--session-id", default="cli")
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "graph":
        result = graph_invoke(
            question=args.question,
            thread_id=args.session_id,
            model=args.model,
            top_k=args.top_k,
        )
    else:
        result = chain_invoke(
            question=args.question,
            session_id=args.session_id,
            model=args.model,
            top_k=args.top_k,
        )

    print("\n" + result["answer"])
    if result.get("sources"):
        print("\nSources:")
        seen = set()
        for s in result["sources"]:
            key = (s["source"], s["page"])
            if key not in seen:
                seen.add(key)
                print(f"  - {s['source']}, page {s['page']}")


if __name__ == "__main__":
    main()
