import argparse

from priority_map.modules.GraphAgent import ask_priority_map_db


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ask a question about a PriorityMap graph database.")
    parser.add_argument("db_path", help="Path to a PriorityMap graph.db file.")
    parser.add_argument("--question", required=True, help="Question to answer using the graph database.")
    parser.add_argument(
        "--scene-model",
        required=True,
        metavar="PROVIDER:MODEL",
        help=(
            "Scene VLM provider and model. Supported providers are "
            "openai, openrouter, and ollama."
        ),
    )
    parser.add_argument("--original-task", help="Backfill the original task for an older database.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = ask_priority_map_db(
        args.db_path,
        args.question,
        original_task=args.original_task,
        debug=args.debug,
        scene_model=args.scene_model,
    )
    print(result["answer"])
    return result


if __name__ == "__main__":
    main()
