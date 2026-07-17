import argparse

from priority_map.modules.GraphAgent import review_priority_map_db


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Review and reprioritize a PriorityMap graph database.")
    parser.add_argument("db_path", help="Path to a PriorityMap graph.db file.")
    parser.add_argument("--update", required=True, help="Freeform task update or additional context.")
    parser.add_argument("--original-task", help="Backfill the original task for an older database.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = review_priority_map_db(
        args.db_path,
        args.update,
        original_task=args.original_task,
        debug=args.debug,
    )
    print(f"Updated {len(result['changes'])} node(s) in {result['db_path']}")
    if result["reasoning"]:
        print(result["reasoning"])
    for node_id, old_score, new_score in result["changes"]:
        print(f"{node_id}: {old_score:.0f} -> {new_score:.0f}")
    return result


if __name__ == "__main__":
    main()
