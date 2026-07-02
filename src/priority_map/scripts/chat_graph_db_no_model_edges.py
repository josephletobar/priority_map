import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-5.4"


def _number(value, default=0.0):
    if value is None:
        return default

    if isinstance(value, bytes):
        for dtype in (np.float64, np.float32, np.int64, np.int32):
            try:
                decoded = np.frombuffer(value, dtype=dtype)
                if decoded.size:
                    return float(decoded[0])
            except Exception:
                pass
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _table_exists(cursor, table_name):
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _table_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def _required_columns(cursor, table_name, required):
    columns = _table_columns(cursor, table_name)
    missing = sorted(set(required) - columns)
    if missing:
        raise ValueError(
            f"Table {table_name!r} is missing required column(s): "
            f"{', '.join(missing)}"
        )
    return columns


def _connect_read_only(db_path):
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Graph DB not found: {db_path}")

    return sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)


def load_graph_db_without_model_edges(db_path):
    conn = _connect_read_only(db_path)
    try:
        cursor = conn.cursor()

        if not _table_exists(cursor, "nodes"):
            raise ValueError("Graph DB does not contain a nodes table")

        node_columns = _required_columns(
            cursor,
            "nodes",
            ["id", "label", "score", "count", "geo_pos_x", "geo_pos_y"],
        )
        reasoning_select = "reasoning" if "reasoning" in node_columns else "''"
        cursor.execute(
            f"""
            SELECT id, label, score, count, geo_pos_x, geo_pos_y, {reasoning_select}
            FROM nodes
            ORDER BY rowid
            """
        )
        nodes = [
            {
                "id": node_id,
                "label": label,
                "score": _number(score),
                "count": int(_number(count, default=1)),
                "position": {
                    "x": _number(x),
                    "y": _number(y),
                },
                "reasoning": reasoning or "",
            }
            for node_id, label, score, count, x, y, reasoning in cursor.fetchall()
        ]

        edges = []
        if _table_exists(cursor, "edges"):
            _required_columns(cursor, "edges", ["source_id", "target_id", "weight"])
            cursor.execute(
                """
                SELECT source_id, target_id, weight
                FROM edges
                ORDER BY source_id, target_id
                """
            )
            edges = [
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "weight": _number(weight),
                }
                for source_id, target_id, weight in cursor.fetchall()
            ]

        return {
            "nodes": nodes,
            "edges": edges,
        }
    finally:
        conn.close()


def build_prompt(graph_data, question):
    graph_json = json.dumps(graph_data, indent=2)
    return f"""
You are answering questions about a saved drone heatmap graph.

Graph JSON:
{graph_json}

User question:
{question}

Graph meaning:
- nodes are detected or clustered map regions.
- node id is the unique region id.
- label is the localizable category.
- score is relevance to the mission objective at the time it was saved.
- reasoning explains why the perception system chose the label and score.
- edges are numeric spatial/proximity edges.
- position is an image/map coordinate, not a real-world GPS coordinate.

Answer using only the graph data. If the graph does not contain enough evidence,
say what is missing. Do not invent detections, coordinates, semantic text edges,
or relationships that are not supported by the graph.
""".strip()


def ask_graph(client, graph_data, question):
    response = client.responses.create(
        model=MODEL,
        input=build_prompt(graph_data, question),
    )
    return response.output_text


def run_chat(db_path):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    graph_data = load_graph_db_without_model_edges(db_path)

    print("\nGraph DB chat without model edges started. Type 'quit' or 'exit' to leave.")
    print(
        f"Loaded {len(graph_data['nodes'])} node(s) and "
        f"{len(graph_data['edges'])} spatial edge(s)."
    )
    print("Model/freeform text edges are intentionally omitted.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        print("\nAssistant:")
        print(ask_graph(client, graph_data, question))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Chat with an LLM about a saved graph.db file while omitting "
            "freeform model_edges."
        )
    )
    parser.add_argument("db_path", help="Path to a saved graph.db SQLite file.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_chat(args.db_path)


if __name__ == "__main__":
    main()
