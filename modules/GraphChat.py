from openai import OpenAI
import json
from dotenv import load_dotenv
from modules.GraphBuilder import GraphBuilder

# Old project imports for embedding/RAG retrieval. Current heatmap graph only has
# node id, label, score, and simple edges.
# from scripts.get_embedding import embed_text
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import spacy
# nlp = spacy.load("en_core_web_sm")

SIM_THRESHOLD = 0.8

RAG = False

class ChatWithGraph:

    def __init__(self, graph):

        load_dotenv(override=True)

        self.client = OpenAI()

        self.graph = graph

        # Old project graph fields. These are not in the current heatmap graph.
        # self.txt_embeddings = np.array([
        #     attrs["txt_embedding"]
        #     for _, attrs in graph.nodes(data=True)
        # ])
        # self.img_embeddings = np.array([
        #     attrs["img_embedding"]
        #     for _, attrs in graph.nodes(data=True)
        # ])
        
        self.node_ids = [
            node_id
            for node_id, _ in graph.nodes(data=True)
        ]

        self.k = max(3, len(self.node_ids) // 4)

    def rag(self, prompt):
        return self.node_ids

        # Old project RAG path expected txt_embedding/img_embedding on nodes.
        # doc = nlp(prompt)
        # keywords = [
        #     token.text
        #     for token in doc
        #     if token.pos_ in [
        #         "NOUN",
        #         "PROPN",
        #         "VERB",
        #         "ADJ"
        #     ]
        # ]

        # query_embedding = embed_text(" ".join(keywords))

        # txt_score = cosine_similarity(
        #     [query_embedding],
        #     self.txt_embeddings
        # )[0]
        # img_score = cosine_similarity(
        #     [query_embedding],
        #     self.img_embeddings
        # )[0]

        # scores = 0.5 * txt_score + 0.5 * img_score
        # best_idx = np.argsort(scores)[-self.k:]

        # top_nodes = [
        #     self.node_ids[i]
        #     for i in best_idx
        # ]

        # print("\nTop Retrieved:")

        # for i in best_idx[::-1]:
        #     print(
        #         f"{self.node_ids[i]:15s} "
        #         f"txt={txt_score[i]:.4f} "
        #         f"img={img_score[i]:.4f} "
        #         f"final={scores[i]:.4f}"
        #     )

        # return top_nodes

    def chat(self):

        prompt = input("\nYou: ")

        if prompt.lower() in ["quit", "exit"]:
            return

        if RAG: 
            top_nodes = self.rag(prompt)

            expanded = set(top_nodes)
            for node in top_nodes:
                expanded.update(self.graph.neighbors(node))

            subgraph = self.graph.subgraph(expanded).copy()


        else: # use the whole graph with no rag
            subgraph = self.graph.copy()


        # in memory json
        graph_json = json.dumps(
                GraphBuilder.to_serializable_data(subgraph),
                indent=2
            )

        # pprint(f"used graph: {graph_json}")

        # print("Retrieved nodes:")
        # for node in subgraph.nodes():
        #     print(node)

        response = self.client.responses.create(
            model="gpt-5",
            input=f"""
            You are reasoning over a simple drone heatmap graph.

            Graph:
            {graph_json}

            User Question:
            {prompt}

            Current graph meaning:
            - Each node is a detected/merged heatmap region.
            - Node id is the unique region name, like field_0 or trees_0.
            - Node label is the semantic category, like field or trees.
            - Node score is the heat/relevance score.
            - Edges only mean these nodes coexist in the current global graph.
            - There are no real-world coordinates yet.

            Answer using only the graph data.

            If there is insufficient information in the graph to answer confidently, say so.

            Do not invent coordinates, physical destinations, or spatial distances.
            """
        )

        print("\nAssistant:")
        print(response.output_text)
