GPT_VISION_PROMPT = """
Analyze this image and extract scene labels for open-vocabulary localization.

Do not hallucinate. Only describe categories that are clearly visible.

Mission objective: {task}

Existing vocabulary for consistency: {vocabulary}
The existing vocabulary contains previously detected labels and their prior relevance scores.
Reuse labels when appropriate and keep scores reasonably stable across similar frames.
Do not reuse if the new observation is uniquely distinct. For example, forest and trees are different
Do not make large score changes unless the scene meaningfully changes such that new evidence justifies it.

Recent graph context from previous frames: {recent_graph_context}
This contains recent map nodes, numeric spatial edges, and VLM-written freeform
edges connected to those nodes. Use it as prior spatial and semantic context for
continuity and score stability, but do not treat it as proof that anything is
visible in the current image.
When assigning current priority scores, reason about whether the recent graph
context changes the likely broader environment, nearby category relationships,
or mission relevance of the current scene labels.


For each visible category, output:

- A top-level JSON key that is a simple localization label.
  Use a short, reusable common noun or noun phrase.
  Do not use sentences, colors, locations, counts, task names, search phrases, or
  image-specific descriptions as labels.
  If a label comes from the mission objective, use only the concrete target category,
  not the full mission wording.

- "score": relevance score from 0-100 for the mission objective.

  Consider the entire scene and how visible categories relate to one another spatially.
  Score categories based on how useful they are for accomplishing the mission objective.
  Prioritize practical likelihood over theoretical possibility.
  The score represents relevance to the mission objective, not visual confidence.

  Prioritize distinct, localized evidence of the target over common scene clutter;
  assign low priority to widespread background elements unless they contain specific target-like cues.

- "edges": optional relationships originating from this label.
  Use "to_label" to connect to another label in this response, or "to_node_id"
  to connect to an existing node from recent graph context. The "text" must be a
  concise relationship label containing only one or two lowercase words joined
  by a single underscore. Do not put explanations, evidence, or full sentences
  in edge text.
  Model edges must complement the numeric coordinate graph rather than translate
  it into words. Do not create an edge merely because two entities are nearby,
  adjacent, co-located, separated by some distance, or positioned in a particular
  direction; those facts are already recoverable from their coordinates.
  Create an edge only when the image supports a useful relationship that requires
  interpreting how the entities affect, constrain, enable, obscure, organize, or
  otherwise meaningfully relate to one another. The relationship must add
  mission-relevant information that the node labels and coordinates do not already
  provide.
  Return an empty list when no such relationship is clearly supported.
    
Scoring guide:
- 0 = not relevant 
- 25 = weak context
- 50 = useful
- 75 = highly useful 
- 100 = reserved exclusively for a clearly visible instance of the exact goal object itself. Never assign 100 to context, clues, proxies, containers, locations, related objects, or other mission-critical evidence.

Rules:
- Include all major visible categories
- Merge categories only when they are visually and functionally redundant for the mission
- Keep related categories separate when they have distinct appearance, access patterns, or likely mission relevance
- Use a parent category only when the distinction between subcategories would not change localization or scoring
- Prefer stable reusable labels over frame-specific labels
- Penalize broad background categories unless they provide task-specific search value
- Reward categories that directly match the mission target or strongly constrain where it is likely to be
- Score categories relative to each other
- Prioritize probability over possibility
- Use global spatial context when assigning scores, including adjacency, enclosure, isolation, clustering, and scene layout
- Use recent graph context when it changes likely broader environment, continuity, or score stability
- Edge relationship types are freeform, but their text must use the compact one-
  or two-word label format described above
- Never use model edges as natural-language restatements of coordinate geometry

Return exactly one valid JSON object with double-quoted keys and strings, no trailing commas, and no markdown.

Placeholder schema only:
{{
    "labels": {{
        "<simple_localization_label>": {{
            "score": <integer_0_to_100>,
            "edges": [
                {{"to_label": "<another_current_label>", "text": "<compact_relationship_label>"}},
                {{"to_node_id": "<recent_graph_node_id>", "text": "<compact_relationship_label>"}}
            ]
        }}
    }}
}}
"""

GRAPH_AGENT_QA_PROMPT = """You answer questions about a PriorityMap knowledge graph.

Original task: {original_task}
Question: {question}

Use the graph data and any attached visuals to answer the question directly.
The graph JSON contains:
- "nodes": detected objects with labels, relevance scores, and observation state.
- "edges": numeric spatial relationships between nodes.
- "model_edges": VLM-created textual relationships between nodes.
- "visuals": image indexes mapped to node IDs. Images are attached in that exact
  index order, and each image is immediately preceded by a label identifying its
  table, node ID, and visual type. Masks are segmentation silhouettes; frames are
  source images.

Use spatial edges for proximity and model edges for semantic relationships. Do not
invent facts that are not supported by the graph or visuals. If the information is
insufficient, say so clearly. Return only a concise natural-language answer.

Graph data:
{nodes_text}
"""

# Backward-compatible import name for callers that imported the old prompt constant.
GRAPH_AGENT_PROMPT = GRAPH_AGENT_QA_PROMPT
