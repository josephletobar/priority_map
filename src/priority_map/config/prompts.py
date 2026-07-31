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
  provide. Prefer a few high-value edges over a dense graph. Return an empty list
  when no such relationship is clearly supported.

Scoring guide:
- 0 = not relevant to the goal
- 25 = weak context to the goal being present
- 50 = useful to achieving the goal
- 75 = highly useful to achieving the goal
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

GRAPH_AGENT_PROMPT = """You are reviewing a PriorityMap spatial graph.

Original task: {original_task}
New information: {update}

You have a spatial graph of detected objects. Scores represent relevance. Objects are connected by edges when they are spatially close.

Decide how the new information should affect relevance. It may add context, change the objective, or do both. Use your judgment; do not assume a fixed interpretation.

Spatial graph ("nodes" list each detected object with current relevance score 0-100; "edges" are MST edges where "dist" is distance between nodes):
{nodes_text}

The existing scores may not account for the new information, graph structure, proximity, or clusters. Your job is to apply global-context corrections.

A node may appear in multiple edges. Do not interpret repeated edge references as duplicate nodes. Only the nodes list defines unique nodes.

Think about spatial neighborhoods and clusters (connected components):
- Which objects form cohesive spatial groups?
- How do spatial neighborhoods affect the relevance of individual objects?
- Are there patterns where certain spatial configurations make finding your target more likely?
- Follow paths through the graph: even if related node categories are not all directly connected, they can still be in the same connected component and form a spatial cluster.

Consider how global spatial context changes relevance:
- An isolated object has different global relevance than the same object adjacent to certain features
- Objects in the same connected component should influence each other based on what makes that cluster likely to contain your target
- Think about what spatial patterns indicate high probability of finding what you are looking for

Reason over spatial clusters, connected components, and transitive relationships between objects, not just individual object types.

Only return no updates if every node's current score already matches its global task relevance.

Always provide your reasoning and list any score adjustments:

{{
  "reasoning": "Your analysis of spatial clusters and connected components, explaining which patterns led you to adjust scores and why...",
  "updates": [
    {{"node_id": "label_00", "delta": x}},
    {{"node_id": "label_01", "delta": y}}
  ]
}}

delta is an integer from -20 to 20. Positive increases relevance, negative decreases.

If no adjustments needed, still explain the spatial reasoning:
{{
  "reasoning": "Why the current spatial arrangement and connected components already reflect task relevance...",
  "updates": []
}}"""

# Always include the concrete target category from the mission objective with 100 percent relevance, even if it is not directly visible.
# Derive this label by removing mission/action wording and keeping only the localizable object or category being searched for.
# The target label must be a simple noun or noun phrase, not the full task wording, not a search phrase, and not a sentence.
