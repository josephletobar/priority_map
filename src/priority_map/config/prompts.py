GPT_VISION_PROMPT = """
Analyze this aerial/satellite image and extract scene labels for open-vocabulary localization.

Do not hallucinate. Only describe categories that are clearly visible.

Mission objective: {task}

Existing vocabulary for consistency: {vocabulary}
The existing vocabulary contains previously detected labels and their prior relevance scores.
Reuse labels when appropriate and keep scores reasonably stable across similar frames.
Do not reuse if the new observation is uniquely distinct.
Do not make large score changes unless the scene meaningfully changes such that new evidence justifies it.

Recent graph context from previous frames: {recent_graph_context}
This contains recent map nodes, numeric spatial edges, and model-written freeform
edges connected to those nodes. Use it as prior spatial and semantic context for
continuity and score stability, but do not treat it as proof that anything is
visible in the current image.
When assigning current priority scores, reason about whether the recent graph
context changes the likely broader environment, nearby category relationships,
or mission relevance of the current scene labels.

Always include the concrete target category from the mission objective with 100 percent relevance, even if it is not directly visible.
Derive this label by removing mission/action wording and keeping only the localizable object or category being searched for.
The target label must be a simple noun or noun phrase, not the full task wording, not a search phrase, and not a sentence.

For each visible category, output:

- A top-level JSON key that is a simple localization label.
  Use a short, reusable common noun or noun phrase.
  Do not use sentences, colors, locations, counts, task names, search phrases, or
  image-specific descriptions as labels.
  If a label comes from the mission objective, use only the concrete target category,
  not the full mission wording.

- "reasoning": an interpretability explanation for why you chose this label.
  First consider spatial context: where this category appears in the scene, what
  surrounds it, whether it is isolated or embedded in other categories, and how
  those spatial relationships change its mission relevance. Also consider the
  recent graph context from previous frames: whether recent nearby labels and
  edges suggest a broader environment or pattern that should raise, lower, or
  stabilize this label's priority. Then explain why this category belongs in the
  scene dictionary, how it relates to the mission objective, and what makes it
  more or less useful than the other selected categories. This is not a visual
  confidence explanation. Do not merely say the category is visible.
  Give balanced reasoning: include why the score is justified, and also why it is
  not substantially higher or lower.
  Explicitly state how the recent graph context affected this score, or state that
  it did not materially affect this score.

- "score": relevance score from 0-100 for the mission objective.

  Choose the score only after writing the reasoning. The score should be the
  conclusion of the reasoning, not a separate first impression. In each label
  object, output "reasoning" first and "score" second.

  Consider the entire scene and how visible categories relate to one another spatially.
  Score categories based on how useful they are for accomplishing the mission objective.
  Prioritize practical likelihood over theoretical possibility.
  The score represents relevance to the mission objective, not visual confidence.

  Prioritize distinct, localized evidence of the target over common scene clutter; 
  assign low priority to widespread background elements unless they contain specific target-like cues.

- "edges": optional freeform edge intents from this label.
  Use "to_label" to connect this label to another label in the same response.
  Use "to_node_id" to connect this label to an existing node from recent graph context.
  Each edge must include "text", which is unconstrained natural language describing
  whatever relationship you believe would help future graph reasoning.
  Do not limit edge text to spatial proximity. Edge text may describe any
  relationship type that is useful for the mission or for interpreting the scene,
  including functional, semantic, contextual, causal, affordance-based,
  hierarchical, evidential, uncertainty-related, or abstract relationships.
  Create edges when the relationship adds information beyond the two labels
  existing separately. Leave edges empty when no meaningful relationship is worth
  preserving.

Scoring guide:
- 0 = not relevant
- 25 = weak context
- 50 = useful
- 75 = highly useful
- 100 = the concrete mission target category, or directly mission-critical visible evidence

Rules:
- Include the concrete mission target category even if it is not directly visible
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
- For each label, reason about spatial context and mission relevance first, then output the score after that reasoning
- Each reasoning field must include both positive and limiting factors for the score unless the score is exactly 0 or 100
- Model-created edge text is freeform graph knowledge, not just geometry; use it
  to preserve relationships that may help later reasoning.

Return exactly one valid JSON object with double-quoted keys and strings, no trailing commas, and no markdown.

Placeholder schema only:
{{
    "labels": {{
        "<simple_localization_label>": {{
            "reasoning": "<interpretability explanation written before choosing the score>",
            "score": <integer_0_to_100>,
            "edges": [
                {{"to_label": "<another_current_label>", "text": "<freeform edge text>"}},
                {{"to_node_id": "<recent_graph_node_id>", "text": "<freeform edge text>"}}
            ]
        }}
    }}
}}
"""

GRAPH_AGENT_PROMPT = """You are analyzing aerial drone footage to help with a search task.

Task: {task_description}

You have a graph of detected objects. Scores represent relevance to the task.
The graph includes numeric spatial edges and freeform model-written edges.

Graph JSON ("nodes" list detected objects with current relevance score 0-100 and prior reasoning; "spatial_edges" are numeric distance edges; "model_edges" are freeform model-written relationships):
{nodes_text}

The existing scores were local perception scores only. They do not account for graph structure, proximity, clusters, or task context. Your job is to add global-context corrections.

A node may appear in multiple edges. Do not interpret repeated edge references as duplicate nodes. Only the nodes list defines unique nodes.

Think about graph neighborhoods and clusters (connected components):
- Which objects form cohesive spatial groups?
- How do spatial neighborhoods affect the relevance of individual objects?
- Are there patterns where certain spatial configurations make finding your target more likely?
- Follow paths through the graph: even if related node categories are not all directly connected, they can still be in the same connected component and form a spatial cluster.
- Which freeform relationships would make the graph more useful for future reasoning?

Consider how global spatial context changes relevance:
- An isolated object has different global relevance than the same object adjacent to certain features
- Objects in the same connected component should influence each other based on what makes that cluster likely to contain your target
- Think about what spatial patterns indicate high probability of finding what you are looking for

Reason over spatial clusters, connected components, and transitive relationships between objects, not just individual object types.

Only return no updates if every node's current score already matches its global task relevance.

You may also create freeform model edges between existing node IDs whenever a connection would help future graph reasoning. The edge text is unconstrained; write whatever relationship or purpose you think matters.

Always provide your reasoning, score adjustments, and any freeform edges:

{{
  "reasoning": "Your analysis of spatial clusters and connected components, explaining which patterns led you to adjust scores and why...",
  "updates": [
    {{"node_id": "label_00", "delta": x}},
    {{"node_id": "label_01", "delta": y}}
  ],
  "edges": [
    {{"source_id": "label_00", "target_id": "label_01", "text": "<freeform edge text>"}}
  ]
}}

delta is an integer from -20 to 20. Positive increases relevance, negative decreases.

If no adjustments needed, still explain the spatial reasoning:
{{
  "reasoning": "Why the current spatial arrangement and connected components already reflect task relevance...",
  "updates": [],
  "edges": []
}}"""

