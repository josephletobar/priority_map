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
This contains recent map nodes and edges connected to those nodes. Use it as prior
spatial and semantic context for continuity and score stability, but do not treat
it as proof that anything is visible in the current image.
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

- "reasoning": a brief interpretability explanation for why you chose this label.
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

Return exactly one valid JSON object with double-quoted keys and strings, no trailing commas, and no markdown.

Placeholder schema only:
{{
    "<simple_localization_label>": {{
        "reasoning": "<interpretability explanation written before choosing the score>",
        "score": <integer_0_to_100>
    }}
}}
"""

GRAPH_AGENT_PROMPT = """You are analyzing aerial drone footage to help with a search task.

Task: {task_description}

You have a spatial graph of detected objects. Scores represent relevance to the task. Objects are connected by edges when they are spatially close.

Spatial graph ("nodes" list each detected object with current relevance score 0-100; "edges" are MST edges where "dist" is distance between nodes):
{nodes_text}

The existing scores were local perception scores only. They do not account for graph structure, proximity, clusters, or task context. Your job is to add global-context corrections.

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

