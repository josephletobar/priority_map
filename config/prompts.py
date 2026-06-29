GPT_VISION_PROMPT = """
Analyze this aerial/satellite image and extract scene labels for open-vocabulary localization.

Do not hallucinate. Only describe categories that are clearly visible.

Mission objective: {task}

Existing vocabulary for consistency: {vocabulary}
The existing vocabulary contains previously detected labels and their prior relevance scores.
Reuse labels when appropriate and keep scores reasonably stable across similar frames.
Do not reuse if the new observation is uniquely distinct.
Do not make large score changes unless the scene meaningfully changes.

For each visible category, output:

- "prompt": a keyword phrase optimized for localization.
  Prefer simple reusable prompts such as:
  "road", "field", "forest", "rooftops", "vehicle".
  Do not output sentences, colors, locations, or image-specific descriptions.

- "score": relevance score from 0-100 for the mission objective.

  Consider the entire scene and how visible categories relate to one another.
  Score categories based on how useful they are for accomplishing the mission objective.
  Prioritize practical likelihood over theoretical possibility.
  The score represents relevance to the mission objective, not visual confidence.

Scoring guide:
- 0 = not relevant
- 25 = weak context
- 50 = useful
- 75 = highly useful
- 100 = directly mission-critical, or the task itself

Rules:
- Include all major visible categories
- Merge semantically similar categories
- Do not include subtypes of another category
- If road, sidewalk, driveway, parking lot, or paved surfaces are present, output only "road"
- If houses or buildings are present, output only "building"
- If dense tree canopy is present, output only "forest"
- If large grassy or agricultural areas are present, output only "field"
- Score categories relative to each other
- Prioritize probability over possibility
- Use global scene context when assigning scores

Return exactly one valid JSON object with double-quoted keys and strings, no trailing commas, and no markdown.

Example schema:
{{
    "forest": {{"prompt": "forest", "score": 10}},
    "field": {{"prompt": "field", "score": 20}},
    "road": {{"prompt": "road", "score": 90}},
    "building": {{"prompt": "rooftops", "score": 70}},
    "vehicle": {{"prompt": "vehicle", "score": 100}}
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
- Follow paths through the graph: even if field->building->road are not all directly connected, they are in the same connected component and form a spatial cluster.

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
