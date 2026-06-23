GPT_VISION_PROMPT = """
Analyze this aerial/satellite image and extract scene labels for open-vocabulary localization.

Do not hallucinate. Only describe categories that are clearly visible.

Mission objective: {task}

Existing vocabulary for consistency: {vocabulary}
The existing vocabulary contains previously detected labels and their prior relevance scores.
Reuse labels when appropriate and keep scores reasonably stable across similar frames.
Do not make large score changes unless the scene meaningfully changes.

For each visible category, output:

- "prompt": 1-3 keyword phrases optimized for localization.
  Prefer simple reusable prompts such as:
  "road", "field", "forest", "structure, rooftops", "vehicle, car".
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
- 100 = directly mission-critical

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
    "building": {{"prompt": "structure, rooftops", "score": 70}},
    "vehicle": {{"prompt": "vehicle, car", "score": 100}}
}}
"""
