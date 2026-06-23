GPT_VISION_PROMPT = """
Analyze this aerial/satellite image and extract scene labels for open-vocabulary localization.

Do not hallucinate or infer what may be in the scene. Only describe what is clearly visible.

Mission objective: {task}

Existing vocabulary for consistency: {vocabulary}
The existing vocabulary contains previously detected labels and their prior relevance scores.
Reuse these labels when they still apply, and keep scores temporally smooth across frames.
Do not abruptly change a score unless the visual evidence clearly changes.
New scores should usually stay close to the previous score for that label.

For each visible, large visually distinct region, output a JSON object with:

- "prompt": 1-3 keyword phrases only, optimized for localization.
  It is acceptable and preferred to reuse the example prompts exactly when appropriate
  (e.g. "road", "field", "forest", "structure, rooftops", "vehicle, car").
  Do not write sentences, explanations, colors, spatial descriptions, or image-specific details.

- "score": search priority score from 0-100 for the mission objective.

  You are deciding where to deploy limited search resources.

  For each category, estimate how valuable it would be to search that category relative
  to the other visible categories in the current scene.

  Think about practical search effectiveness:
  - Where would you search first?
  - Where is the target most likely to be found?
  - Which categories are strongest indicators of the target's presence?
  - Which categories would you deprioritize because they are unlikely to contain the target?

  Score based on real-world likelihood and operational value, not theoretical possibility.

  Use comparative reasoning. If one category is substantially more useful than another
  for accomplishing the mission, its score should be substantially higher.

  A category that directly matches the mission objective should receive a score near 100.

  The score represents search priority, not visual confidence.

Scoring guide:
- 0 = not relevant
- 25 = weak background context
- 50 = possibly useful
- 75 = strongly useful
- 100 = directly mission-critical

Rules:
- Include all major visible categories (roads, buildings, fields, forests, vehicles, etc.)
- Do not include parts or subtypes of another label
- Merge semantically similar categories
- If road, sidewalk, driveway, parking lot, or paved surfaces are present, output only "road"
- If houses or buildings are present, output only "building"
- If dense tree canopy is present, output only "forest"
- If large grassy or agricultural areas are present, output only "field"
- Do not merge forest, field, or building
- Roads are high-priority, always include when visible
- Score categories relative to each other, not independently
- Prioritize probability over possibility
- Ask: "If I could only search a few areas, where would I send my team first?"
- Categories that directly correspond to the mission objective should score near 100

Return exactly one valid JSON object with double-quoted keys/strings, no trailing commas, no markdown.

Example schema, use only if the labels are present in the image:
{{
    "forest": {{"prompt": "forest", "score": 10}},
    "field": {{"prompt": "field", "score": 20}},
    "road": {{"prompt": "road", "score": 90}},
    "building": {{"prompt": "structure, rooftops", "score": 70}},
    "vehicle": {{"prompt": "vehicle, car", "score": 100}}
}}
"""