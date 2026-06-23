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
  (e.g. "road", "field", "trees", "structure, rooftops", "vehicle, car"). 
  Do not write sentences, explanations, colors, spatial descriptions, or image-specific details.
- "score": relevance score 0-100 for the mission objective

Rules:
- Include all major visible categories (roads, buildings, fields, trees, vehicles, etc.)
- Do not include parts or subtypes of another label
- Merge semantically similar categories
- If road, sidewalk, driveway, parking lot, or paved surfaces are present, output only "road"
- If houses or buildings are present, output only "building"
- If dense tree canopy is present, output "forest"
- If large grassy or agricultural areas are present, output "field"
- Do not merge forest, field, or vegetation
- Roads are high-priority, always include when visible

Return exactly one valid JSON object with double-quoted keys/strings, no trailing commas, no markdown.

Example schema, use only if they are present in the image:
{{
    "trees": {{"prompt": "trees", "score": 0}},
    "field": {{"prompt": "field", "score": 30}},
    "road": {{"prompt": "road", "score": 90}},
    "building": {{"prompt": "structure, rooftops", "score": 55}},
    "vehicle": {{"prompt": "vehicle, car", "score": 100}}
}}
"""

# Rules:
# - Labels must correspond to large visually distinct regions.
# - Include all major visible categories, not just the most important ones.
# - Do not include parts or subtypes of another label.
# - Merge semantically similar categories, including if it is already in the vocabulary.
# - If road, sidewalk, driveway, parking lot, or other traversable paved surfaces are present, output only "road".
# - If houses or buildings are present, output only "building".
# - If dense contiguous tree canopy is present, output "forest".
# - If large open grassy or agricultural areas are present, output "field".
# - Do not merge forest, field, or vegetation into a single category.
# - Roads are high-priority categories and should always be included when visible.
# - For each label, generate localization synonyms as a single '+'-separated string.
# - The synonyms string must always start with the canonical label itself.
# - Include a mission relevance score from 0-100.
# - Relevance reflects mission usefulness, not detection confidence.
# - Return exactly one JSON object. The top-level value must be an object (not a list/array), must begin with '{{' and end with '}}', and must contain no markdown, code fences, comments, explanations, or extra text.

