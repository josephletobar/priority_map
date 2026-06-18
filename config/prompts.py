VLM_PROMPT = """
Generate labels describing localizable landscape/objects from aerial/satellite imagery

Do not hallucinate or infer what may be in the scene. 

Do not explain

Objects must be easily localizable by a segmentation model

For help in object detection, the image is slightly green tinted.
"""

LLM_PROMPT = """
These are the observations in the current scene for you to format: {observation}

Format the observations for open-vocabulary localization from aerial or satellite imagery for the following mission objective: {task}

The "score" field should be a relevance score 0-100 of this area/object in achieving the given task.

Here is the existing vocabulary, along with what you outputted for the previous score of this category, so you remain consistent (unless new observation has changed relevance): {vocabulary}

Return exactly one valid JSON object using double-quoted keys/strings and no trailing commas. No markdown or extra text.

Schema:
{{
    "trees": {{
        "prompt": "dense forest, woodland, tree canopy, or heavily wooded area",
        "score": 0
    }},
    "field": {{
        "prompt": "open field, grassland, meadow, pasture, lawn",
        "score": 30
    }},
    "road": {{
        "prompt": "road, street, or highway",
        "score": 90
    }},
    "building": {{
        "prompt": "building, house, facility",
        "score": 80
    }},
    "vehicle": {{
        "prompt": "vehicle, car, truck, van, or motorized ground transportation",
        "score": 100
    }}
}}
"""

PROMPT_TEMPLATES = [
    "aerial imagery of {}.",
    "aerial imagery of {}s.",
    "an aerial image of {}.",
    "an aerial image of {}s.",
    "a satellite image of {}.",
    "a satellite image of {}s.",
    "an overhead view of {}.",
    "an overhead view of {}s.",
    "{} seen from above.",
    "{}s seen from above.",
    "{} in aerial imagery.",
    "{}s in aerial imagery.",
    "{} in satellite imagery.",
    "{}s in satellite imagery.",
    "there is {} in the aerial scene.",
    "there are {}s in the aerial scene.",
]

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

