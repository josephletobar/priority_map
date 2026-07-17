# Priority Map

## Quick run

```bash
priority-map --img-folder D:\Train\Train\query_images --task "Find cars"
```

You can also pass the image folder positionally:

```bash
priority-map D:\Train\Train\query_images --task "Find cars"
```

## Example Commands

Plain image folder, using optical-flow localization:

```bash
priority-map --img-folder D:\Train\Train\query_images --task "Find cars"
```

Image folder with per-frame GPS/pose metadata:

```bash
priority-map --img-folder D:\Train\Train\query_images --gps D:\Train\Train\query.csv --task "Find cars"
```

With an explicit output folder:

```bash
priority-map --img-folder D:\Train\Train\query_images --gps D:\Train\Train\query.csv --output-dir examples\car_search --task "Find cars"
```

With debug windows:

```bash
priority-map --img-folder D:\Train\Train\query_images --gps D:\Train\Train\query.csv --debug --task "Find cars"
```

## Review an existing graph DB

`priority-map-agent` revises priorities in an existing PriorityMap `graph.db` using a freeform update:

```bash
priority-map-agent examples\car_search\graph.db --update "new information for the search"
```

For an older database that does not yet record its original task, provide it once:

```bash
priority-map-agent examples\car_search\graph.db --original-task "original task" --update "new information"
```

With optional camera intrinsics path stored for future localization work:

```bash
priority-map --img-folder D:\Train\Train\query_images --gps D:\Train\Train\query.csv --camera-intrinsics camera.json --task "Find cars"
```

Using OpenAI for scene understanding instead of the default Gemma/OpenRouter path:

```bash
priority-map --img-folder D:\Train\Train\query_images --scene-model gpt-5.4 --task "Find cars"
```

## Notes

Use `--output-dir` to choose an output folder. If omitted, outputs are written under `examples/YYYY-MM-DD_HH-MM-SS`. The CLI saves `video.avi` and `heatmap.avi` by default; use `--debug` to display frames with OpenCV and print debug logs.

Use `--debrief "extra task context"` to add optional context to the task prompt.
Input images are resized to a 640px longest edge by default; use `--max-image-edge 0` to disable resizing.

If `--gps` is provided, frame metadata is matched by the CSV `name` column. Without GPS, the runner uses flow-based localization.

Scene understanding defaults to Gemma through OpenRouter. Use `--scene-model gpt-5.4` for OpenAI, `--scene-model gemma` for the default Gemma model, or set `SCENE_UNDERSTANDING_MODEL` in the environment.

The knowledge-graph visualization uses compact relationship labels proposed by
the scene VLM. Numeric proximity edges are still retained in `graph.db` and
supplied to later scene-understanding calls as spatial context. In the debug
window, press `1` for the coordinate-based spatial MST or `2` for the
force-directed VLM relationship graph. The spatial view is shown by default.

Use `--panoramic` to enable experimental panorama generation. The runner saves a
standard stitched panorama every 10 frames under `<output-dir>/panorama` and a
corresponding heatmap-overlay panorama under `<output-dir>/heat_panorama`.
