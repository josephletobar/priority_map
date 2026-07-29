# Priority Map

## Quick run

```bash
priority-map --img-folder D:\Train\Train\query_images --task "Find cars"
```

You can also pass the image folder positionally:

```bash
priority-map D:\Train\Train\query_images --task "Find cars"
```

## Process Frames as They Arrive

`PriorityMapRunner` can be initialized without an image folder and reused for
in-memory video frames:

```python
from priority_map.runner import PriorityMapRunner

runner = PriorityMapRunner(task="Find cars", record=False)
try:
    while video_is_running:
        image = get_next_video_frame()  # NumPy BGR, BGRA, or grayscale image
        frame_result = runner.run_frame(image)
        send_direction(frame_result.direction)
finally:
    runner.close()
```

An image path can be supplied instead:

```python
frame_result = runner.run_frame("incoming/frame_001.png")
```

Optional per-frame values such as `image_name`, `frame_index`, `easting`,
`northing`, `altitude`, and `orientation` can be passed as keyword arguments.
Calling `run_frame()` without an image retains the original behavior and reads
the next image from the configured folder.

Each `PriorityFrameResult` includes:

- `numerical_heatmap`: the `0..100` scalar field after dilation and Gaussian
  blur, before colorization.
- `heatmap_only`: the JET-colored version used for display.
- `direction`: a two-element unit vector pointing from the image center toward
  the strongest heatmap region. Regions are ranked by total heat, so both their
  size and intensity matter.
- `came_from`: a two-element unit vector pointing back toward the previous
  drone position. It uses consecutive GPS poses when available and otherwise
  falls back to optical flow.

Direction vectors use navigation coordinates: positive X points right and
positive Y points up/forward. An empty heatmap or centered target region returns
`[0.0, 0.0]`. This is a visual navigation suggestion, not a flight-control
command.

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
Debug output draws both vectors from the scene center: the heatmap direction
in white and `came_from` in magenta.

Use `--debrief "extra task context"` to add optional context to the task prompt.
Input images are resized to a 640px longest edge by default; use `--max-image-edge 0` to disable resizing.

If `--gps` is provided, frame metadata is matched by the CSV `name` column.
GPS is preferred for both object localization and the `came_from` vector;
motion falls back to optical flow until a valid consecutive GPS delta and
orientation are available.

Scene understanding defaults to Gemma through OpenRouter. Use `--scene-model gpt-5.4` for OpenAI, `--scene-model gemma` for the default Gemma model, or set `SCENE_UNDERSTANDING_MODEL` in the environment.

The knowledge-graph visualization uses compact relationship labels proposed by
the scene VLM. Numeric proximity edges are still retained in `graph.db` and
supplied to later scene-understanding calls as spatial context. In the debug
window, press `1` for the coordinate-based spatial MST or `2` for the
force-directed VLM relationship graph. The spatial view is shown by default.

Use `--panoramic` to enable experimental panorama generation. The runner saves a
standard stitched panorama every 10 frames under `<output-dir>/panorama` and a
corresponding heatmap-overlay panorama under `<output-dir>/heat_panorama`.
