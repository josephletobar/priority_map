# Priority Map

## Quick run example

```bash
priority-map --img-folder path\to\images --scene-model provider:model --task "your task"
```

## Complete CLI reference

### `priority-map`

```text
priority-map [IMAGE_FOLDER] --scene-model PROVIDER:MODEL [OPTIONS]
```

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `IMAGE_FOLDER` | No | None | Positional folder containing the input images. |
| `--img-folder PATH` | No | None | Input-image folder; overrides positional `IMAGE_FOLDER`. |
| `--scene-model PROVIDER:MODEL` | Yes | None | Scene VLM provider and provider-owned model identifier. Supported providers are `openai`, `openrouter`, and `ollama`. |
| `--task TEXT` | No | `Find cars` | Mission objective used to score scene relevance. |
| `--debrief TEXT` | No | None | Additional mission context appended to the task for scene understanding. |
| `--gps PATH`, `--gps-csv PATH` | No | None | Per-frame GPS/pose CSV whose `name` column matches image filenames. |
| `--camera-intrinsics PATH` | No | None | Camera-intrinsics file retained by the runner for future localization work; currently unused. |
| `--output-dir PATH` | No | `examples/YYYY-MM-DD_HH-MM-SS` | Directory for videos, heatmaps, observations, and `graph.db`. |
| `--mask [LABEL ...]` | No | Empty | Include side previews for the listed segmentation labels. Matching is case-insensitive. |
| `--sam-step INTEGER` | No | `60` | Run scene understanding and fresh SAM segmentation every Nth frame. |
| `--sam-thresh FLOAT` | No | `0.25` | SAM prediction confidence threshold. |
| `--sam-model-path PATH` | No | `models/sam3.pt` | Path to the SAM model weights. |
| `--blur-spread FLOAT` | No | `101.0` | Base heatmap blur/spread amount; larger values produce broader, smoother heat. |
| `--dilation-scale FLOAT` | No | `1.0` | Multiplier for heatmap dilation; `1.0` preserves the standard behavior. |
| `--max-image-edge INTEGER` | No | `640` | Resize inputs so their longest edge does not exceed this value; use `0` to disable resizing. |
| `--debug` | No | Off | Show diagnostic composite and SAM windows instead of the single normal heatmap preview, and print debug timing/output. |
| `--panoramic` | No | Off | Enable experimental panorama and heatmap-panorama generation. |
| `-h`, `--help` | No | — | Print the CLI help and exit. |

### `priority-map-agent`

```text
priority-map-agent DB_PATH --update TEXT [OPTIONS]
```

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `DB_PATH` | Yes | None | Path to an existing PriorityMap `graph.db`. |
| `--update TEXT` | Yes | None | Freeform task update or additional context used to reprioritize graph nodes. |
| `--original-task TEXT` | No | Stored database value | Backfill the original mission task when reviewing an older database. |
| `--debug` | No | Off | Print model and graph-agent debugging details. |
| `-h`, `--help` | No | — | Print the CLI help and exit. |

## Process Frames as They Arrive

Using this for live autonomy

`PriorityMapRunner` can be initialized without an image folder and reused for
in-memory video frames:

```python
from priority_map.runner import PriorityMapRunner

runner = PriorityMapRunner(
    task="Find cars",
    scene_model="openai:gpt-5.4",
    record=False,
)
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
  size and intensity matter. Before selecting a region, directions toward
  `came_from` and up to 10 nearby graph nodes are masked from the search.
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
priority-map --img-folder D:\Train\Train\query_images --scene-model openai:gpt-5.4 --task "Find cars"
```

Image folder with per-frame GPS/pose metadata:

```bash
priority-map --img-folder D:\Train\Train\query_images --gps D:\Train\Train\query.csv --scene-model openai:gpt-5.4 --task "Find cars"
```

With an explicit output folder:

```bash
priority-map --img-folder D:\Train\Train\query_images --gps D:\Train\Train\query.csv --output-dir examples\car_search --scene-model openai:gpt-5.4 --task "Find cars"
```

With debug windows:

```bash
priority-map --img-folder D:\Train\Train\query_images --gps D:\Train\Train\query.csv --scene-model openai:gpt-5.4 --debug --task "Find cars"
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

## Scene-understanding providers

`--scene-model` is required and uses `provider:model` format. The supported
providers are `openai`, `openrouter`, and `ollama`:

```bash
priority-map --img-folder D:\Train\Train\query_images --scene-model openai:gpt-5.4 --task "Find cars"
priority-map --img-folder D:\Train\Train\query_images --scene-model openrouter:google/gemma-4-31b-it --task "Find cars"
```

For local inference, start Ollama and ensure the model you want to use is
installed on that device, then select it with the same format:

```bash
ollama pull YOUR_VISION_MODEL
priority-map --img-folder D:\Train\Train\query_images --scene-model ollama:YOUR_VISION_MODEL --task "Find cars"
```

The application validates the provider name only. The model identifier is sent
unchanged to that provider, which determines whether the model exists and can
accept image input. Ollama requests use `http://localhost:11434/v1`.

## Notes

Use `--output-dir` to choose an output folder. If omitted, outputs are written under `examples/YYYY-MM-DD_HH-MM-SS`. The CLI saves `video.avi` and `heatmap.avi` by default. Normal runs show one live `Priority Heatmap` window as frames are processed; press `q` or Escape to stop. Use `--debug` to replace that single preview with the diagnostic OpenCV windows and print debug logs.
Debug output draws both vectors from the scene center: the heatmap direction
in white and `came_from` in magenta.

Use `--debrief "extra task context"` to add optional context to the task prompt.
Input images are resized to a 640px longest edge by default; use `--max-image-edge 0` to disable resizing.

If `--gps` is provided, frame metadata is matched by the CSV `name` column.
GPS is preferred for both object localization and the `came_from` vector;
motion falls back to optical flow until a valid consecutive GPS delta and
orientation are available.

The knowledge-graph visualization uses compact relationship labels proposed by
the scene VLM. Numeric proximity edges are still retained in `graph.db` and
supplied to later scene-understanding calls as spatial context. In the debug
window, press `1` for the coordinate-based spatial MST or `2` for the
force-directed VLM relationship graph. The spatial view is shown by default.

Experimental: Use `--panoramic` to enable experimental panorama generation. The runner saves a
standard stitched panorama every 10 frames under `<output-dir>/panorama` and a
corresponding heatmap-overlay panorama under `<output-dir>/heat_panorama`.
