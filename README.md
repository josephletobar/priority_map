# Priority Map

## Quick run example

```bash
priority-map --img-folder path\to\images --scene-model provider:model --sam-model-path path\to\sam3.pt --task "your task"
```

## Complete CLI reference

### `priority-map`

```text
priority-map --img-folder PATH --scene-model PROVIDER:MODEL --sam-model-path PATH [OPTIONS]
```

#### Required arguments

| Argument | Description |
| --- | --- |
| `--img-folder PATH` | Folder containing the input images. |
| `--scene-model PROVIDER:MODEL` | Scene VLM provider and provider-owned model identifier. Supported providers are `openai`, `openrouter`, and `ollama`. |
| `--sam-model-path PATH` | Path folder that SAM model weights. `sam3.pt` |

#### Optional arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--task TEXT` | `Find cars` | Mission objective used to score scene relevance. |
| `--debrief TEXT` | None | Additional mission context appended to the task for scene understanding. |
| `--gps PATH`, `--gps-csv PATH` | None | Per-frame GPS/pose CSV whose `name` column matches image filenames. |
| `--camera-intrinsics PATH` | None | Camera-intrinsics file retained by the runner for future localization work; currently unused. |
| `--output-dir PATH` | `examples/YYYY-MM-DD_HH-MM-SS` | Directory for videos, heatmaps, observations, and `graph.db`. |
| `--sam-step INTEGER` | `60` | Run scene understanding and fresh SAM segmentation every Nth frame. |
| `--sam-thresh FLOAT` | `0.25` | SAM prediction confidence threshold. |
| `--blur-spread FLOAT` | `101.0` | Base heatmap blur/spread amount; larger values produce broader, smoother heat. |
| `--dilation-scale FLOAT` | `1.0` | Multiplier for heatmap dilation; `1.0` preserves the standard behavior. |
| `--max-image-edge INTEGER` | `640` | Resize inputs so their longest edge does not exceed this value; use `0` to disable resizing. |
| `--debug` | Off | Show diagnostic composite and SAM windows instead of the single normal heatmap preview, and print debug timing/output. |
| `--panoramic` | Off | Enable experimental panorama and heatmap-panorama generation. |
| `-h`, `--help` | — | Print the CLI help and exit. |


### `priority-map-agent`

```text
priority-map-agent DB_PATH --update TEXT [OPTIONS]
```

#### Required arguments

| Argument | Description |
| --- | --- |
| `DB_PATH` | Path to an existing PriorityMap `graph.db`. |
| `--update TEXT` | Freeform task update or additional context used to reprioritize graph nodes. |

#### Optional arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--original-task TEXT` | Stored database value | Backfill the original mission task when reviewing an older database. |
| `--debug` | Off | Print model and graph-agent debugging details. |
| `-h`, `--help` | — | Print the CLI help and exit. |

## Process Frames as They Arrive

Using this for live autonomy

`PriorityMapRunner` can be initialized without an image folder and reused for
in-memory video frames. Its `sam_model_path` argument is required:

```python
from priority_map.runner import PriorityMapRunner

runner = PriorityMapRunner(
    image_folder=None,
    task="Find cars",
    scene_model="openai:gpt-5.4",
    sam_model_path="models/sam3.pt",
    direction_ema_alpha_min=0.1,
    direction_ema_alpha_max=0.8,
    max_observed_coverage_ratio=0.25,
    coverage_lookahead_seconds=2.0,
    record=False,
)
try:
    while video_is_running:
        image = get_next_video_frame()  # NumPy BGR, BGRA, or grayscale image
        frame_result = runner.run_frame(
            image,
            speed_mps=current_speed_mps,
            cesium_metadata=current_camera_metadata,
        )
        send_direction(frame_result.direction)
finally:
    runner.close()
```

An image path can be supplied instead:

```python
frame_result = runner.run_frame("incoming/frame_001.png")
```

Optional per-frame values such as `image_name`, `frame_index`, `easting`,
`northing`, `altitude`, `orientation`, and `speed_mps` can be passed as keyword
arguments.
Calling `run_frame()` without an image retains the original behavior and reads
the next image from the configured folder.

Each `PriorityFrameResult` includes:

- `numerical_heatmap`: the `0..100` scalar field after dilation and Gaussian
  blur, before colorization.
- `heatmap_only`: the JET-colored version used for display.
- `direction`: a two-element unit vector pointing from the image center toward
  the strongest heatmap region. Regions are ranked by total heat, so both their
  size and intensity matter. Before selecting a region, the cone around
  `came_from` is masked from the search.
- `came_from`: a two-element unit vector pointing back toward the previous
  drone position. It uses consecutive GPS poses when available and otherwise
  falls back to optical flow.

Direction vectors use navigation coordinates: positive X points right and
positive Y points up/forward. An empty heatmap or centered target region returns
`[0.0, 0.0]`. This is a visual navigation suggestion, not a flight-control
command.

With valid Cesium metadata and `speed_mps`, georeferenced graph nodes that have
left the camera view are marked `observed`. Heat-ranked candidates are previewed
through the direction EMA, then projected over `coverage_lookahead_seconds` and
avoided when those regions cover more than `max_observed_coverage_ratio` of any
sampled view. If every candidate is blocked, the hottest EMA result is used.

Direction EMA alpha scales linearly between `direction_ema_alpha_min` and
`direction_ema_alpha_max` using the normalized variation of the 5x5 patch-heat
distribution. Uniform heat stays strongly smoothed; concentrated heat responds
more quickly.

## Example Commands

Plain image folder, using optical-flow localization:

```bash
priority-map --img-folder D:\Train\Train\query_images --scene-model openai:gpt-5.4 --sam-model-path models\sam3.pt --task "Find cars"
```

Image folder with per-frame GPS/pose metadata:

```bash
priority-map --img-folder D:\Train\Train\query_images --gps D:\Train\Train\query.csv --scene-model openai:gpt-5.4 --sam-model-path models\sam3.pt --task "Find cars"
```

With an explicit output folder:

```bash
priority-map --img-folder D:\Train\Train\query_images --gps D:\Train\Train\query.csv --output-dir examples\car_search --scene-model openai:gpt-5.4 --sam-model-path models\sam3.pt --task "Find cars"
```

With debug windows:

```bash
priority-map --img-folder D:\Train\Train\query_images --gps D:\Train\Train\query.csv --scene-model openai:gpt-5.4 --sam-model-path models\sam3.pt --debug --task "Find cars"
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
priority-map --img-folder D:\Train\Train\query_images --scene-model openai:gpt-5.4 --sam-model-path models\sam3.pt --task "Find cars"
priority-map --img-folder D:\Train\Train\query_images --scene-model openrouter:google/gemma-4-31b-it --sam-model-path models\sam3.pt --task "Find cars"
```

For local inference, start Ollama and ensure the model you want to use is
installed on that device, then select it with the same format:

```bash
ollama pull YOUR_VISION_MODEL
priority-map --img-folder D:\Train\Train\query_images --scene-model ollama:YOUR_VISION_MODEL --sam-model-path models\sam3.pt --task "Find cars"
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
