# Priority Map

## Quick run

```bash
priority-map "a folder of images of video frames"
```

Use `--output-dir` to choose an output folder. If omitted, outputs are written under `examples/YYYY-MM-DD_HH-MM-SS`. The CLI saves `video.avi` and `heatmap.avi` by default; use `--show` to also display frames with OpenCV.

Use `--debrief "extra task context"` to add optional context to the task prompt.
Input images are resized to a 640px longest edge by default; use `--max-image-edge 0` to disable resizing.

Panoramic images output into the selected output folder, respectively for heatmap and standard images.
