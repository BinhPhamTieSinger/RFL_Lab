# Curating research history

Edit `frontend/research/history-manifest.json` by hand. Each entry selects a real research-output folder relative to RFL_Lab. Optional `title` and Markdown `notes` fields provide a human-written description.

```json
[
  {
    "source": "outputs/example_run_s42",
    "title": "Reviewed experiment",
    "notes": "Describe the hypothesis, findings and caveats here."
  }
]
```

Add or remove entries intentionally. The existing research selection has been preserved. A new output folder is published only after its path is added to the manifest. Metrics and configuration are read from that folder during the build. Push the updated research files and manifest to GitHub to publish the changes. The open browser refreshes history every 15 minutes.

Visualization runs have a separate lifecycle. Python workbench runs stay in the ignored local visualizer data folder. The checked-in captures under `frontend/research/captures` supply educational playback of actual executions for each model. They are excluded from research history. A capture is consumed through `/api/live/captures`; curated research entries use `/api/history`.

To produce a fresh capture, run `python visualizer/scripts/capture_rounds.py METHOD` with the research virtual environment. This executes one CPU round using a small, real CIFAR-100 partition. Capture collection does not edit the history manifest.

Each result has a Configuration link and Copy configuration link button. A URL with `?experiment=archive-ID&section=config` opens that experiment and expands its saved configuration, even when the entry is beyond the first 25 results. The table preserves the original saved device and values. Load configuration creates an editable CPU draft for a new visualization. Shared links never launch training. IDs identify curated source folders, so editing a folder’s saved configuration also updates its shared page after deployment.
