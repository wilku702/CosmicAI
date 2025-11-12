## Chart Preprocessing & Spec-Only Evaluation

This project offers two deterministic utilities for chart critique:

1. **Preprocess** a directory of chart images into overlays + metadata (no GT required).
2. **Evaluate** a single *generated* chart against a textual spec—only the spec and the generated pixels are considered. No ground-truth images or metadata are ever loaded.

---

### Requirements

- Python 3.9+
- `opencv-python`
- `numpy`

Install dependencies:

```bash
python -m pip install opencv-python numpy
```

---

### Preprocessing Command

```bash
python region_split.py preprocess images/ --output preprocess_outputs
```

Per-image artifacts (under `preprocess_outputs/<image>/`):

- `normalized.png`, `enhanced.png`, `binary.png`, `edges.png`, `gradient.png`, `skeleton.png`
- `horizontal_lines.png`, `vertical_lines.png`, `line_map.png`, `component_map.png`
- `observation.json` containing stats, connected-component summaries, dominant palette, and region descriptors

Useful flags: `--limit`, `--max-dim`, `--no-pad`, `--clahe-clip`, `--clahe-grid W H`, `--adaptive-block`, `--adaptive-c`, `--skeleton-kernel`, `--palette-size`, `--quiet`.

---

### Spec-Only Evaluation

Prepare a JSON spec (see `chart_split/spec_schema.py` for the schema), then run:

```bash
python region_split.py eval-gen \
    --spec examples/specs/bar3.json \
    --image outputs/bar3.png \
    --out reports/bar3
```

Outputs:

- `reports/bar3/spec.json` – normalized view of the spec
- `reports/bar3/observation.json` – observed properties derived solely from the generated image
- `reports/bar3/comparison.json` – mismatches, score (0–1), and deltas
- `reports/bar3/vlm_prompt.txt` – ready-to-send instructions for a VLM
- `reports/bar3/overlays/` – same overlays generated during preprocessing

The evaluator infers chart type, legend presence/position, series count, and axis/tick heuristics from the generated image only, then compares them to the textual expectations.

---

### Run Observation Tests on a Folder

Quickly sanity-check every chart in a directory (no specs) and capture inferred properties:

```bash
python region_split.py test-images \
    --images images/ \
    --out reports/image_tests
```

For each image you’ll get `reports/image_tests/<name>/observation.json` plus overlays, and the root `summary.json` aggregates inferred chart types, legend presence, series counts, and axis heuristics.

---

### Programmatic APIs

```python
from chart_split.preprocess import preprocess_for_llm
from chart_split.eval_generated import (
    observe_generated_chart,
    compare_spec_to_observation,
    build_vlm_prompt,
)
from chart_split.spec_schema import ExpectedSpec, from_json_file

artifacts = preprocess_for_llm(image)
observation = observe_generated_chart("gen.png", artifacts=artifacts)
spec = from_json_file("specs/bar3.json")
comparison = compare_spec_to_observation(spec, observation)
prompt = build_vlm_prompt(spec, observation, comparison["issues"])
```

---

### Tests

Run the suite (synthetic fixtures only—no GT assets):

```bash
pytest
```

---

### Notes

- A guardrail prevents refs to historical GT folders (strings like `llm_ready/` or `metadata.json`) when invoking spec-based evaluation inputs.
- Lint/type-check: `python -m py_compile region_split.py chart_split/*.py`.

Happy chart critiquing!
