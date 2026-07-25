# Face Authenticity Scanner

Flask web app for the "Detecting Deepfake Faces" course project. On startup it
downloads `deepfake_hybrid_final.onnx` and `deepfake_hybrid_final.onnx.data`
from your public Hugging Face repo, runs inference with ONNX Runtime (CPU),
and serves a browser UI where a user uploads a face photo and gets a
real/fake verdict.

## How it works

- `MODEL_URL` / `MODEL_DATA_URL` (in `.env`) point at the two raw file URLs
  on Hugging Face.
- On first startup, `model_utils.ensure_model_files()` downloads both into
  `MODEL_DIR` (default `models/`), **keeping the original filenames** — this
  matters because ONNX external-data loading resolves `.onnx.data` by name
  relative to the `.onnx` file.
- Subsequent restarts skip the download if the files already exist on disk.
- `/predict` accepts a multipart image upload, resizes it to `IMG_SIZE`
  (260×260), normalizes with ImageNet mean/std, and runs it through the
  model. The model's single sigmoid output is treated as **P(fake)**.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit if needed
python app.py                    # downloads the model on first run, serves on :5000
```

Open http://localhost:5000.

## Configuration (`.env`)

All the knobs live in `.env.example` — copy it to `.env` and adjust. Key ones:

| Variable | Meaning |
|---|---|
| `MODEL_URL` / `MODEL_DATA_URL` | Direct Hugging Face file URLs |
| `MODEL_DIR` | Local folder the model is cached in |
| `IMG_SIZE` | Model input resolution (260) |
| `IMAGENET_MEAN` / `IMAGENET_STD` | Preprocessing normalization |
| `FAKE_THRESHOLD` | Cutoff on P(fake) for the real/fake label (default 0.5) |

`.env` is gitignored — never commit it. On Render, set the same variables in
the dashboard's Environment tab (or use `render.yaml`, included).

## Deploying to Render

1. Push this project to a GitHub repo.
2. In Render: **New → Web Service**, connect the repo. Render will pick up
   `render.yaml` automatically (or set Build Command
   `pip install -r requirements.txt` and Start Command from `Procfile`
   manually).
3. Fill in the environment variables (or leave the defaults from
   `render.yaml`, which already point at your Hugging Face repo).
4. Deploy. First boot will take longer than usual since it downloads the
   model — `--timeout 180` in the Procfile gives gunicorn enough headroom for
   that. Health checks hit `/health`, which doesn't block on the model, so
   the service reports healthy while the model finishes loading.
5. On Render's free tier, the disk is ephemeral, so every cold start /
   redeploy re-downloads the model from Hugging Face. That's expected and is
   what "low computational power hosting" in the synopsis is describing —
   inference-only, no training, no GPU.

## Project structure

```
app.py            Flask routes: / , /health , /predict
model_utils.py     Model download/cache + preprocessing + inference
templates/index.html  Upload UI
static/style.css      Visual design
static/script.js      Upload/drag-drop, scan animation, result rendering
requirements.txt
.env.example
Procfile           gunicorn start command for Render
render.yaml         Render blueprint (optional, for one-click IaC deploys)
```

## Notes / known limitations

- This is a course-project deployment: single gunicorn worker, CPU-only
  inference, no rate limiting or auth. Fine for a demo, not for production
  traffic.
- If the model's actual output turns out to be a raw logit instead of a
  post-sigmoid probability, `model_utils.predict()` already has a safety net
  that applies sigmoid when the raw value falls outside `[0, 1]` — but do
  double check against your training/export code that the export layer
  really is post-sigmoid.
