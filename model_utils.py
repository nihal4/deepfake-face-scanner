"""
Handles downloading the ONNX model (+ external data file) from a public
Hugging Face repo on first startup, and running inference on an uploaded
face image.
"""

import os
import logging
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image, ImageOps

logger = logging.getLogger("deepfake_app")

# ---------------------------------------------------------------------------
# Config (all overridable via .env)
# ---------------------------------------------------------------------------
MODEL_URL = os.environ["MODEL_URL"]
MODEL_DATA_URL = os.environ["MODEL_DATA_URL"]

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "deepfake_hybrid_final.onnx")
MODEL_DATA_FILENAME = os.environ.get(
    "MODEL_DATA_FILENAME", "deepfake_hybrid_final.onnx.data"
)

IMG_SIZE = int(os.environ.get("IMG_SIZE", 260))
IMAGENET_MEAN = np.array(
    [float(x) for x in os.environ.get("IMAGENET_MEAN", "0.485,0.456,0.406").split(",")],
    dtype=np.float32,
)
IMAGENET_STD = np.array(
    [float(x) for x in os.environ.get("IMAGENET_STD", "0.229,0.224,0.225").split(",")],
    dtype=np.float32,
)
FAKE_THRESHOLD = float(os.environ.get("FAKE_THRESHOLD", 0.5))

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_DATA_PATH = os.path.join(MODEL_DIR, MODEL_DATA_FILENAME)

_session = None
_input_name = None
_output_name = None


def _download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> None:
    """Stream-download a file to dest_path, skipping if it already exists
    and looks complete (non-zero size)."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        logger.info("Already have %s (%.1f MB), skipping download",
                     dest_path, os.path.getsize(dest_path) / 1e6)
        return

    tmp_path = dest_path + ".part"
    logger.info("Downloading %s -> %s", url, dest_path)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        if total and downloaded != total:
            os.remove(tmp_path)
            raise IOError(
                f"Incomplete download for {url}: got {downloaded} of {total} bytes"
            )
    os.replace(tmp_path, dest_path)
    logger.info("Saved %s (%.1f MB)", dest_path, os.path.getsize(dest_path) / 1e6)


def ensure_model_files() -> None:
    """Downloads the .onnx and .onnx.data files from Hugging Face into
    MODEL_DIR if they aren't already present locally."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    _download_file(MODEL_DATA_URL, MODEL_DATA_PATH)  # download data file first
    _download_file(MODEL_URL, MODEL_PATH)            # .onnx references it by filename


def load_session() -> ort.InferenceSession:
    """Lazily creates (and caches) the ONNX Runtime inference session."""
    global _session, _input_name, _output_name
    if _session is not None:
        return _session

    ensure_model_files()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]

    logger.info("Loading ONNX Runtime session from %s", MODEL_PATH)
    _session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)
    _input_name = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name

    in_info = _session.get_inputs()[0]
    out_info = _session.get_outputs()[0]
    logger.info(
        "Model loaded. input=%s shape=%s dtype=%s | output=%s shape=%s dtype=%s",
        in_info.name, in_info.shape, in_info.type,
        out_info.name, out_info.shape, out_info.type,
    )
    if len(_session.get_outputs()) > 1:
        logger.warning(
            "Model has %d outputs; only using the first one (%s). "
            "If the real prediction is a different output, this needs updating.",
            len(_session.get_outputs()), _output_name,
        )
    return _session


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize/normalize a PIL image into an NCHW float32 tensor matching the
    training preprocessing: IMG_SIZE x IMG_SIZE, RGB, ImageNet mean/std."""
    image = ImageOps.exif_transpose(image)  # respect camera orientation
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    arr = np.asarray(image).astype(np.float32) / 255.0       # HWC, [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD                # normalize
    arr = arr.transpose(2, 0, 1)                               # CHW
    arr = np.expand_dims(arr, axis=0).astype(np.float32)       # NCHW
    return arr


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict(image: Image.Image) -> dict:
    """Runs the model on a PIL image and returns a result dict.

    Handles a few possible output shapes defensively, since the exported
    model's actual output doesn't always match what was assumed at design
    time:
      - size 1  -> single sigmoid/logit for P(fake)
      - size 2  -> two-class output, class order [fake, real] (index 0 =
                   fake, index 1 = real), as either softmax probabilities
                   or raw logits
      - other   -> logged and rejected with a clear error, rather than
                   silently producing a meaningless number
    """
    session = load_session()
    tensor = preprocess_image(image)

    raw = session.run([_output_name], {_input_name: tensor})[0]
    flat = np.asarray(raw).astype(np.float64).reshape(-1)
    logger.info("Raw model output shape=%s values=%s", raw.shape, flat)

    if flat.size == 1:
        raw_value = float(flat[0])
        # Some exports emit raw logits instead of post-sigmoid probabilities;
        # if the value falls outside [0, 1] we apply sigmoid ourselves.
        if raw_value < 0.0 or raw_value > 1.0:
            fake_prob = float(_sigmoid(np.array(raw_value)))
        else:
            fake_prob = raw_value

    elif flat.size == 2:
        # Model's class order is [fake, real] (confirmed by user: 0=fake, 1=real).
        # If probabilities already look like they sum to ~1, treat them as-is;
        # otherwise treat as logits and apply softmax.
        if abs(flat.sum() - 1.0) < 1e-3 and flat.min() >= 0.0 and flat.max() <= 1.0:
            probs = flat
        else:
            probs = _softmax(flat)
        fake_prob = float(probs[0])

    else:
        logger.error(
            "Unexpected model output size=%d shape=%s values=%s — "
            "don't know how to interpret this, update predict() to match "
            "the model's real output format.",
            flat.size, raw.shape, flat,
        )
        raise ValueError(
            f"Unexpected model output shape {raw.shape} (size {flat.size}); "
            "expected a single value or a 2-class output."
        )

    is_fake = fake_prob >= FAKE_THRESHOLD
    label = "fake" if is_fake else "real"
    confidence = fake_prob if is_fake else (1.0 - fake_prob)

    return {
        "label": label,
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(1.0 - fake_prob, 4),
        "confidence": round(confidence, 4),
        "threshold": FAKE_THRESHOLD,
    }
