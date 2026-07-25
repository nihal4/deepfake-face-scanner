import io
import logging
import os

from dotenv import load_dotenv

load_dotenv()  # must run before model_utils reads os.environ

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

import model_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepfake_app")

# Download the model files once at import time, before gunicorn forks worker
# processes (run gunicorn with --preload, see Procfile). If this fails here,
# it's retried lazily on the first /predict call instead of crash-looping
# the whole process.
try:
    model_utils.ensure_model_files()
except Exception:
    logger.exception(
        "Model files could not be pre-downloaded at startup; "
        "will retry on first /predict request."
    )

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = (
    int(os.environ.get("MAX_CONTENT_LENGTH_MB", 10)) * 1024 * 1024
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Lightweight endpoint for Render's health check. Does not force a
    model load, so the app reports healthy while the model warms up."""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (field name 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        image.load()
    except UnidentifiedImageError:
        return jsonify({"error": "File is not a valid image"}), 400

    try:
        result = model_utils.predict(image)
    except Exception:
        logger.exception("Inference failed")
        return jsonify({"error": "Inference failed on the server"}), 500

    return jsonify(result)


if __name__ == "__main__":
    # Warm the model up on local dev runs so the first request isn't slow.
    logger.info("Warming up model (downloading if needed)...")
    model_utils.load_session()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
