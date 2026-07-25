# Face Authenticity Scanner — Deepfake Face Detector Web App
A lightweight web application for detecting **AI-generated / deepfake face images** in real time. Upload a photo and get an instant **real / fake** classification with a confidence score, powered by a fine-tuned hybrid CNN running in ONNX format.

**🔗 Live demo:** [deepfake-face-scanner.onrender.com](https://deepfake-face-scanner.onrender.com/)

- Upload screen — drag-and-drop dropzone before a scan is run
  
<img width="1427" height="774" alt="Screenshot 2026-07-25 at 9 36 02 PM" src="https://github.com/user-attachments/assets/5e808f4c-c216-4458-bcc3-de42b4f22674" />

- Result screen — verdict, probability meter, and confidence metrics after a scan
<img width="1427" height="774" alt="Screenshot 2026-07-25 at 9 36 41 PM" src="https://github.com/user-attachments/assets/84b7c99b-426e-4072-8518-abb354e505a1" />


## Overview

Face Authenticity Scanner wraps a fine-tuned hybrid CNN deepfake classifier — exported to ONNX for fast, CPU-only inference — behind a Flask web interface. It was built as the deployment/delivery phase of a deepfake face detection course project (AI Lab, SE334, Daffodil International University).

Model card and full evaluation results: [nihal4/Deep_Fake_Hybrid_Model](https://huggingface.co/nihal4/Deep_Fake_Hybrid_Model)

## Features

- Drag-and-drop or click-to-upload web UI — no setup needed to try the live demo
- ONNX Runtime inference — no GPU required
- Model weights fetched automatically from Hugging Face on first startup and cached locally
- Fake-probability score, confidence, and a visual probability meter
- Simple Flask backend, easy to self-host or extend

## Repo Structure

```
.
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── LICENSE
├── Procfile
├── README.md
├── app.py
├── model_utils.py
├── render.yaml
├── requirements.txt
└── .env.example
```

> Model weights are not stored in the repo — `deepfake_hybrid_final.onnx` and `deepfake_hybrid_final.onnx.data` are downloaded automatically into a `models/` folder the first time the app runs.

## Example `.env`

Copy this to `.env` before running — the defaults already point at the public Hugging Face model repo, so no edits are required to get started:

```env
# --- Model source (Hugging Face public repo, direct file URLs) ---
MODEL_URL=https://huggingface.co/nihal4/Deep_Fake_Hybrid_Model/resolve/main/deepfake_hybrid_final.onnx
MODEL_DATA_URL=https://huggingface.co/nihal4/Deep_Fake_Hybrid_Model/resolve/main/deepfake_hybrid_final.onnx.data

# --- Local storage ---
MODEL_DIR=models
MODEL_FILENAME=deepfake_hybrid_final.onnx
MODEL_DATA_FILENAME=deepfake_hybrid_final.onnx.data

# --- Inference / preprocessing ---
IMG_SIZE=260
IMAGENET_MEAN=0.485,0.456,0.406
IMAGENET_STD=0.229,0.224,0.225
FAKE_THRESHOLD=0.5

# --- Flask ---
FLASK_ENV=production
PORT=5000
MAX_CONTENT_LENGTH_MB=10
```

## Running Locally

1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/nihal4/deepfake-face-scanner.git
   cd deepfake-face-scanner
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   See [Example `.env`](#example-env) above — the defaults already work out of the box.
5. **Run the app**
   ```bash
   python app.py
   ```
   On first run this downloads `deepfake_hybrid_final.onnx` and `deepfake_hybrid_final.onnx.data` from Hugging Face into `models/` — no manual download needed. Subsequent runs use the cached copy.

That's it — open **http://localhost:5000** and the app should now be running locally.

## Model

- **Architecture:** hybrid CNN combining **EfficientNet-B1** and **RegNet_Y_800MF** backbones, fine-tuned for binary `real` / `fake` face classification
- **Format:** ONNX (exported for lightweight CPU inference), input size 260×260, ImageNet normalization
- **Details, dataset, and evaluation metrics:** see the [model card](https://huggingface.co/nihal4/Deep_Fake_Hybrid_Model)

## License

This project is licensed under the MIT License — see [`LICENSE`](./LICENSE) for details.

## Authors & Contributors

- S. M. Nihal Ahmed
