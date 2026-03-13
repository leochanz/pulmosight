import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pydicom
import torch
from flask import Flask, jsonify, request

from interference_classification import infer_one_dicom, load_model as load_cls_model
from medsam_segmentation_TRIAL2_auto_thr import (
    MedSAMSegModel,
    SAM_PIXEL_MEAN,
    SAM_PIXEL_STD,
    window_and_norm,
)

app = Flask(__name__)

MODEL_PT = os.getenv("MODEL_PT", "medsam_seg_fold2_best.pt")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

ENABLE_SEGMENTATION_MODEL = os.getenv("ENABLE_SEGMENTATION_MODEL", "1") == "1"
ENABLE_CLASSIFICATION_MODEL = os.getenv("ENABLE_CLASSIFICATION_MODEL", "1") == "1"

CLASSIFIER_CKPT = os.getenv("CLASSIFIER_CKPT", "best_model_fold5.pth")
CLASSIFIER_LOCAL_DIR = os.getenv("CLASSIFIER_LOCAL_DIR", "medsiglip-448")
CLASSIFIER_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.7"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

ALLOW_ORIGIN = os.getenv("MODEL_SERVER_ALLOW_ORIGIN", "*")

device = None
seg_model = None
seg_model_error = None
cls_bundle = None
cls_model_error = None

seg_lock = Lock()
cls_lock = Lock()


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_dicom_bytes_to_img01(raw: bytes) -> np.ndarray:
    ds = pydicom.dcmread(BytesIO(raw), force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    return window_and_norm(hu, win=(-1000, 400))


def _to_model_input(img01: np.ndarray, size: int) -> torch.Tensor:
    img_resized = cv2.resize(img01, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = (img_resized * 255.0).astype(np.float32)
    x = np.stack([x, x, x], axis=0)
    m = np.array(SAM_PIXEL_MEAN, dtype=np.float32)[:, None, None]
    s = np.array(SAM_PIXEL_STD, dtype=np.float32)[:, None, None]
    x = (x - m) / s
    return torch.from_numpy(x).unsqueeze(0).float()


def _validate_upload(raw: bytes, filename: str) -> str | None:
    if not raw:
        return "Empty file"
    if len(raw) > MAX_UPLOAD_BYTES:
        return f"File too large; max {MAX_UPLOAD_MB}MB"
    ext = Path(filename or "").suffix.lower()
    if ext and ext not in {".dcm", ".dicom"}:
        return "Invalid file type; expected .dcm/.dicom"
    return None


def _init_segmentation_model(run_device: torch.device):
    start = time.perf_counter()
    model = MedSAMSegModel(
        sam_type="vit_b",
        medsam_ckpt=MODEL_PT,
        unfreeze_last_n=2,
        device=run_device,
    ).to(run_device)
    ckpt = torch.load(MODEL_PT, map_location=torch.device("cpu"))
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    elapsed = time.perf_counter() - start
    print(f"[model_server] segmentation model loaded in {elapsed:.2f}s")
    return model


def _init_classification_model(run_device: torch.device):
    start = time.perf_counter()
    model, mean, std, win, stride_val, pool_mode, topk, meta = load_cls_model(
        ckpt_path=CLASSIFIER_CKPT,
        local_dir=CLASSIFIER_LOCAL_DIR,
        device=run_device,
    )
    elapsed = time.perf_counter() - start
    print(f"[model_server] classification model loaded in {elapsed:.2f}s")
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "win": win,
        "stride": stride_val,
        "pool_mode": pool_mode,
        "topk": topk,
        "meta": meta,
    }


def _initialize_models():
    global device, seg_model, seg_model_error, cls_bundle, cls_model_error

    device = _pick_device()
    print(f"[model_server] device={device}")

    if ENABLE_SEGMENTATION_MODEL:
        try:
            seg_model = _init_segmentation_model(device)
        except Exception as e:
            seg_model_error = f"{type(e).__name__}: {e}"
            print(f"[model_server] segmentation model failed to load: {seg_model_error}")

    if ENABLE_CLASSIFICATION_MODEL:
        try:
            cls_bundle = _init_classification_model(device)
        except Exception as e:
            cls_model_error = f"{type(e).__name__}: {e}"
            print(f"[model_server] classification model failed to load: {cls_model_error}")


@app.after_request
def _add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOW_ORIGIN
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "device": str(device),
            "segmentation_ready": seg_model is not None,
            "classification_ready": cls_bundle is not None,
            "segmentation_error": seg_model_error,
            "classification_error": cls_model_error,
        }
    )


@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():
    if request.method == "OPTIONS":
        return ("", 204)

    if cls_bundle is None:
        return jsonify({"error": "Classification model unavailable", "detail": cls_model_error}), 503

    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file field: 'file'"}), 400

        f = request.files["file"]
        raw = f.read()
        validation_error = _validate_upload(raw, f.filename or "")
        if validation_error:
            return jsonify({"error": validation_error}), 400

        t0 = time.perf_counter()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            with cls_lock:
                result = infer_one_dicom(
                    model=cls_bundle["model"],
                    dcm_path=tmp_path,
                    device=device,
                    mean=cls_bundle["mean"],
                    std=cls_bundle["std"],
                    win=cls_bundle["win"],
                    stride=cls_bundle["stride"],
                    pool_mode=cls_bundle["pool_mode"],
                    topk=cls_bundle["topk"],
                    threshold=CLASSIFIER_THRESHOLD,
                )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        elapsed = time.perf_counter() - t0
        confidence = float(result["pooled_score"])
        has_cancer = bool(result["pred_label"] == 1)
        return jsonify(
            {
                "has_cancer": has_cancer,
                "confidence": confidence,
                "processing_time": round(elapsed, 4),
                "label": result["pred_name"],
                "threshold": float(result["threshold"]),
                "n_windows": int(result["n_windows"]),
                "top_windows": result["top_windows"],
            }
        )
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/segment", methods=["POST", "OPTIONS"])
def segment():
    if request.method == "OPTIONS":
        return ("", 204)

    if seg_model is None:
        return jsonify({"error": "Segmentation model unavailable", "detail": seg_model_error}), 503

    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file field: 'file'"}), 400

        f = request.files["file"]
        raw = f.read()
        validation_error = _validate_upload(raw, f.filename or "")
        if validation_error:
            return jsonify({"error": validation_error}), 400

        t0 = time.perf_counter()
        img01 = _read_dicom_bytes_to_img01(raw)
        sam_img_size = int(getattr(seg_model.sam.image_encoder, "img_size", 1024))
        x = _to_model_input(img01, sam_img_size).to(device)

        with seg_lock, torch.no_grad():
            logits = seg_model(x)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        mask_u8 = (prob > THRESHOLD).astype(np.uint8)
        elapsed = time.perf_counter() - t0

        return jsonify(
            {
                "mask": mask_u8.tolist(),
                "threshold": THRESHOLD,
                "shape": [int(mask_u8.shape[0]), int(mask_u8.shape[1])],
                "processing_time": round(elapsed, 4),
            }
        )
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/infer", methods=["POST", "OPTIONS"])
def infer_alias():
    return segment()


def _run_self_test(sample_dicom: str):
    if not os.path.exists(sample_dicom):
        print(f"[self-test] sample dicom not found: {sample_dicom}")
        return

    with open(sample_dicom, "rb") as fh:
        raw = fh.read()

    with app.test_client() as c:
        seg_resp = c.post(
            "/segment",
            data={"file": (BytesIO(raw), "sample.dcm")},
            content_type="multipart/form-data",
        )
        print(f"[self-test] /segment -> {seg_resp.status_code}")

        cls_resp = c.post(
            "/classify",
            data={"file": (BytesIO(raw), "sample.dcm")},
            content_type="multipart/form-data",
        )
        print(f"[self-test] /classify -> {cls_resp.status_code}")


_initialize_models()


if __name__ == "__main__":
    sample_dicom = os.getenv("SELF_TEST_DICOM", "").strip()
    if sample_dicom:
        _run_self_test(sample_dicom)
    app.run(host="0.0.0.0", port=5001, debug=False)