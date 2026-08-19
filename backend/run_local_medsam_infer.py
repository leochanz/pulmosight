# ...existing code...
import argparse
import json
import traceback
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import pydicom
import torch
import requests
import os

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover - used only when FastAPI is unavailable
    FastAPI = None
    HTTPException = None
    BaseModel = None

from medsam_segmentation_TRIAL2_auto_thr import (
    window_and_norm,
    SAM_PIXEL_MEAN,
    SAM_PIXEL_STD,
)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _write_status(job_dir: Path, status: str, error: str | None = None):
    payload = {"status": status}
    if error:
        payload["error"] = error
    (job_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_status_ex(
    job_dir: Path,
    status: str,
    error: str | None = None,
    stage: str | None = None,
    warnings: list[str] | None = None,
):
    payload: dict[str, Any] = {"status": status}
    if error:
        payload["error"] = error
    if stage:
        payload["stage"] = stage
    if warnings:
        payload["warnings"] = warnings
    (job_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def _read_dicom_to_img01(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    return window_and_norm(hu, win=(-1000, 400))


def _to_model_input(img01: np.ndarray, size: int = 1024) -> torch.Tensor:
    img_resized = cv2.resize(img01, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = (img_resized * 255.0).astype(np.float32)
    x = np.stack([x, x, x], axis=0)
    m = np.array(SAM_PIXEL_MEAN, dtype=np.float32)[:, None, None]
    s = np.array(SAM_PIXEL_STD, dtype=np.float32)[:, None, None]
    x = (x - m) / s
    return torch.from_numpy(x).unsqueeze(0).float(), img_resized


def _connected_components(mask_u8: np.ndarray):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    components = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 25:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        components.append((area, cx, cy))
    components.sort(key=lambda t: t[0], reverse=True)
    return components


def _save_outputs(job_dir: Path, img01_resized: np.ndarray, mask_u8: np.ndarray):
    original_png = (img01_resized * 255.0).clip(0, 255).astype(np.uint8)
    mask_png = (mask_u8 * 255).astype(np.uint8)

    rgb = cv2.cvtColor(original_png, cv2.COLOR_GRAY2BGR)
    overlay = rgb.copy()
    overlay[mask_u8 > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)

    original_path = job_dir / "original.png"
    mask_path = job_dir / "mask.png"
    overlay_path = job_dir / "overlay.png"
    cv2.imwrite(str(original_path), original_png)
    cv2.imwrite(str(mask_path), mask_png)
    cv2.imwrite(str(overlay_path), overlay)

    return original_path.name, mask_path.name, overlay_path.name

def _post_file_with_retry(
    url: str,
    dicom_path: Path,
    timeout_sec: float,
    retries: int,
    retry_backoff: float,
) -> requests.Response:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            with dicom_path.open("rb") as f:
                files = {"file": (dicom_path.name, f, "application/dicom")}
                resp = requests.post(url, files=files, timeout=timeout_sec)

            if resp.status_code >= 500 and attempt < retries:
                time.sleep(retry_backoff * (2**attempt))
                continue

            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(retry_backoff * (2**attempt))
                continue
            break

    raise RuntimeError(f"Request failed after retries: {type(last_exc).__name__}: {last_exc}")


def _infer_classification_from_api(
    classification_api: str,
    dicom_path: Path,
    timeout_sec: float,
    retries: int,
    retry_backoff: float,
) -> dict[str, Any]:
    url = f"{classification_api.rstrip('/')}/classify"
    resp = _post_file_with_retry(
        url=url,
        dicom_path=dicom_path,
        timeout_sec=timeout_sec,
        retries=retries,
        retry_backoff=retry_backoff,
    )

    if resp.status_code >= 400:
        raise RuntimeError(f"Classification API error {resp.status_code}: {resp.text}")

    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"Classification API returned error: {payload['error']}")
    if "has_cancer" not in payload:
        raise RuntimeError("Classification API response missing 'has_cancer'")

    return payload


def _infer_mask_from_api(
    segmentation_api: str,
    dicom_path: Path,
    timeout_sec: float,
    retries: int,
    retry_backoff: float,
) -> np.ndarray:
    url = f"{segmentation_api.rstrip('/')}/segment"
    resp = _post_file_with_retry(
        url=url,
        dicom_path=dicom_path,
        timeout_sec=timeout_sec,
        retries=retries,
        retry_backoff=retry_backoff,
    )

    if resp.status_code >= 400:
        raise RuntimeError(f"Model API error {resp.status_code}: {resp.text}")

    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"Model API returned error: {payload['error']}")
    if "mask" not in payload:
        raise RuntimeError("Model API response missing 'mask'")

    mask_u8 = np.array(payload["mask"], dtype=np.uint8)
    if mask_u8.ndim != 2:
        raise RuntimeError(f"Invalid mask shape from model API: {mask_u8.shape}")

    return (mask_u8 > 0).astype(np.uint8)


PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "local_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SEGMENTATION_JOB_STORE: dict[str, dict[str, Any]] = {}


def _public_asset_url(job_id: str, filename: str) -> str:
    rel = f"/api/local_uploads/{job_id}/{filename}"
    return f"{PUBLIC_BASE_URL}{rel}" if PUBLIC_BASE_URL else rel


def _run_segmentation_job(
    job_id: str,
    input_path: Path | None = None,
    segmentation_api: str | None = None,
    segmentation_timeout: float = 60.0,
    retries: int = 2,
    retry_backoff: float = 0.8,
) -> dict[str, Any]:
    """Manual segmentation gate: this function is only called from the explicit API route."""
    if input_path is None:
        input_path = next(iter(sorted((UPLOAD_DIR / job_id).glob("*.dcm"))), None)
    if input_path is None:
        raise FileNotFoundError(f"No DICOM file found for job_id={job_id}")

    job_dir = input_path.parent
    result_path = job_dir / "result.json"

    # Idempotent check: if segmentation already completed, return the stored result.
    if result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("segmentation") not in (None, {}) and existing.get("segmentation", {}).get("shape"):
                return {"success": True, "jobId": job_id, "status": "completed", "result": existing}
        except Exception:
            pass

    try:
        _write_status_ex(job_dir, "running", stage="segmentation")
        payload = {}
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}

        classification_result = payload.get("classification") or {}
        if not bool(classification_result.get("has_cancer", False)):
            raise RuntimeError("Segmentation is blocked for a negative classification result.")

        img01 = _read_dicom_to_img01(input_path)
        seg_t0 = time.perf_counter()
        mask_u8 = _infer_mask_from_api(
            segmentation_api=segmentation_api or os.getenv("SEGMENTATION_API_URL", "http://127.0.0.1:5001"),
            dicom_path=input_path,
            timeout_sec=segmentation_timeout,
            retries=max(0, retries),
            retry_backoff=max(0.0, retry_backoff),
        )
        seg_time = time.perf_counter() - seg_t0

        h, w = mask_u8.shape
        img_resized = cv2.resize(img01, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        original_name, mask_name, overlay_name = _save_outputs(job_dir, img_resized, mask_u8)

        comps = _connected_components(mask_u8)
        coords = []
        for idx, (_, cx, cy) in enumerate(comps[:5], start=1):
            coords.append(
                {
                    "x": round((cx / max(1.0, float(w))) * 100.0, 2),
                    "y": round((cy / max(1.0, float(h))) * 100.0, 2),
                    "label": f"N{idx}",
                }
            )

        nodule_count = len(comps)
        malignancy_score = min(99, 10 + nodule_count * 15)
        confidence = 90 if nodule_count > 0 else 82
        findings = [
            f"Detected {nodule_count} candidate nodule region(s).",
            "Segmentation generated from remote model API.",
            "Clinical review is recommended to confirm findings.",
        ]

        result_payload = {
            "requestId": job_id,
            "classification": {
                "has_cancer": bool(classification_result.get("has_cancer", False)),
                "confidence": float(classification_result.get("confidence", 0.0)),
                "processing_time": float(classification_result.get("processing_time", 0.0)),
                "label": classification_result.get("label"),
                "threshold": classification_result.get("threshold"),
            },
            "segmentation": {
                "processing_time": round(seg_time, 4),
                "shape": [int(h), int(w)],
                "maskUrl": _public_asset_url(job_id, mask_name),
                "overlayUrl": _public_asset_url(job_id, overlay_name),
            },
            "malignancyScore": malignancy_score,
            "confidence": confidence,
            "noduleCount": nodule_count,
            "coordinates": coords,
            "findings": findings,
            "originalScan": _public_asset_url(job_id, original_name),
            "segmentationImages": [
                _public_asset_url(job_id, overlay_name),
                _public_asset_url(job_id, mask_name),
            ],
        }
        result_path.write_text(json.dumps(result_payload, ensure_ascii=False), encoding="utf-8")
        _write_status_ex(job_dir, "completed", stage="completed")
        SEGMENTATION_JOB_STORE[job_id] = {"status": "completed", "jobId": job_id, "result": result_payload}
        return {"success": True, "jobId": job_id, "status": "completed", "result": result_payload}
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print("[run_local_medsam_infer] segmentation error:")
        print(traceback.format_exc())
        _write_status_ex(job_dir, "failed", err_msg, stage="failed")
        SEGMENTATION_JOB_STORE[job_id] = {"status": "failed", "jobId": job_id, "error": err_msg}
        return {"success": False, "jobId": job_id, "status": "failed", "error": err_msg}


if FastAPI is not None:
    app = FastAPI()

    class RunSegmentationRequest(BaseModel):
        jobId: str | None = None
        requestId: str | None = None
        analysisId: str | None = None

    @app.post("/api/segmentation/run")
    async def run_segmentation_route(payload: RunSegmentationRequest):
        """Manual trigger endpoint. Segmentation starts only after user click."""
        job_id = payload.jobId or payload.requestId or payload.analysisId
        if not job_id:
            raise HTTPException(status_code=400, detail="jobId/requestId/analysisId is required")

        job_dir = UPLOAD_DIR / job_id
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        status_path = job_dir / "status.json"
        if status_path.exists():
            try:
                current_status = json.loads(status_path.read_text(encoding="utf-8"))
                if current_status.get("status") in {"running", "pending"}:
                    return {
                        "success": True,
                        "jobId": job_id,
                        "status": current_status.get("status", "running"),
                        "message": "Segmentation is already in progress for this job.",
                    }
            except Exception:
                pass

        result_path = job_dir / "result.json"
        if result_path.exists():
            try:
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
                if result_payload.get("segmentation") not in (None, {}) and result_payload.get("segmentation", {}).get("shape"):
                    return {
                        "success": True,
                        "jobId": job_id,
                        "status": "completed",
                        "result": result_payload,
                        "message": "Segmentation already completed for this job.",
                    }
            except Exception:
                pass

        result = _run_segmentation_job(
            job_id=job_id,
            input_path=next(iter(sorted(job_dir.glob("*.dcm"))), None),
            segmentation_api=os.getenv("SEGMENTATION_API_URL", "http://127.0.0.1:5001"),
            segmentation_timeout=float(os.getenv("SEGMENTATION_TIMEOUT_SEC", "60")),
            retries=int(os.getenv("MODEL_API_RETRIES", "2")),
            retry_backoff=float(os.getenv("MODEL_API_RETRY_BACKOFF_SEC", "0.8")),
        )

        if result.get("success"):
            return {
                "success": True,
                "jobId": job_id,
                "status": result["status"],
                "result": result.get("result"),
                "resultUrl": f"/api/analysis/result/{job_id}",
            }

        raise HTTPException(status_code=500, detail=result.get("error", "Segmentation failed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--classification_api", required=True)
    ap.add_argument("--segmentation_api", required=True)
    ap.add_argument("--job_id", required=True)
    ap.add_argument("--classification_timeout", type=float, default=30.0)
    ap.add_argument("--segmentation_timeout", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry_backoff", type=float, default=0.8)
    args = ap.parse_args()

    input_path = Path(args.input)
    job_dir = input_path.parent
    request_id = args.job_id
    _write_status_ex(job_dir, "running", stage="classification")

    try:
        if input_path.suffix.lower() != ".dcm":
            raise RuntimeError("Current local inference supports .dcm uploads only.")

        classification_t0 = time.perf_counter()
        cls_payload = _infer_classification_from_api(
            classification_api=args.classification_api,
            dicom_path=input_path,
            timeout_sec=args.classification_timeout,
            retries=max(0, args.retries),
            retry_backoff=max(0.0, args.retry_backoff),
        )
        classification_time = time.perf_counter() - classification_t0

        has_cancer = bool(cls_payload.get("has_cancer", False))
        cls_conf = float(cls_payload.get("confidence", 0.0))

        base_payload = {
            "requestId": request_id,
            "classification": {
                "has_cancer": has_cancer,
                "confidence": cls_conf,
                "processing_time": float(cls_payload.get("processing_time", classification_time)),
                "label": cls_payload.get("label"),
                "threshold": cls_payload.get("threshold"),
            },
            "segmentation": None,
        }

        (job_dir / "result.json").write_text(
            json.dumps(base_payload, ensure_ascii=False),
            encoding="utf-8",
        )

        if not has_cancer:
            base_payload.update(
                {
                    "malignancyScore": int(round(cls_conf * 100)),
                    "confidence": int(round(cls_conf * 100)),
                    "noduleCount": 0,
                    "coordinates": [],
                    "findings": [
                        "Classification model predicts no cancer evidence on this slice.",
                        "Segmentation skipped because classification result is negative.",
                        "Clinical review is recommended to confirm findings.",
                    ],
                    "originalScan": "",
                    "segmentationImages": [],
                }
            )
            (job_dir / "result.json").write_text(
                json.dumps(base_payload, ensure_ascii=False),
                encoding="utf-8",
            )
            _write_status_ex(job_dir, "completed", stage="completed")
            return

        # IMPORTANT: manual gate. The segmentation step is intentionally not triggered here.
        # The explicit user action must call /api/segmentation/run.
        _write_status_ex(
            job_dir,
            "completed",
            stage="classification",
            warnings=["Classification complete; segmentation awaiting explicit user trigger."],
        )
        return

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print("[run_local_medsam_infer] error:")
        print(traceback.format_exc())
        _write_status_ex(job_dir, "failed", err_msg, stage="failed")


if __name__ == "__main__":
    main()