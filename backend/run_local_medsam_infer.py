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

def _public_asset_url(job_id: str, filename: str) -> str:
    rel = f"/api/local_uploads/{job_id}/{filename}"
    return f"{PUBLIC_BASE_URL}{rel}" if PUBLIC_BASE_URL else rel

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
    warnings: list[str] = []
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

        _write_status_ex(job_dir, "running", stage="segmentation")

        img01 = _read_dicom_to_img01(input_path)
        seg_t0 = time.perf_counter()
        mask_u8 = _infer_mask_from_api(
            segmentation_api=args.segmentation_api,
            dicom_path=input_path,
            timeout_sec=args.segmentation_timeout,
            retries=max(0, args.retries),
            retry_backoff=max(0.0, args.retry_backoff),
        )
        seg_time = time.perf_counter() - seg_t0

        h, w = mask_u8.shape
        img_resized = cv2.resize(img01, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        original_name, mask_name, overlay_name = _save_outputs(job_dir, img_resized, mask_u8)

        comps = _connected_components(mask_u8)
        coords = []
        h, w = mask_u8.shape
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
            "requestId": request_id,
            "classification": {
                "has_cancer": True,
                "confidence": cls_conf,
                "processing_time": float(cls_payload.get("processing_time", classification_time)),
                "label": cls_payload.get("label"),
                "threshold": cls_payload.get("threshold"),
            },
            "segmentation": {
                "processing_time": round(seg_time, 4),
                "shape": [int(h), int(w)],
                "maskUrl": _public_asset_url(args.job_id, mask_name),
                "overlayUrl": _public_asset_url(args.job_id, overlay_name),
            },
            "malignancyScore": malignancy_score,
            "confidence": confidence,
            "noduleCount": nodule_count,
            "coordinates": coords,
            "findings": findings,
            "originalScan": _public_asset_url(args.job_id, original_name),
            "segmentationImages": [
                _public_asset_url(args.job_id, overlay_name),
                _public_asset_url(args.job_id, mask_name),
            ],
        }
        (job_dir / "result.json").write_text(
            json.dumps(result_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_status_ex(job_dir, "completed", stage="completed", warnings=warnings)
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print("[run_local_medsam_infer] error:")
        print(traceback.format_exc())
        res_path = job_dir / "result.json"
        if res_path.exists():
            try:
                partial = json.loads(res_path.read_text(encoding="utf-8"))
            except Exception:
                partial = {}

            if partial.get("classification", {}).get("has_cancer") is True:
                partial["segmentation"] = {
                    "error": err_msg,
                    "failed": True,
                }
                partial.setdefault("findings", [])
                partial["findings"].append("Segmentation failed after positive classification.")
                res_path.write_text(
                    json.dumps(partial, ensure_ascii=False),
                    encoding="utf-8",
                )
                _write_status_ex(job_dir, "completed", stage="completed", warnings=[err_msg])
                return

        _write_status_ex(job_dir, "failed", err_msg, stage="failed")
    
if __name__ == "__main__":
    main()