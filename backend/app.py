# ...existing code...
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uuid, os, subprocess, sys, json
from datetime import datetime
from pathlib import Path

from config import (
    CLASSIFICATION_API_URL,
    SEGMENTATION_API_URL,
    CLASSIFICATION_TIMEOUT_SEC,
    SEGMENTATION_TIMEOUT_SEC,
    MODEL_API_RETRIES,
    MODEL_API_RETRY_BACKOFF_SEC,
)
from database import (
    fetch_analysis_detail,
    fetch_history,
    initialize_db,
    save_analysis_result,
    upsert_patient,
    upsert_scan_job,
)
from run_local_medsam_infer import app as segmentation_app

load_dotenv()
initialize_db()

app = FastAPI()
app.include_router(segmentation_app.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://154.85.62.123", "http://pulmosight.top", "https://pulmosight.top", "http://localhost:8080", "http://127.0.0.1:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "local_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount(
    "/api/local_uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="local_uploads",
)


def _status_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def _result_path(job_dir: Path) -> Path:
    return job_dir / "result.json"


def _write_status(
    job_dir: Path,
    status: str,
    error: str | None = None,
    stage: str | None = None,
    warnings: list[str] | None = None,
):
    payload = {"status": status}
    if error:
        payload["error"] = error
    if stage:
        payload["stage"] = stage
    if warnings:
        payload["warnings"] = warnings
    _status_path(job_dir).write_text(json.dumps(payload), encoding="utf-8")

async def _start_analysis_job(
    ctScan: UploadFile,
    patientId: str,
    patientName: str,
    age: str,
    gender: str,
):
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_status(job_dir, "pending", stage="queued")

    safe_name = Path(ctScan.filename or "upload.bin").name
    scan_path = job_dir / safe_name
    with open(scan_path, "wb") as f:
        f.write(await ctScan.read())

    # Optional: save metadata
    (job_dir / "meta.json").write_text(
        (
            "{"
            f"\"patientId\":\"{patientId}\","
            f"\"patientName\":\"{patientName}\","
            f"\"age\":\"{age}\","
            f"\"gender\":\"{gender}\""
            "}"
        ),
        encoding="utf-8",
    )

    upsert_patient(patientId, patientName, age, gender)
    upsert_scan_job(
        job_id,
        patientId,
        datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        str(scan_path),
    )

    # Run local inference script (non-blocking)
    subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "run_local_medsam_infer.py"),
            "--input",
            str(scan_path),
            "--classification_api",
            CLASSIFICATION_API_URL,
            "--segmentation_api",
            SEGMENTATION_API_URL,
            "--job_id",
            job_id,
            "--classification_timeout",
            str(CLASSIFICATION_TIMEOUT_SEC),
            "--segmentation_timeout",
            str(SEGMENTATION_TIMEOUT_SEC),
            "--retries",
            str(MODEL_API_RETRIES),
            "--retry_backoff",
            str(MODEL_API_RETRY_BACKOFF_SEC),
        ],
        cwd=str(ROOT),
        stdout=open(job_dir / "subprocess.log", "w"),
        stderr=subprocess.STDOUT,
    )

    return {"jobId": job_id}

@app.post("/api/analysis/start")
async def start_analysis_api(
    ctScan: UploadFile = File(...),
    patientId: str = Form(...),
    patientName: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
):
    return await _start_analysis_job(ctScan, patientId, patientName, age, gender)


@app.post("/analysis/start-local")
async def start_local_analysis(
    ctScan: UploadFile = File(...),
    patientId: str = Form(...),
    patientName: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
):
    return await _start_analysis_job(ctScan, patientId, patientName, age, gender)


@app.get("/api/analysis/status/{job_id}")
async def analysis_status(job_id: str):
    job_dir = UPLOAD_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    st_path = _status_path(job_dir)
    if not st_path.exists():
        return {"status": "pending"}

    try:
        return json.loads(st_path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "pending"}


# ...existing code...
@app.get("/api/analysis/result/{job_id}")
async def analysis_result(job_id: str):
    job_dir = UPLOAD_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    st_path = _status_path(job_dir)
    if st_path.exists():
        try:
            st = json.loads(st_path.read_text(encoding="utf-8"))
        except Exception:
            st = {"status": "pending"}
    else:
        st = {"status": "pending"}

    status = st.get("status", "pending")
    if status != "completed":
        payload = {"status": status, "stage": st.get("stage", "queued")}
        if "error" in st:
            payload["error"] = st.get("error")
        if "warnings" in st:
            payload["warnings"] = st.get("warnings")
        res_path = _result_path(job_dir)
        if res_path.exists():
            try:
                partial = json.loads(res_path.read_text(encoding="utf-8"))
                payload.update(partial)
            except Exception:
                pass
        return payload

    res_path = _result_path(job_dir)
    if not res_path.exists():
        return {"status": "running"}

    try:
        data = json.loads(res_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corrupted result.json: {e}")

    data["status"] = "completed"
    data["stage"] = st.get("stage", "completed")
    if "warnings" in st:
        data["warnings"] = st.get("warnings")

    if "requestId" not in data:
        data["requestId"] = job_id

    # Persist the job result into SQLite so it can be displayed by the history UI.
    classification = data.get("classification") or {}
    is_cancer = bool(classification.get("has_cancer", False))
    malignancy_score = data.get("malignancyScore")
    confidence = data.get("confidence")

    try:
        meta = {}
        meta_path = job_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        patient_id = meta.get("patientId")
        patient_name = meta.get("patientName") or "Unknown"
        age = meta.get("age")
        gender = meta.get("gender")
        if patient_id:
            upsert_patient(patient_id, patient_name, age, gender)
            upsert_scan_job(
                job_id,
                patient_id,
                datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                str(next(iter(sorted(job_dir.glob("*.dcm"))), "")) or None,
            )

        save_analysis_result(job_id, is_cancer, malignancy_score, confidence)
    except Exception:
        pass

    print(f"[api] /api/analysis/result completed job_id={job_id}")
    print(f"[api] result.json path: {res_path}")
    print("[api] result payload:\n" + json.dumps(data, indent=2, ensure_ascii=False))

    return data


@app.get("/api/history")
async def history_api():
    """Return a joined view of patient + scan + analysis data for the history dashboard."""
    rows = fetch_history()
    payload = []
    for row in rows:
        payload.append(
            {
                "jobId": row.get("job_id"),
                "patientId": row.get("patient_id"),
                "patientName": row.get("patient_name") or "Unknown",
                "scanDate": row.get("scan_date"),
                "malignancyScore": row.get("malignancy_score"),
                "isCancer": bool(row.get("is_cancer")),
                "confidence": row.get("confidence"),
            }
        )
    return payload


@app.get("/api/analysis/{job_id}")
async def analysis_detail_api(job_id: str):
    """Return the persisted result for a single job, suitable for the detail page route."""
    job_dir = UPLOAD_DIR / job_id
    result_path = _result_path(job_dir) if job_dir.exists() else None
    result_data: dict[str, Any] = {}

    if result_path and result_path.exists():
        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            result_data = {}

    item = fetch_analysis_detail(job_id)
    if item is None and not result_data:
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        if not result_path or not result_path.exists():
            raise HTTPException(status_code=404, detail="Analysis result not found")

    meta = {}
    if job_dir.exists():
        meta_path = job_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

    if item is None:
        return {
            "jobId": job_id,
            "patientId": meta.get("patientId"),
            "patientName": meta.get("patientName") or "Unknown",
            "age": meta.get("age"),
            "gender": meta.get("gender"),
            "scanDate": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "imagePath": str(next(iter(sorted(job_dir.glob("*.dcm"))), "")) or None if job_dir.exists() and any(job_dir.glob("*.dcm")) else None,
            "classification": result_data.get("classification") or {"has_cancer": False, "confidence": 0},
            "segmentation": result_data.get("segmentation"),
            "malignancyScore": result_data.get("malignancyScore"),
            "confidence": result_data.get("confidence"),
            "noduleCount": result_data.get("noduleCount"),
            "coordinates": result_data.get("coordinates") or [],
            "findings": result_data.get("findings") or [],
            "originalScan": result_data.get("originalScan") or "",
            "segmentationImages": result_data.get("segmentationImages") or [],
            "status": "completed",
            "stage": "completed",
        }

    return {
        "jobId": item.get("job_id"),
        "patientId": item.get("patient_id"),
        "patientName": item.get("patient_name") or meta.get("patientName") or "Unknown",
        "age": item.get("age") if item.get("age") is not None else meta.get("age"),
        "gender": item.get("gender") if item.get("gender") is not None else meta.get("gender"),
        "scanDate": item.get("scan_date") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "imagePath": item.get("image_path"),
        "classification": result_data.get("classification") or {
            "has_cancer": bool(item.get("is_cancer")),
            "confidence": float(item.get("confidence") or 0),
        },
        "segmentation": result_data.get("segmentation"),
        "malignancyScore": result_data.get("malignancyScore", item.get("malignancy_score")),
        "confidence": result_data.get("confidence", item.get("confidence")),
        "noduleCount": result_data.get("noduleCount", 0),
        "coordinates": result_data.get("coordinates") or [],
        "findings": result_data.get("findings") or [],
        "originalScan": result_data.get("originalScan") or "",
        "segmentationImages": result_data.get("segmentationImages") or [],
        "status": "completed",
        "stage": "completed",
    }


@app.post("/api/analysis")
async def save_analysis_api(payload: dict):
    """Persist a completed analysis result. This is used by the history screen and detail page."""
    job_id = payload.get("jobId") or payload.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="jobId is required")

    patient = payload.get("patient") or {}
    patient_id = patient.get("id") or patient.get("patientId")
    patient_name = patient.get("name") or patient.get("patientName") or "Unknown"
    age = patient.get("age")
    gender = patient.get("gender")
    if patient_id:
        upsert_patient(patient_id, patient_name, age, gender)

    scan_date = payload.get("scanDate") or payload.get("scan_date") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    image_path = payload.get("imagePath") or payload.get("image_path")
    if patient_id:
        upsert_scan_job(job_id, patient_id, scan_date, image_path)

    classification = payload.get("classification") or {}
    is_cancer = bool(classification.get("has_cancer", False))
    malignancy_score = payload.get("malignancyScore")
    confidence = payload.get("confidence")
    save_analysis_result(job_id, is_cancer, malignancy_score, confidence)
    return {"success": True, "jobId": job_id}
# ...existing code...