# ...existing code...
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uuid, os, subprocess, sys, json
from pathlib import Path

from config import (
    CLASSIFICATION_API_URL,
    SEGMENTATION_API_URL,
    CLASSIFICATION_TIMEOUT_SEC,
    SEGMENTATION_TIMEOUT_SEC,
    MODEL_API_RETRIES,
    MODEL_API_RETRY_BACKOFF_SEC,
)

load_dotenv()

app = FastAPI()
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

    # DEBUG LOGS: print completed job response
    print(f"[api] /api/analysis/result completed job_id={job_id}")
    print(f"[api] result.json path: {res_path}")
    print("[api] result payload:\n" + json.dumps(data, indent=2, ensure_ascii=False))

    return data
# ...existing code...