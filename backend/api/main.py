"""
MDS-ICU FastAPI Backend
Exposes all features of the ICU-BACKEND ML codebase via HTTP endpoints.
"""
import sys, pathlib

# Ensure the ML src/ directory is never accidentally on sys.path when the API
# starts — this prevents uvicorn's --reload scanner from importing main_all.py
_src = str(pathlib.Path(__file__).parent.parent / "src")
sys.path = [p for p in sys.path if not p.startswith(_src)]

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import training, inference, data, models, metrics, config, system, patients


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure a trained model exists in backend/models/
    from .services.inference_service import ensure_model_exists
    ensure_model_exists()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="MDS-ICU API",
    description="""
## Medical Decision Support in ICU — REST API

Provides programmatic access to all features of the MDS-ICU deep learning framework:

- **Training** — Configure, start, monitor, and stop model training runs
- **Inference** — Run predictions on ECG + tabular patient data (33 clinical outcome labels)
- **Data** — Upload and visualize ECG waveforms and tabular features
- **Models** — Manage model checkpoints
- **Metrics** — View training metrics, AUROC curves, per-class results
- **Configuration** — Inspect and modify training configuration
- **System** — Hardware info and health check

### Quick Start
1. `GET /system/info` — check hardware
2. `POST /inference/predict/demo` — run a demo prediction
3. `GET /data/ecg/demo/12lead` — get a demo ECG waveform
4. `GET /metrics/demo` — get synthetic training curves
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow Next.js dev server + prod) ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(system.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(inference.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(config.router, prefix="/api")
app.include_router(patients.router, prefix="/api")


@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({
        "name": "MDS-ICU API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/system/health",
    })
