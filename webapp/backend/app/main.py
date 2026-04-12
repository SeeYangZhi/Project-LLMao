from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import human_eval, inference, metrics, mislabels, samples
from app.services.data_loader import data_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_store.load()
    print(f"Loaded {len(data_store.results)} model results into memory")
    yield


app = FastAPI(
    title="Project LLMao API",
    description="Sarcasm Style Transfer - CS4248 Team 14",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(metrics.router)
app.include_router(samples.router)
app.include_router(inference.router)
app.include_router(human_eval.router)
app.include_router(mislabels.router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": len(data_store.results),
        "total_samples": sum(len(df) for df in data_store.results.values()),
    }
