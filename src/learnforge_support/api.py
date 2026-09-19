"""FastAPI HTTP application for LearnForge Support Assistant.

Exposes:
- POST /chat: Conversational support endpoint with grounding and citation validation.
- GET /health: Health check, index status, and model metadata.
HTTP layer only. Contains no retrieval, ranking, or prompt logic.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from learnforge_support.config import Settings, get_settings
from learnforge_support.observability import flush_observability, initialize_observability
from learnforge_support.schemas import ChatRequest, ChatResponse, HealthResponse
from learnforge_support.service import SupportService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initializes models and shared resources once per application lifecycle."""
    settings = get_settings()
    initialize_observability(settings)
    service = SupportService(settings=settings)
    app.state.service = service
    try:
        yield
    finally:
        flush_observability()


app = FastAPI(
    title="LearnForge Support Assistant API",
    description="Production-minded AI Customer Support RAG Assistant for LearnForge.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_support_service() -> SupportService:
    """FastAPI dependency yielding the shared SupportService instance."""
    service: SupportService | None = getattr(app.state, "service", None)
    if service is None:
        settings = get_settings()
        service = SupportService(settings=settings)
        app.state.service = service
    return service


@app.get("/health", response_model=HealthResponse)
def health(
    service: Annotated[SupportService, Depends(get_support_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Returns application health, Qdrant index readiness, and model configuration."""
    is_ready = service.is_index_ready()
    record_count = service.get_indexed_record_count()
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        index_ready=is_ready,
        record_count=record_count,
        llm_model=settings.llm_model,
        dense_model=settings.dense_model,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: Annotated[SupportService, Depends(get_support_service)],
) -> ChatResponse:
    """Processes incoming customer messages through the RAG pipeline."""
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Customer message cannot be empty.",
        )

    try:
        response, _ = service.process_chat(request)
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {exc}",
        ) from exc
