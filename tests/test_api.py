"""Integration tests for FastAPI endpoints (/health and /chat).

Uses FastAPI TestClient to verify HTTP status codes, request/response contracts,
multi-turn state preservation, and safety behaviors.
"""

from fastapi.testclient import TestClient

from learnforge_support.api import app, get_support_service
from learnforge_support.config import get_settings
from learnforge_support.service import SupportService


class ControlledMockLLM:
    """Mock LLM returning predictable structured responses for route testing."""

    def __init__(self) -> None:
        self.mock_decision: str = (
            '{"decision": "answer", "message": "Eligible refunds are available within 14 days.", '
            '"reason_code": "grounded_answer", "citations": ["POLICY-02"], "handoff_summary": null}'
        )

    def complete_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        return self.mock_decision


def test_health_endpoint():
    """Verify /health returns HTTP 200 and index availability information."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "index_ready" in data
        assert "record_count" in data
        assert "llm_model" in data


def test_chat_empty_message_rejected():
    """Verify empty user messages return HTTP 422 error."""
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "   "})
        assert response.status_code == 422


def test_chat_endpoint_success():
    """Verify /chat processes query, returns typed ChatResponse, and validates citations."""
    settings = get_settings()
    mock_llm = ControlledMockLLM()
    service = SupportService(settings=settings, llm_client=mock_llm)
    app.dependency_overrides[get_support_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={"message": "How do course refunds work?"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["decision"] == "answer"
            assert data["reason_code"] == "grounded_answer"
            assert len(data["citations"]) > 0
            assert data["citations"][0]["record_id"] == "POLICY-02"
            assert "title" in data["citations"][0]
            assert "session_id" in data
    finally:
        app.dependency_overrides.clear()


def test_chat_multiturn_conversation_preserves_session():
    """Verify multiple turns using the same session_id are retained in history."""
    settings = get_settings()
    mock_llm = ControlledMockLLM()
    service = SupportService(settings=settings, llm_client=mock_llm)
    app.dependency_overrides[get_support_service] = lambda: service

    try:
        with TestClient(app) as client:
            # Turn 1
            res1 = client.post(
                "/chat",
                json={"session_id": "test-session-123", "message": "Can I get a refund?"},
            )
            assert res1.status_code == 200

            # Turn 2
            res2 = client.post(
                "/chat",
                json={"session_id": "test-session-123", "message": "What about Apple purchases?"},
            )
            assert res2.status_code == 200

            # Verify session history contains both turns in conversation store
            history = service.conversation_store.get_history("test-session-123")
            assert len(history) == 4  # 2 user turns + 2 assistant turns
            assert history[0]["content"] == "Can I get a refund?"
            assert history[2]["content"] == "What about Apple purchases?"
    finally:
        app.dependency_overrides.clear()
