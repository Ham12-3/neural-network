"""Unit tests for the summariser API."""

import pytest
from fastapi.testclient import TestClient

from .main import app, summariser as _check

client = TestClient(app)


def _model_loaded() -> bool:
    """Check whether the summarisation model is available."""
    resp = client.get("/health")
    return resp.json().get("model_loaded", False)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_empty_text_rejected():
    resp = client.post("/summarise", json={"text": ""})
    assert resp.status_code == 422


def test_whitespace_only_rejected():
    resp = client.post("/summarise", json={"text": "   "})
    assert resp.status_code == 422


def test_missing_text_rejected():
    resp = client.post("/summarise", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Summarisation (integration – requires model to be loaded via lifespan)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not True, reason="always run with lifespan context")
def test_summarise_basic():
    sample = (
        "Artificial intelligence has transformed many industries. "
        "Machine learning models can now process natural language, "
        "recognise images, and generate creative content. "
        "Companies across the world invest billions of dollars each year "
        "in AI research and development."
    )
    resp = client.post("/summarise", json={"text": sample})
    if resp.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert len(data["summary"]) > 0
    assert "model" in data
    assert "took_ms" in data
    assert data["took_ms"] >= 0


def test_summarise_with_max_words():
    sample = (
        "The global economy has faced significant challenges in recent years. "
        "Supply chain disruptions, inflation, and geopolitical tensions have "
        "created uncertainty across markets. Central banks have responded with "
        "interest rate adjustments while governments have implemented fiscal "
        "policies to support growth and employment."
    )
    resp = client.post("/summarise", json={"text": sample, "max_words": 30})
    if resp.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["summary"]) > 0


def test_summarise_clamps_max_words():
    """max_words outside the allowed range should be clamped, not rejected."""
    sample = "Some reasonably long text that can be summarised by the model."
    resp = client.post("/summarise", json={"text": sample, "max_words": 9999})
    if resp.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert resp.status_code == 200
