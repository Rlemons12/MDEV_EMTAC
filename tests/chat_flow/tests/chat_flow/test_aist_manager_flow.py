import pytest
from modules.emtac_ai.aist_manager import AistManager


def test_aist_manager_flow(monkeypatch):
    """
    End-to-end smoke test:
    - Initializes AistManager
    - Mocks UnifiedSearch backend to return a fake result
    - Verifies answer_question produces a formatted response
    """

    mgr = AistManager()

    # Patch execute_unified_search to avoid hitting real DB/models
    monkeypatch.setattr(
        mgr,
        "execute_unified_search",
        lambda question, user_id=None, request_id=None: {
            "status": "success",
            "results": [{"type": "part", "part_number": "MOCK-123"}],
            "search_method": "orchestrator",
            "total_results": 1,
            "detected_intent": "find_part",
            "entities": {"part_number": "MOCK-123"}
        }
    )

    out = mgr.answer_question("tester", "find MOCK-123")

    # Assertions
    assert out["status"] == "success"
    assert "MOCK-123" in out["answer"]
    assert out["results"]["total_results"] == 1
    assert out["detected_intent"] == "find_part"
    assert out["entities"]["part_number"] == "MOCK-123"
