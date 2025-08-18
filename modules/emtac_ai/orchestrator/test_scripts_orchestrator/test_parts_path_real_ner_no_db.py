#!/usr/bin/env python3
"""
DB-free wiring test that uses the REAL Parts NER model.
- Forces intent='parts' (fake intent classifier)
- Loads your Parts NER from --ner-dir
- Uses a FakePartsAdapter to capture that orchestrator calls .search()
- No database or Part ORM required

Run:
  python -m modules.emtac_ai.orchestrator.tests.test_parts_path_real_ner_no_db \
    --ner-dir modules/emtac_ai/models/parts \
    --prompt "Do you stock A101576 filters from Balston?"

Exit code 0 = PASS
"""

import argparse
import sys
import traceback
from typing import Any, Dict, List

from modules.emtac_ai.orchestrator.orchestrator import Orchestrator

# ---------------- Real NER loader (HF) ----------------
def load_real_parts_ner(ner_dir: str):
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    tok = AutoTokenizer.from_pretrained(ner_dir)
    mdl = AutoModelForTokenClassification.from_pretrained(ner_dir)
    return pipeline(
        "ner",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy="simple",
        trust_remote_code=False
    )

# ---------------- Fakes (intent + adapter) -----------

class FakeIntentClassifier:
    """Always returns 'parts' with high confidence."""
    def __call__(self, text: str) -> List[Dict[str, Any]]:
        return [{"label": "parts", "score": 0.99}]

class FakePartsAdapter:
    """
    Stand-in for the real PartsSearchAdapter.
    We only care that orchestrator calls .search() with the real NER entities.
    """
    calls: List[Dict[str, Any]] = []

    def __init__(self, request_id: str | None = None):
        self.request_id = request_id

    def search(self, query: str, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        FakePartsAdapter.calls.append({"query": query, "entities": entities, "request_id": self.request_id})
        # minimal sanity checks
        assert isinstance(entities, dict), "entities must be a dict"
        # return a synthetic result echoing key info so we can inspect
        return [{
            "type": "part",
            "source": "FakePartsAdapter",
            "score": 1.0,
            "echo_query": query,
            "entities_seen": list(entities.keys())
        }]

# ---------------- Test runner -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ner-dir", required=True, help="Path to your trained Parts NER model directory")
    ap.add_argument("--prompt", default="Do you stock A101576 filters from Balston?",
                    help="Test prompt to run through the pipeline")
    args = ap.parse_args()

    try:
        # Build orchestrator but override intent + parts NER + adapter
        orch = Orchestrator(intent_model_dir=None, ner_model_dirs={})
        orch.intent_classifier = FakeIntentClassifier()

        # Load REAL Parts NER
        parts_ner = load_real_parts_ner(args.ner_dir)
        orch.ner_models["parts"] = parts_ner

        # Route 'parts' to our fake adapter so no DB is touched
        orch.adapter_map["parts"] = FakePartsAdapter

        # Drive the pipeline
        out = orch.process_prompt(args.prompt)

        # Assertions focused on wiring (not NER quality):
        if out.get("intent") != "parts":
            raise AssertionError(f"intent should be 'parts', got: {out.get('intent')}")
        if not isinstance(out.get("entities"), dict):
            raise AssertionError("entities should be a dict")
        if not FakePartsAdapter.calls:
            raise AssertionError("Adapter.search was not called")
        if not isinstance(out.get("results"), list):
            raise AssertionError("results should be a list")

        # Print a compact summary
        print("✅ PASS: parts pathway wired with REAL NER")
        print("Intent:", out["intent"])
        print("Entities (keys):", list(out["entities"].keys()))
        print("Result sample:", out["results"][0])
        return 0

    except Exception as e:
        print("❌ FAIL:", e)
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())
