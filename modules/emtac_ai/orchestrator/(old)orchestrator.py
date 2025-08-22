# modules/emtac_ai/orchestrator/orchestrator.py

import os
from typing import Dict, Any, List, Optional

from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    warning_id,
    error_id,
    get_request_id,
    log_timed_operation,
)
from modules.configuration.config import (ORC_PROJECT_ROOT, ORC_INTENT_MODEL_DIR, ORC_PARTS_MODEL_DIR,ORC_IMAGES_MODEL_DIR,
    ORC_DOCUMENTS_MODEL_DIR, ORC_DRAWINGS_MODEL_DIR, ORC_TOOLS_MODEL_DIR, ORC_TROUBLESHOOTING_MODEL_DIR )

def to_abs_path(path: Optional[str]) -> Optional[str]:
    """
    Resolve a possibly-relative path against ORC_PROJECT_ROOT.
    Never join against the orchestrator module directory.
    """
    if not path:
        return None
    if os.path.isabs(path):
        return path
    # Prefer resolving from the project root
    candidate = os.path.abspath(os.path.join(ORC_PROJECT_ROOT, path))
    if os.path.exists(candidate):
        return candidate
    # Fallback: normalize whatever was passed
    return os.path.abspath(path)


class Orchestrator:
    """
    Orchestrates:
      1) Intent classification
      2) Intent-specific NER
      3) Adapter routing + search
    Now with detailed, request-scoped logging.
    """

    @with_request_id
    def __init__(
            self,
            intent_model_dir: Optional[str] = None,
            ner_model_dirs: Optional[Dict[str, str]] = None,
            intent_labels: Optional[List[str]] = None,
            ner_labels: Optional[Dict[str, List[str]]] = None,
    ):
        rid = get_request_id()
        info_id("=== Orchestrator initialization ===", rid)

        # -------- Defaults from unified project config --------
        if intent_model_dir is None:
            intent_model_dir = ORC_INTENT_MODEL_DIR

        if ner_model_dirs is None:
            ner_model_dirs = {
                "parts": ORC_PARTS_MODEL_DIR,
                "images": ORC_IMAGES_MODEL_DIR,
                "documents": ORC_DOCUMENTS_MODEL_DIR,
                "drawings": ORC_DRAWINGS_MODEL_DIR,
                "prints": ORC_DRAWINGS_MODEL_DIR,  # alias
                "tools": ORC_TOOLS_MODEL_DIR,
                "troubleshooting": ORC_TROUBLESHOOTING_MODEL_DIR,
            }

        # Normalize/resolve paths once (against ORC_PROJECT_ROOT)
        self.intent_model_dir = to_abs_path(intent_model_dir)
        self.ner_model_dirs = {k: to_abs_path(v) for k, v in (ner_model_dirs or {}).items()}

        # Extra visibility for troubleshooting
        debug_id(f"Requested intent dir: {intent_model_dir}", rid)
        debug_id(f"Resolved intent dir:  {self.intent_model_dir}", rid)
        debug_id(f"ORC_PROJECT_ROOT:     {ORC_PROJECT_ROOT}", rid)

        for k, v in self.ner_model_dirs.items():
            debug_id(f"NER[{k}] dir -> {v} (exists={os.path.exists(v) if v else False})", rid)

        # ---- Load intent classifier
        self.intent_classifier = None
        if self.intent_model_dir and os.path.exists(self.intent_model_dir):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
                with log_timed_operation("load_intent_classifier", rid):
                    tok = AutoTokenizer.from_pretrained(self.intent_model_dir)
                    mdl = AutoModelForSequenceClassification.from_pretrained(self.intent_model_dir)
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model=mdl,
                        tokenizer=tok,
                        trust_remote_code=False
                    )
                info_id(f"Intent classifier loaded from {self.intent_model_dir}", rid)
            except Exception as e:
                warning_id(f"Could not load intent classifier ({self.intent_model_dir}): {e}", rid)
        else:
            warning_id(
                f"Intent classifier path missing or not found: {self.intent_model_dir}", rid
            )

        # ---- Load NER pipelines per intent (unchanged logic, better logging)
        self.ner_models: Dict[str, Any] = {}
        for intent, ner_dir in self.ner_model_dirs.items():
            if ner_dir and os.path.exists(ner_dir):
                try:
                    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
                    with log_timed_operation(f"load_ner[{intent}]", rid):
                        tok = AutoTokenizer.from_pretrained(ner_dir)
                        mdl = AutoModelForTokenClassification.from_pretrained(ner_dir)
                        self.ner_models[intent] = pipeline(
                            "ner",
                            model=mdl,
                            tokenizer=tok,
                            aggregation_strategy="simple",
                            trust_remote_code=False
                        )
                    info_id(f"NER model loaded for intent '{intent}' from {ner_dir}", rid)
                except Exception as e:
                    warning_id(f"Could not load NER model for '{intent}' at {ner_dir}: {e}", rid)
                    self.ner_models[intent] = None
            else:
                warning_id(f"NER model directory does not exist for '{intent}': {ner_dir}", rid)
                self.ner_models[intent] = None

        # ---- Import adapters (unchanged) ----
        try:
            from modules.emtac_ai.orchestrator.adpators.base_search_adapter import (
                PartsSearchAdapter,
                DrawingsSearchAdapter,
            )
        except Exception as e:
            error_id(f"Failed to import search adapters: {e}", rid)
            raise

        self.adapter_map = {
            "parts": PartsSearchAdapter,
            "prints": DrawingsSearchAdapter,
            "drawing": DrawingsSearchAdapter,
            "drawings": DrawingsSearchAdapter,
        }
        debug_id(f"Adapter map: {list(self.adapter_map.keys())}", rid)
        info_id("=== Orchestrator initialization complete ===", rid)

    # ---------------- Normalization helpers ----------------

    @with_request_id
    def _normalize_intent_label(self, raw_label: str) -> str:
        rid = get_request_id()
        lab_in = (raw_label or "").strip()
        lab = lab_in.lower()
        if lab in ("drawing", "drawings", "print", "prints"):
            lab = "drawings"
        elif lab in ("part", "parts"):
            lab = "parts"
        debug_id(f"_normalize_intent_label: '{lab_in}' -> '{lab}'", rid)
        return lab

    @with_request_id
    def _normalize_entities(self, ner_output: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Convert HF pipeline NER output into {LABEL: [values]}.
        We use 'entity_group' (the aggregated label) and 'word' text.
        """
        rid = get_request_id()
        ent_map: Dict[str, List[str]] = {}
        for e in ner_output or []:
            label = str(e.get("entity_group", "")).strip().upper()
            text = str(e.get("word", "")).strip()
            if not label or not text:
                continue
            ent_map.setdefault(label, []).append(text)
        debug_id(f"_normalize_entities: {ent_map}", rid)
        return ent_map

    @with_request_id
    def _pick_adapter(self, intent_label: str):
        rid = get_request_id()
        Adapter = self.adapter_map.get(intent_label)
        if Adapter:
            debug_id(f"_pick_adapter: intent='{intent_label}' -> {Adapter.__name__}", rid)
        else:
            warning_id(f"_pick_adapter: no adapter registered for intent '{intent_label}'", rid)
        return Adapter

    # ---------------- Core processing ----------------

    @with_request_id
    def process_prompt(self, text: str) -> Dict[str, Any]:
        """
        Process a text prompt:
          1) Intent classification
          2) NER for that intent (if available)
          3) Route to adapter and call search()
        """
        rid = get_request_id()
        info_id(f"process_prompt: received text (len={len(text) if text else 0})", rid)

        if not text or not text.strip():
            warning_id("process_prompt: empty/blank input", rid)
            return {'intent': None, 'confidence': 0.0, 'entities': {}, 'results': []}

        if not self.intent_classifier:
            warning_id("process_prompt: Intent classifier not loaded; returning empty result", rid)
            return {'intent': None, 'confidence': 0.0, 'entities': {}, 'results': []}

        # 1) Intent
        intent = None
        confidence = 0.0
        with log_timed_operation("intent_classification", rid):
            try:
                results = self.intent_classifier(text)
                debug_id(f"intent_raw: {results}", rid)
                if results:
                    intent = results[0].get('label')
                    confidence = float(results[0].get('score', 0.0))
                    info_id(f"intent: '{intent}' (confidence={confidence:.3f})", rid)
            except Exception as e:
                error_id(f"Intent classification error: {e}", rid)

        norm_intent = self._normalize_intent_label(intent or "")

        # 2) NER for that intent
        entities_norm: Dict[str, List[str]] = {}
        ner_pipe = self.ner_models.get(norm_intent)
        if ner_pipe:
            with log_timed_operation(f"ner[{norm_intent}]", rid):
                try:
                    ner_raw = ner_pipe(text) or []
                    debug_id(f"ner_raw[{norm_intent}]: {ner_raw}", rid)
                    entities_norm = self._normalize_entities(ner_raw)
                except Exception as e:
                    warning_id(f"NER extraction failed for intent '{norm_intent}': {e}", rid)
        else:
            debug_id(f"No NER pipeline available for intent '{norm_intent}'", rid)

        # 3) Route to adapter and search
        results_payload: List[Dict[str, Any]] = []
        AdapterCls = self._pick_adapter(norm_intent)
        if AdapterCls is not None:
            with log_timed_operation(f"adapter_search[{norm_intent}]", rid):
                try:
                    adapter = AdapterCls()  # should open its own DB session if needed
                    debug_id(f"Calling adapter.search with entities={entities_norm}", rid)
                    results_payload = adapter.search(text, entities_norm)
                    info_id(f"adapter_search[{norm_intent}] returned {len(results_payload)} result(s)", rid)
                except Exception as e:
                    warning_id(f"Adapter search failed for intent '{norm_intent}': {e}", rid)
        else:
            info_id(f"No adapter for intent '{norm_intent}'; returning no results", rid)

        payload = {
            'intent': norm_intent or intent,
            'confidence': confidence,
            'entities': entities_norm,
            'results': results_payload
        }
        debug_id(f"process_prompt: final payload keys={list(payload.keys())}", rid)
        return payload


# Manual sanity quick-test (optional)
if __name__ == "__main__":
    ner_dirs = {
        "parts": "modules/emtac_ai/models/parts",
        "images": "modules/emtac_ai/models/images",
        "documents": "modules/emtac_ai/models/documents",
        "prints": "modules/emtac_ai/models/prints",
        "tools": "modules/emtac_ai/models/tools",
        "troubleshooting": "modules/emtac_ai/models/troubleshooting"
    }

    orch = Orchestrator(
        intent_model_dir="modules/emtac_ai/models/intent_classifier",
        ner_model_dirs=ner_dirs,
    )

    prompt = input("Enter your question: ")
    result = orch.process_prompt(prompt)
    print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
    print("Entities:", result["entities"])
    print("Results:", result["results"])
