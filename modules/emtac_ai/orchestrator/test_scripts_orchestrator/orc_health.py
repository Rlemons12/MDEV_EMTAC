# orchestrator_healthcheck.py
import sys, traceback

def main():
    try:
        from modules.emtac_ai.orchestrator.orchestrator import Orchestrator
    except Exception as e:
        print("[FAIL] Import error: modules.emtac_ai.orchestrator.orchestrator.Orchestrator")
        print("Reason:", e)
        print(traceback.format_exc())
        sys.exit(1)

    try:
        orch = Orchestrator(intent_model_dir=None, ner_model_dirs=None)  # let it use defaults
        print("[OK] Orchestrator constructed")

        if not hasattr(orch, "process_prompt"):
            print("[FAIL] Orchestrator has no process_prompt()")
            sys.exit(2)

        probe_q = "I need a cylinder"
        print("[INFO] Calling process_prompt(...) with:", probe_q)
        out = orch.process_prompt(probe_q)
        if isinstance(out, dict):
            print("[OK] process_prompt returned dict with keys:", sorted(out.keys()))
            print("[OK] Detected intent:", out.get("intent"))
            print("[OK] Result count:", len(out.get("results") or []))
            sys.exit(0)
        else:
            print("[WARN] process_prompt returned non-dict:", type(out))
            sys.exit(3)
    except Exception as e:
        print("[FAIL] Orchestrator runtime error:", e)
        print(traceback.format_exc())
        sys.exit(4)

if __name__ == "__main__":
    main()
