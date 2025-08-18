import os
from modules.emtac_ai.training_module.training_mod import IntentTrainer  # Adjust import path as needed
from modules.emtac_ai.config import ORC_INTENT_TRAIN_DATA_DIR, ORC_INTENT_MODEL_DIR
def main():
    # Build paths from config
    train_data_path = os.path.join(ORC_INTENT_TRAIN_DATA_DIR, "intent_train.jsonl")
    output_dir = ORC_INTENT_MODEL_DIR

    intent_labels = ["parts", "images", "documents", "prints", "tools", "troubleshooting"]

    trainer = IntentTrainer(base_model_dir="distilbert-base-uncased", labels=intent_labels)
    trainer.train(
        train_data_path=train_data_path,
        output_dir=output_dir,
        epochs=3,
        batch_size=8
    )


if __name__ == "__main__":
    main()
