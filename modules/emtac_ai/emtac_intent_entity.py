import os
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

def to_abs_path(path, base_dir):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


class IntentEntityPlugin:
    def __init__(self, intent_model_dir=None, ner_model_dir=None, intent_labels=None, ner_labels=None):
        """
        intent_model_dir: Path to intent classifier model directory (relative or absolute)
        ner_model_dir: Path to NER model directory (relative or absolute)
        intent_labels: List of intent labels
        ner_labels: List of NER labels
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Normalize paths if provided
        self.intent_model_dir = to_abs_path(intent_model_dir, base_dir) if intent_model_dir else None
        self.ner_model_dir = to_abs_path(ner_model_dir, base_dir) if ner_model_dir else None

        # Auto-correct intent_model_dir if checkpoint folder exists (common with HuggingFace saving)
        if self.intent_model_dir and not os.path.exists(os.path.join(self.intent_model_dir, "config.json")):
            for f in os.scandir(self.intent_model_dir):
                if f.is_dir() and os.path.exists(os.path.join(f.path, "config.json")):
                    print(f"Auto-detected intent model checkpoint folder: {f.path}")
                    self.intent_model_dir = f.path
                    break

        # Same auto-correction for ner_model_dir
        if self.ner_model_dir and not os.path.exists(os.path.join(self.ner_model_dir, "config.json")):
            for f in os.scandir(self.ner_model_dir):
                if f.is_dir() and os.path.exists(os.path.join(f.path, "config.json")):
                    print(f"Auto-detected NER model checkpoint folder: {f.path}")
                    self.ner_model_dir = f.path
                    break

        # Labels mapping (defaults if none provided)
        self.intent_labels = intent_labels or ["parts", "images", "documents", "prints", "tools", "troubleshooting"]
        self.intent_id2label = {i: label for i, label in enumerate(self.intent_labels)}
        self.intent_label2id = {label: i for i, label in enumerate(self.intent_labels)}

        self.ner_labels = ner_labels or ["O", "B-PARTDESC", "B-PARTNUM"]
        self.ner_id2label = {i: label for i, label in enumerate(self.ner_labels)}
        self.ner_label2id = {label: i for i, label in enumerate(self.ner_labels)}

        # Initialize pipelines to None (lazy loading)
        self.intent_classifier = None
        self.ner = None

        # Load intent classifier pipeline if model directory exists and has model files
        if self.intent_model_dir and os.path.exists(self.intent_model_dir):
            try:
                config_path = os.path.join(self.intent_model_dir, "config.json")
                model_files = [f for f in os.listdir(self.intent_model_dir) if f.endswith(('.bin', '.safetensors'))]
                if os.path.exists(config_path) and model_files:
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model=self.intent_model_dir,
                        local_files_only=True,
                        trust_remote_code=False
                    )
                else:
                    print(f"Warning: Intent model files not found in {self.intent_model_dir}")
            except Exception as e:
                print(f"Warning: Could not load intent classifier from {self.intent_model_dir}: {e}")

        # Load NER pipeline if model directory exists and has model files
        if self.ner_model_dir and os.path.exists(self.ner_model_dir):
            try:
                config_path = os.path.join(self.ner_model_dir, "config.json")
                model_files = [f for f in os.listdir(self.ner_model_dir) if f.endswith(('.bin', '.safetensors'))]
                if os.path.exists(config_path) and model_files:
                    self.ner = pipeline(
                        "ner",
                        model=self.ner_model_dir,
                        aggregation_strategy="simple",
                        local_files_only=True,
                        trust_remote_code=False
                    )
                else:
                    print(f"Warning: NER model files not found in {self.ner_model_dir}")
            except Exception as e:
                print(f"Warning: Could not load NER model from {self.ner_model_dir}: {e}")

    def classify_intent(self, text):
        """Return intent label and confidence score"""
        if not self.intent_classifier:
            print("Warning: Intent classifier not loaded")
            return None, 0.0
        try:
            results = self.intent_classifier(text)
            if results:
                return results[0]['label'], results[0]['score']
        except Exception as e:
            print(f"Error during intent classification: {e}")
        return None, 0.0

    def extract_entities(self, text):
        """Extract entities and map generic labels to your labels"""
        if not self.ner:
            print("Warning: NER model not loaded")
            return []
        try:
            raw_entities = self.ner(text)
            for ent in raw_entities:
                entity_group = ent.get('entity_group', None)
                if entity_group and entity_group.startswith("LABEL_"):
                    label_id = int(entity_group.split("_")[1])
                    ent['entity_group'] = self.ner_id2label.get(label_id, entity_group)
            return raw_entities
        except Exception as e:
            print(f"Error during entity extraction: {e}")
            return []

    @staticmethod
    def load_data_from_jsonl(path):
        """Load dataset from a JSONL file"""
        return load_dataset('json', data_files=path)['train']

    def train_ner(self, train_data, output_dir="models/ner-custom", epochs=3):
        """Fine-tune NER model"""
        if not self.ner_model_dir or not os.path.exists(self.ner_model_dir):
            print("Error: Base NER model directory not found for training")
            return

        tokenizer = AutoTokenizer.from_pretrained(self.ner_model_dir)  # no local_files_only here
        model = AutoModelForTokenClassification.from_pretrained(
            self.ner_model_dir,
            num_labels=len(self.ner_labels),
            id2label=self.ner_id2label,
            label2id=self.ner_label2id,
            ignore_mismatched_sizes=True
        )

        def tokenize_and_align_labels(example):
            tokenized_inputs = tokenizer(
                example["tokens"],
                truncation=True,
                padding="max_length",
                max_length=128,
                is_split_into_words=True,
                return_offsets_mapping=True
            )
            labels = []
            word_ids = tokenized_inputs.word_ids()
            prev_word_id = None
            label_ids = example["ner_tags"]
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id != prev_word_id:
                    labels.append(label_ids[word_id])
                else:
                    labels.append(label_ids[word_id])
                prev_word_id = word_id
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        if isinstance(train_data, str):
            dataset = self.load_data_from_jsonl(train_data)
        else:
            dataset = train_data

        tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False, remove_columns=dataset.column_names)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=10,
            save_total_limit=2,
            logging_steps=5,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned NER model saved to: {output_dir}")

    def train_intent(self, train_data, output_dir="models/intent-custom", epochs=3):
        """Fine-tune Intent classifier"""
        if not self.intent_model_dir or not os.path.exists(self.intent_model_dir):
            print("Error: Base intent model directory not found for training")
            return

        tokenizer = AutoTokenizer.from_pretrained(self.intent_model_dir)  # no local_files_only here
        model = AutoModelForSequenceClassification.from_pretrained(
            self.intent_model_dir,
            num_labels=len(self.intent_labels),
            id2label=self.intent_id2label,
            label2id=self.intent_label2id,
            ignore_mismatched_sizes=True
        )

        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, padding=True)

        if isinstance(train_data, str):
            dataset = self.load_data_from_jsonl(train_data)
        else:
            dataset = train_data

        # Map intent strings to label ids
        def map_intent(example):
            example["label"] = self.intent_label2id[example["intent"]]
            return example

        dataset = dataset.map(map_intent)

        tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            save_steps=10,
            save_total_limit=2,
            logging_steps=5,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned Intent model saved to: {output_dir}")
