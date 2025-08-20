import os
import random
import math
import torch
from torch.utils.data import IterableDataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from modules.configuration.config_env import DatabaseConfig
from seqeval.metrics import f1_score, precision_score, recall_score

# ===================== Label schema (customize as needed) =====================
LABELS = ["O", "B-PART", "I-PART"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# =============================== DB Setup ====================================
db_conf = DatabaseConfig()
engine = db_conf.get_engine()
# expire_on_commit=False avoids state expiration between our short-lived sessions
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# ============================ Helper Functions ===============================
PAGE_SIZE = 1000       # page size for full-dataset streaming
ID_CHUNK_SIZE = 1000   # chunk size when fetching by id list


def _fetch_random_ids(session, n: int) -> list:
    rows = session.execute(
        text(
            "SELECT id FROM training_sample "
            "WHERE sample_type='ner' ORDER BY random() LIMIT :n"
        ),
        {"n": n},
    ).fetchall()
    return [r[0] for r in rows]


def _fetch_rows_by_ids(session, id_list: list):
    return session.execute(
        text(
            "SELECT id, text, entities FROM training_sample "
            "WHERE id = ANY(:ids)"
        ),
        {"ids": id_list},
    ).fetchall()


def _fetch_rows_page(session, last_id: int, limit: int):
    return session.execute(
        text(
            "SELECT id, text, entities FROM training_sample "
            "WHERE sample_type='ner' AND id > :last_id ORDER BY id LIMIT :lim"
        ),
        {"last_id": last_id, "lim": limit},
    ).fetchall()


def convert_example_to_ner_format(text_val: str, entities):
    entities = entities or []
    words = text_val.split()
    word_starts, pos = [], 0
    for w in words:
        s = text_val.find(w, pos)
        word_starts.append(s)
        pos = s + len(w)
    labels = ["O"] * len(words)
    for ent in entities:
        e_start, e_end = ent.get("start"), ent.get("end")
        e_type = ent.get("label") or ent.get("entity")
        if e_start is None or e_end is None or not e_type:
            continue
        first = last = None
        for i, w_start in enumerate(word_starts):
            w_end = w_start + len(words[i])
            if w_start < e_end and w_end > e_start:
                if first is None:
                    first = i
                last = i
        if first is not None:
            labels[first] = f"B-{e_type}"
            for i in range(first + 1, (last or first) + 1):
                labels[i] = f"I-{e_type}"
    label_ids = [LABEL2ID.get(l, 0) for l in labels]
    return {"tokens": words, "ner_tags": label_ids}


def tokenize_example(tokenizer, example, max_length=128):
    tok = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = tok.word_ids()
    labels, prev = [], None
    for wid in word_ids:
        if wid is None:
            labels.append(-100)
        else:
            lab = example["ner_tags"][wid]
            if wid != prev:
                labels.append(lab)
            else:
                if LABELS[lab].startswith("B-"):
                    inside = "I-" + LABELS[lab][2:]
                    labels.append(LABEL2ID.get(inside, lab))
                else:
                    labels.append(lab)
            prev = wid
    return {
        "input_ids": tok["input_ids"].squeeze(),
        "attention_mask": tok["attention_mask"].squeeze(),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def compute_token_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    true_tags, pred_tags = [], []
    for p_row, l_row in zip(preds, labels):
        p_seq, l_seq = [], []
        for p_i, l_i in zip(p_row, l_row):
            li = int(l_i)
            if li == -100:
                continue
            p_seq.append(ID2LABEL[int(p_i)])
            l_seq.append(ID2LABEL[li])
        if l_seq:
            true_tags.append(l_seq)
            pred_tags.append(p_seq)

    return {
        "precision": precision_score(true_tags, pred_tags),
        "recall": recall_score(true_tags, pred_tags),
        "f1": f1_score(true_tags, pred_tags),
    }

# ================================ Datasets ===================================
class DBStreamingNERDataset(IterableDataset):
    """Streaming dataset that reads from Postgres with short-lived sessions.
       Resilient to dropped connections via retry logic.
    """

    def __init__(self, tokenizer, session_factory, max_length=128,
                 max_examples=None, shuffle_buffer_size=1000,
                 skip_examples=0, epoch=0, request_id=None,
                 exclude_ids=None, include_only_ids=None):
        self.tokenizer = tokenizer
        self.session_factory = session_factory
        self.max_length = max_length
        self.max_examples = max_examples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.skip_examples = skip_examples
        self.epoch = epoch
        self.request_id = request_id
        self.exclude_ids = set(exclude_ids or [])
        self.include_only_ids = set(include_only_ids or [])
        # Pre-compute a length for Trainer (__len__)
        s = self.session_factory()
        try:
            if self.include_only_ids:
                self._length = s.execute(
                    text("SELECT COUNT(*) FROM training_sample WHERE sample_type='ner' AND id = ANY(:ids)"),
                    {"ids": list(self.include_only_ids)},
                ).scalar_one()
            elif self.max_examples is not None:
                self._length = int(self.max_examples)
            else:
                total = s.execute(
                    text("SELECT COUNT(*) FROM training_sample WHERE sample_type='ner'")
                ).scalar_one()
                self._length = total - len(self.exclude_ids)
        finally:
            s.close()

    def __len__(self):
        return max(1, int(self._length))

    def __iter__(self):
        random.seed(42 + self.epoch)
        buffer = []

        def _yield_buffer():
            nonlocal buffer
            random.shuffle(buffer)
            for item in buffer:
                yield item
            buffer = []

        produced = 0

        # ---------- Case 1: include-only IDs (eval set for full runs) ----------
        if self.include_only_ids:
            id_list = list(self.include_only_ids)
            for i in range(0, len(id_list), ID_CHUNK_SIZE):
                chunk_ids = id_list[i:i + ID_CHUNK_SIZE]
                retry = 0
                while True:
                    s = None
                    try:
                        s = self.session_factory()
                        rows = _fetch_rows_by_ids(s, chunk_ids)
                        try:
                            s.close()
                        except Exception:
                            pass
                        break
                    except OperationalError:
                        try:
                            if s is not None:
                                s.close()
                        except Exception:
                            pass
                        retry += 1
                        if retry > 3:
                            raise
                        continue
                for _, text_val, entities_val in rows:
                    ex = convert_example_to_ner_format(text_val, entities_val)
                    tok = tokenize_example(self.tokenizer, ex, self.max_length)
                    buffer.append(tok)
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield from _yield_buffer()
            if buffer:
                yield from _yield_buffer()
            return

        # ---------- Case 2: Small / Medium (random sample) ----------
        if self.max_examples:
            target = int(self.max_examples)
            # one short session to get random ids
            s = None
            try:
                s = self.session_factory()
                sample_ids = _fetch_random_ids(s, target + len(self.exclude_ids))
            finally:
                try:
                    if s is not None:
                        s.close()
                except Exception:
                    pass
            if self.exclude_ids:
                sample_ids = [i for i in sample_ids if i not in self.exclude_ids][:target]

            for i in range(0, len(sample_ids), ID_CHUNK_SIZE):
                chunk_ids = sample_ids[i:i + ID_CHUNK_SIZE]
                retry = 0
                while True:
                    s = None
                    try:
                        s = self.session_factory()
                        rows = _fetch_rows_by_ids(s, chunk_ids)
                        try:
                            s.close()
                        except Exception:
                            pass
                        break
                    except OperationalError:
                        try:
                            if s is not None:
                                s.close()
                        except Exception:
                            pass
                        retry += 1
                        if retry > 3:
                            raise
                        continue
                for _, text_val, entities_val in rows:
                    ex = convert_example_to_ner_format(text_val, entities_val)
                    tok = tokenize_example(self.tokenizer, ex, self.max_length)
                    buffer.append(tok)
                    produced += 1
                    if len(buffer) >= self.shuffle_buffer_size:
                        yield from _yield_buffer()
                    if produced >= target:
                        if buffer:
                            yield from _yield_buffer()
                        return
            if buffer:
                yield from _yield_buffer()
            return

        # ---------- Case 3: Full dataset (paged by id) ----------
        last_id = 0
        while True:
            rows = None
            retry = 0
            while True:
                s = None
                try:
                    s = self.session_factory()
                    rows = _fetch_rows_page(s, last_id, PAGE_SIZE)
                    try:
                        s.close()
                    except Exception:
                        pass
                    break
                except OperationalError:
                    try:
                        if s is not None:
                            s.close()
                    except Exception:
                        pass
                    retry += 1
                    if retry > 3:
                        raise
                    continue

            if not rows:
                break

            for rid, text_val, entities_val in rows:
                if self.exclude_ids and rid in self.exclude_ids:
                    last_id = rid
                    continue
                ex = convert_example_to_ner_format(text_val, entities_val)
                tok = tokenize_example(self.tokenizer, ex, self.max_length)
                buffer.append(tok)
                if len(buffer) >= self.shuffle_buffer_size:
                    yield from _yield_buffer()
                last_id = rid

        if buffer:
            yield from _yield_buffer()


class ListNERDataset(Dataset):
    def __init__(self, tokenizer, rows, max_length=128):
        self.tokenizer = tokenizer
        self.rows = rows
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        _, text_val, entities_val = self.rows[idx]
        ex = convert_example_to_ner_format(text_val, entities_val)
        return tokenize_example(self.tokenizer, ex, self.max_length)

# =============================== Utilities ===================================

def choose_dataset_size():
    print("Select training dataset size:")
    print("1) Small (≈1,000 samples)")
    print("2) Medium (≈10,000 samples)")
    print("3) Full (all available samples)")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "1":
        return 1000
    elif choice == "2":
        return 10000
    else:
        return None  # full dataset


def count_ner_rows(session_factory) -> int:
    s = session_factory()
    try:
        return s.execute(
            text("SELECT COUNT(*) FROM training_sample WHERE sample_type='ner'")
        ).scalar_one()
    finally:
        s.close()

# ================================= Main =====================================

def maybe_freeze_encoder(model, freeze=True, unfreeze_last_n=2):
    if not freeze:
        return
    # Freeze all encoder layers
    for name, param in model.named_parameters():
        if name.startswith("bert.embeddings") or name.startswith("bert.encoder.layer"):
            param.requires_grad = False
    # Unfreeze last N blocks (for BERT base: 12 layers 0..11)
    for i in range(11, 11 - unfreeze_last_n, -1):
        for name, param in model.named_parameters():
            if f"bert.encoder.layer.{i}." in name:
                param.requires_grad = True


def main():
    # Prompt user for dataset size
    max_examples = choose_dataset_size()

    # Hyperparameters
    per_device_bs = 16
    num_epochs = 3
    grad_accum = 1
    eval_frac = 0.10  # 10% evaluation split

    # Determine pool and split
    session = SessionLocal()
    try:
        if max_examples is None:
            # FULL: eval = random 10% of ids; train = stream excluding those ids
            total_examples = count_ner_rows(SessionLocal)
            eval_count = max(1, int(total_examples * eval_frac))
            eval_ids = [r[0] for r in session.execute(
                text("SELECT id FROM training_sample WHERE sample_type='ner' ORDER BY random() LIMIT :k"),
                {"k": eval_count},
            ).fetchall()]

            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            train_dataset = DBStreamingNERDataset(
                tokenizer=tok,
                session_factory=SessionLocal,
                max_length=128,
                shuffle_buffer_size=1000,
                max_examples=None,
                exclude_ids=set(eval_ids),
            )
            eval_rows = session.execute(
                text("SELECT id, text, entities FROM training_sample WHERE id = ANY(:ids)"),
                {"ids": eval_ids},
            ).fetchall()
            eval_dataset = ListNERDataset(tok, eval_rows, max_length=128)
        else:
            # SMALL/MEDIUM: sample N rows, then split 90/10 in-memory
            pool_rows = session.execute(
                text("SELECT id, text, entities FROM training_sample WHERE sample_type='ner' ORDER BY random() LIMIT :n"),
                {"n": max_examples},
            ).fetchall()
            random.shuffle(pool_rows)
            cut = max(1, int(len(pool_rows) * (1 - eval_frac)))
            train_rows = pool_rows[:cut]
            eval_rows = pool_rows[cut:]

            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            train_dataset = ListNERDataset(tok, train_rows, max_length=128)
            eval_dataset = ListNERDataset(tok, eval_rows, max_length=128)
            total_examples = len(train_rows)
    finally:
        session.close()

    # Compute max_steps
    if max_examples is None:
        # for full runs, recompute total based on exclude_ids
        total_examples = count_ner_rows(SessionLocal) - len(eval_rows)
    steps_per_epoch = max(1, math.ceil(total_examples / (per_device_bs * grad_accum)))
    max_steps = steps_per_epoch * num_epochs

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    # Optional: reduce capacity for small/medium to reduce overfitting
    if max_examples is not None:
        maybe_freeze_encoder(model, freeze=True, unfreeze_last_n=2)

    training_args = TrainingArguments(
        output_dir="./ner_model",
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,          # for logs
        max_steps=max_steps,                  # IterableDataset needs explicit steps
        # ---- eval/checkpoint cadence ----
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        # ---- regularization / generalization ----
        weight_decay=0.01,
        warmup_ratio=0.06,
        learning_rate=5e-5,
        label_smoothing_factor=0.05,
        logging_steps=100,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_token_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
