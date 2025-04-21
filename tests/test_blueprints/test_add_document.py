# tests/test_blueprints/test_add_document.py
import os
import io
import tempfile
import pytest
from flask import Flask
import requests

# 1) import the blueprint and your DatabaseConfig
from blueprints.add_document_bp import add_document_bp, POST_URL, REQUEST_DELAY
from modules.configuration.config_env import DatabaseConfig
import plugins.ai_modules as ai_mod

# --- FIXTURES --------------------------------------------------------

@pytest.fixture
def app(tmp_path, monkeypatch):
    """
    Create a Flask app, configure it for testing,
    register the blueprint, and monkey‑patch all external calls.
    """
    app = Flask(__name__)
    app.config.update({
        "TESTING": True,
        # point your storage folders at tmp_path
        "DATABASE_DOC": str(tmp_path / "docs"),
        "DATABASE_DIR": str(tmp_path),
        "TEMPORARY_UPLOAD_FILES": str(tmp_path / "tmp"),
        # force no embeddings so we don’t actually call your AI model
        "CURRENT_EMBEDDING_MODEL": "NoEmbeddingModel",
    })

    # make sure the folders exist
    os.makedirs(app.config["DATABASE_DOC"], exist_ok=True)
    os.makedirs(app.config["TEMPORARY_UPLOAD_FILES"], exist_ok=True)

    # 2) Force your DatabaseConfig to use in‑memory SQLite
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    # if you have a separate URL for revision control:
    monkeypatch.setenv("DATABASE_REVISION_URL", "sqlite:///:memory:")

    # 3) Stub out your embedding generator so it never errors
    monkeypatch.setattr(ai_mod, "generate_embedding", lambda text, model: [0.0] * 10)

    # 4) Stub out requests.post so image‐upload always returns success + an image_id
    class DummyResponse:
        status_code = 200
        def json(self):
            return {"image_id": 123}
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: DummyResponse())

    # 5) Register blueprint
    app.register_blueprint(add_document_bp)

    return app

@pytest.fixture
def client(app):
    """Flask test client for sending HTTP requests."""
    return app.test_client()

# --- TESTS -----------------------------------------------------------

def test_add_document_no_files(client):
    """If you POST without any files, you get a 400 + proper JSON message."""
    resp = client.post("/add_document", data={})
    assert resp.status_code == 400
    assert resp.get_json() == {"message": "No files uploaded"}

def test_add_document_with_txt_file(client):
    """POST a single .txt file plus metadata, expect a redirect to upload_success."""
    data = {
        "area": "AreaX",
        "equipment_group": "GroupY",
        "model": "ModelZ",
        "asset_number": "A123",
        "location": "Loc1",
        "site_location": "SiteRoom",
        # you can pass title as form-data; if omitted the code will derive it
        # here we let the code generate title from filename:
        # "title": "My Document",

        # files must be a list of (file‑object, filename)
        "files": (io.BytesIO(b"hello world"), "test.txt"),
    }

    resp = client.post(
        "/add_document",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=False
    )

    # your route ends with `return redirect(url_for('upload_success'))`
    assert resp.status_code == 302
    assert "upload_success" in resp.headers["Location"]

def test_add_document_docx_to_pdf_and_embedding(client, tmp_path, monkeypatch):
    """
    Exercise the .docx → .pdf conversion path by stubbing add_docx_to_db,
    then confirm we still get a redirect.
    """
    # Stub add_docx_to_db to write a dummy PDF file
    from modules.emtacdb.utlity.main_database.database import add_docx_to_db
    def fake_add_docx_to_db(title, fp, position_id):
        # create a dummy PDF in the same folder
        out = fp.replace(".docx", ".pdf")
        with open(out, "wb") as f:
            f.write(b"%PDF-1.4 dummy")
        return True

    monkeypatch.setattr(
        "modules.emtacdb.utlity.main_database.database.add_docx_to_db",
        fake_add_docx_to_db
    )

    data = {
        "area": "A", "equipment_group": "B", "model": "C",
        "asset_number": "D", "location": "E", "site_location": "F",
        # file payload
        "files": (io.BytesIO(b"dummy docx content"), "myfile.docx"),
    }

    resp = client.post(
        "/add_document",
        data=data,
        content_type="multipart/form-data"
    )
    assert resp.status_code == 302
    assert "upload_success" in resp.headers["Location"]
