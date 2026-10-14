from patent_rag.config import load_dotenv


def test_dotenv_loads_values_without_overriding_existing_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nPRAG_A="from-file"\nPRAG_B=file-b\nnot a pair\n', encoding="utf-8")
    monkeypatch.delenv("PRAG_A", raising=False)
    monkeypatch.setenv("PRAG_B", "from-shell")
    load_dotenv(env)
    import os

    assert os.environ["PRAG_A"] == "from-file"
    assert os.environ["PRAG_B"] == "from-shell"
    monkeypatch.delenv("PRAG_A")


def test_missing_dotenv_is_ignored(tmp_path):
    load_dotenv(tmp_path / "does-not-exist")


def test_dotenv_tolerates_windows_byte_order_mark(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_bytes("﻿PRAG_BOM=yes\n".encode("utf-8"))
    monkeypatch.delenv("PRAG_BOM", raising=False)
    load_dotenv(env)
    import os

    assert os.environ.get("PRAG_BOM") == "yes"
    monkeypatch.delenv("PRAG_BOM")
