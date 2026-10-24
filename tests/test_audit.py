"""Verify audit hash-chain integrity and tamper detection."""

from myagent.audit.log import log, verify


def test_verify_detects_tampering(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path))
    log({"tool": "file_read", "result": {"ok": True}})
    log({"tool": "git_read", "result": {"ok": True}})
    assert verify()

    path = tmp_path / "audit.log"
    content = path.read_text(encoding="utf-8").replace("file_read", "file_write", 1)
    path.write_text(content, encoding="utf-8")
    assert not verify()

