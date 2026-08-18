import pytest


@pytest.fixture(autouse=True)
def disable_legacy_mirror_state(monkeypatch):
    monkeypatch.setenv("SGLANG_WELM_MTP_LEGACY_MIRROR_STATE", "0")
