import pytest

pytest.importorskip("streamlit")


def test_streamlit_app_uses_transcript_builder():
    import app.streamlit_app as ui
    from app.client import build_transcript_blocks

    # the UI must render via the pure builder (so rendering logic stays testable)
    assert ui.build_transcript_blocks is build_transcript_blocks
