import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


def test_selecting_a_preset_fills_the_single_question_box():
    at = AppTest.from_file("app/streamlit_app.py").run()
    # index 0 is "(tự nhập)"; pick the first real preset
    at.selectbox[0].select_index(1).run()

    value = at.text_area[0].value
    assert value.strip(), "preset should fill the single question box"
    assert "A)" in value, "preset embeds the options inline in the one box"

    # the separate A/B/C/D option inputs are gone
    opt_inputs = [ti for ti in at.text_input if ti.label in ("A", "B", "C", "D")]
    assert opt_inputs == []
