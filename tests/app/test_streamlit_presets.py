import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


def test_selecting_a_preset_populates_question_and_all_option_fields():
    at = AppTest.from_file("app/streamlit_app.py").run()
    # index 0 is "(tự nhập)"; pick the first real preset
    at.selectbox[0].select_index(1).run()

    assert at.text_area[0].value.strip(), "preset should fill the question"

    opt_inputs = [ti for ti in at.text_input if ti.label in ("A", "B", "C", "D")]
    assert len(opt_inputs) == 4
    assert all(ti.value.strip() for ti in opt_inputs), (
        "preset must populate every option field, not just the question",
        [ti.value for ti in opt_inputs],
    )
