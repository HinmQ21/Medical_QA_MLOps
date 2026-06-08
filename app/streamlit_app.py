"""Streamlit demo UI for the Medical QA platform.

Server-side only: the gateway API key is read from the pod environment and
attached to requests by ``app.client``; it never reaches the browser. All
request/response logic lives in ``app.client`` — this module is rendering only.
"""

from __future__ import annotations

import os
import sys

# ``streamlit run app/streamlit_app.py`` puts this file's own directory (app/)
# on sys.path[0], NOT the project root, so ``import app`` would fail with
# ModuleNotFoundError. Prepend the project root (the parent of this file's
# directory) so ``app.client`` resolves the same way it does under pytest
# (pythonpath=["."]) and inside the container (WORKDIR /app). Must run before
# the ``app.client`` import below.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st  # noqa: E402  (after sys.path bootstrap above)

from app.client import (  # noqa: E402  (after sys.path bootstrap above)
    PredictError,
    build_payload,
    build_transcript_blocks,
    fetch_version,
    predict,
)

DEFAULT_BASE_URL = os.environ.get("API_BASE_URL", "http://medical-qa-nginx:8080")
ENV_API_KEY = os.environ.get("API_KEY", "")

PRESETS: dict[str, str] = {
    "Đái tháo đường type 2 — first-line": (
        "Which medication is first-line for type 2 diabetes mellitus?\n"
        "A) Metformin\n"
        "B) Amoxicillin\n"
        "C) Atorvastatin\n"
        "D) Furosemide"
    ),
    "Nhồi máu cơ tim — marker": (
        "Which serum marker is most specific for acute myocardial infarction?\n"
        "A) Troponin I\n"
        "B) Amylase\n"
        "C) ALT\n"
        "D) Creatinine"
    ),
}


def _sidebar() -> tuple[str, str]:
    st.sidebar.header("Kết nối")
    base_url = st.sidebar.text_input("API endpoint", value=DEFAULT_BASE_URL)
    key_override = st.sidebar.text_input(
        "API key (override)",
        value="",
        type="password",
        help="Để trống sẽ dùng key tiêm sẵn trong pod (env API_KEY).",
    )
    api_key = key_override or ENV_API_KEY
    if st.sidebar.button("Kiểm tra kết nối"):
        try:
            info = fetch_version(base_url, api_key)
            st.sidebar.success(
                f"backend={info.get('backend')} · model={info.get('model_version')} · "
                f"contract={info.get('contract_version')}"
            )
        except PredictError as exc:
            st.sidebar.error(str(exc))
    return base_url, api_key


def main() -> None:
    st.set_page_config(page_title="Medical QA Demo", page_icon="🩺")
    st.title("🩺 Medical QA — Demo")
    st.caption("Nhập câu hỏi trắc nghiệm y khoa; trợ lý suy luận và tự tra cứu tri thức (KG) khi cần.")

    base_url, api_key = _sidebar()

    preset = st.selectbox("Câu hỏi mẫu", ["(tự nhập)", *PRESETS])
    seed = PRESETS.get(preset, "")

    question = st.text_area(
        "Câu hỏi (kèm phương án)",
        value=seed,
        height=200,
        help="Dán nguyên câu hỏi trắc nghiệm kèm các phương án, ví dụ 'A) ... B) ...'.",
    )

    if not st.button("Chẩn đoán", type="primary"):
        return

    try:
        payload = build_payload(question)
    except ValueError as exc:
        st.warning(str(exc))
        return

    with st.spinner("Đang truy hồi tri thức và suy luận..."):
        try:
            result = predict(base_url, api_key, payload)
        except PredictError as exc:
            st.error(str(exc))
            return

    # The headline is the per-turn transcript now; the parsed letter is a small badge.
    if result.answer:
        st.caption(f"Đáp án (parse tự động): **{result.answer}**")

    st.subheader("Diễn tiến suy luận")
    blocks = build_transcript_blocks(result.trace or [])
    if blocks:
        for label, body in blocks:
            st.markdown(f"**{label}**")
            st.code(body or "(rỗng)")
    else:
        # No trace (older API / single-turn): fall back to the raw final output.
        st.code(result.raw_output or "(rỗng)")

    with st.expander(f"Bằng chứng KG ({len(result.evidence)})", expanded=False):
        if result.evidence:
            for i, evidence in enumerate(result.evidence, 1):
                st.markdown(f"{i}. {evidence}")
        else:
            st.write("(không có bằng chứng nào được truy hồi)")

    st.caption(
        f"backend={result.backend} · model={result.model_version} · "
        f"contract={result.contract_version} · latency={result.latency_ms:.0f}ms · "
        f"trace={result.trace_id}"
    )


if __name__ == "__main__":
    main()
