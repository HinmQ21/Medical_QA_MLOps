"""Regression: ``streamlit run app/streamlit_app.py`` must resolve ``app.client``.

``streamlit run`` puts the *script's own directory* (``app/``) on ``sys.path[0]``
and does NOT add the project root, so ``from app.client import ...`` raised
``ModuleNotFoundError: No module named 'app'`` until ``streamlit_app.py`` grew a
``sys.path`` bootstrap. pytest's ``pythonpath = ["."]`` masks this (it always puts
the project root on the path), which is why the AppTest-based test did not catch
it. This test reproduces the real launch path in a clean subprocess.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("streamlit")  # only meaningful where the [demo] extra is installed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_streamlit_run_launch_resolves_app_client():
    script = os.path.join(ROOT, "app", "streamlit_app.py")
    code = (
        "import sys, os, runpy, contextlib, io\n"
        "root = os.getcwd()\n"
        f"script = {script!r}\n"
        # Mimic `streamlit run`: only the script's own dir on sys.path; drop the
        # implicit cwd ('') and project root so nothing but the bootstrap can
        # make `app` importable.
        "sys.path = [os.path.dirname(script)] + [p for p in sys.path if p not in ('', root)]\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):\n"
        "    runpy.run_path(script, run_name='__main__')\n"  # bare-mode run: st calls just warn
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "No module named 'app'" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, proc.stderr
