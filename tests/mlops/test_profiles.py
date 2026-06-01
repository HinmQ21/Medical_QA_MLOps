from pathlib import Path

import pytest

from mlops.pipelines.profiles import PipelineProfile, load_profile


def test_load_smoke_profile_from_params():
    profile = load_profile("smoke")
    assert isinstance(profile, PipelineProfile)
    assert profile.name == "smoke"
    assert profile.top_k == 2
    assert profile.model_backend == "mock"
    assert profile.model_version == "smoke-dev"
    assert profile.registered_model_name == "medical-qa-smoke"
    assert profile.metrics_path == Path("artifacts/smoke/eval_metrics.json")


def test_unknown_profile_raises_clear_error():
    with pytest.raises(ValueError, match="unknown pipeline profile"):
        load_profile("missing")
