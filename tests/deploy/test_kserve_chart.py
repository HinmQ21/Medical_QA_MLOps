from pathlib import Path

import yaml

from tests.deploy.helm_helpers import find_kind, render_chart


ROOT = Path(__file__).parents[2]


def test_kserve_values_serve_real_qwen_llamacpp_not_a_mock():
    values = yaml.safe_load((ROOT / "deploy/helm/kserve/values.yaml").read_text())
    # RawDeployment has no scale-to-zero without KEDA, so we hold one replica.
    assert values["minReplicas"] == 1
    assert values["deploymentMode"] == "RawDeployment"
    # upstream image, no hand-built mock package
    assert values["image"]["repository"] == "ghcr.io/ggml-org/llama.cpp"
    assert values["image"]["tag"] == "server"
    assert "kserve-mock" not in values["image"]["repository"]
    assert values["model"]["hfRepo"] == "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"
    # fits the 8-vCPU free-trial cap: request 2 / burst to 4
    assert values["resources"]["requests"]["cpu"] == "2"
    assert values["resources"]["requests"]["memory"] == "2Gi"
    assert values["resources"]["limits"]["cpu"] == "4"
    assert values["resources"]["limits"]["memory"] == "3Gi"


def test_kserve_chart_renders_raw_llamacpp_inferenceservice():
    resources = render_chart("kserve")
    isvc = find_kind(resources, "InferenceService", "medical-qa-kserve")
    assert isvc["apiVersion"] == "serving.kserve.io/v1beta1"
    assert (
        isvc["metadata"]["annotations"]["serving.kserve.io/deploymentMode"]
        == "RawDeployment"
    )
    predictor = isvc["spec"]["predictor"]
    assert predictor["minReplicas"] == 1
    container = predictor["containers"][0]
    assert container["image"] == "ghcr.io/ggml-org/llama.cpp:server"
    assert container["ports"][0]["containerPort"] == 8080
    # OpenAI-compatible health gate, not the old /ready mock probe
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["startupProbe"]["httpGet"]["path"] == "/health"
    # serves the Qwen2.5-1.5B GGUF, downloaded at startup
    assert "-hf" in container["args"]
    assert "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M" in container["args"]
    assert "--ctx-size" in container["args"]
    # model cache lives on a writable volume (container image user-agnostic)
    mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
    assert mounts["model-cache"] == "/models"
