from pathlib import Path

import yaml

from tests.deploy.helm_helpers import find_kind, render_chart


ROOT = Path(__file__).parents[2]


def test_kserve_values_default_to_scale_to_zero():
    values = yaml.safe_load((ROOT / "deploy/helm/kserve/values.yaml").read_text())
    assert values["minReplicas"] == 0
    assert values["image"]["repository"].endswith("medical-qa-kserve-mock")


def test_kserve_chart_renders_inferenceservice_with_min_replicas_zero():
    resources = render_chart("kserve")
    service = find_kind(resources, "InferenceService", "medical-qa-kserve")
    assert service["apiVersion"] == "serving.kserve.io/v1beta1"
    predictor = service["spec"]["predictor"]
    assert predictor["minReplicas"] == 0
    container = predictor["containers"][0]
    assert container["image"].endswith("medical-qa-kserve-mock:latest")
    assert container["ports"][0]["containerPort"] == 8080
