from tests.deploy.helm_helpers import find_kind, render_chart


def test_retrieval_renders_workload_identity_sa_and_real_bucket():
    resources = render_chart(
        "retrieval",
        set_values={
            "serviceAccount.gcpServiceAccount": "medical-qa-retrieval@demo.iam.gserviceaccount.com",
            "dvc.bucket": "demo-medical-qa-dvc",
        },
    )
    sa = find_kind(resources, "ServiceAccount", "medical-qa-retrieval")
    assert (
        sa["metadata"]["annotations"]["iam.gke.io/gcp-service-account"]
        == "medical-qa-retrieval@demo.iam.gserviceaccount.com"
    )
    deployment = find_kind(resources, "Deployment", "medical-qa-retrieval")
    assert deployment["spec"]["template"]["spec"]["serviceAccountName"] == "medical-qa-retrieval"
    config = find_kind(resources, "ConfigMap", "medical-qa-retrieval-dvc")
    assert "gs://demo-medical-qa-dvc/dvc" in config["data"]["dvc-config"]


def test_retrieval_base_render_still_has_sa_without_annotation():
    resources = render_chart("retrieval")
    sa = find_kind(resources, "ServiceAccount", "medical-qa-retrieval")
    assert "annotations" not in sa.get("metadata", {})
