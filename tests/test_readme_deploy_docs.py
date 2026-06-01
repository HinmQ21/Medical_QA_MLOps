from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_readme_documents_plan3_deploy_artifacts():
    text = (ROOT / "README.md").read_text()
    for expected in [
        "make install-deploy-tools",
        "make helm-lint",
        "make helm-template",
        "make helm-dry-run",
        "make docker-build",
        "docker/api.Dockerfile",
        "docker/retrieval.Dockerfile",
        "docker/kserve-mock.Dockerfile",
        "docker/pipeline-init.Dockerfile",
        "KServe",
        "NGINX",
        "aarch64",
        "linux/amd64",
    ]:
        assert expected in text
