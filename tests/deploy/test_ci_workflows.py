from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def _all_run_commands(workflow: dict) -> str:
    commands = []
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                commands.append(step["run"])
    return "\n".join(commands)


def test_ci_workflow_runs_tests_helm_checks_and_builds_images():
    workflow = _workflow("ci.yml")
    commands = _all_run_commands(workflow)
    assert "make install-pipeline" in commands
    assert "make install-deploy-tools" in commands
    assert "make helm-lint" in commands
    assert "make helm-template" in commands
    assert "--cov-fail-under=80" in commands
    assert "docker/api.Dockerfile" in commands
    assert "docker/retrieval.Dockerfile" in commands
    assert "docker/kserve-mock.Dockerfile" in commands
    assert "docker/pipeline-init.Dockerfile" in commands


def test_deploy_workflow_is_manual_dispatch_skeleton():
    workflow = _workflow("deploy.yml")
    triggers = workflow.get("on", workflow.get(True))
    assert "workflow_dispatch" in triggers
    commands = _all_run_commands(workflow)
    assert "helm upgrade --install medical-qa-api deploy/helm/api" in commands
    assert "helm upgrade --install medical-qa-retrieval deploy/helm/retrieval" in commands
    assert "helm upgrade --install medical-qa-nginx deploy/helm/nginx" in commands
    assert "helm upgrade --install medical-qa-kserve deploy/helm/kserve" in commands
    assert "/health" in commands
    assert "/predict" in commands
