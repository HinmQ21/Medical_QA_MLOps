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


def test_deploy_workflow_auto_runs_after_ci_and_guards_cluster():
    text = (ROOT / ".github/workflows/deploy.yml").read_text()
    workflow = _workflow("deploy.yml")
    triggers = workflow.get("on", workflow.get(True))
    assert "workflow_run" in triggers
    assert triggers["workflow_run"]["workflows"] == ["CI"]
    assert "main" in triggers["workflow_run"]["branches"]
    commands = _all_run_commands(workflow)
    assert "gcloud container clusters describe medical-qa" in commands
    assert "up=false" in commands  # skip-green path when the cluster is down
    assert "scripts/cloud/deploy.sh" in commands
