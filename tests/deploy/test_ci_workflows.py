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


def test_ci_runs_tests_then_builds_all_images_in_parallel():
    workflow = _workflow("ci.yml")
    jobs = workflow["jobs"]

    # test job runs the suite + helm checks
    test_cmds = "\n".join(s["run"] for s in jobs["test"]["steps"] if "run" in s)
    assert "make install-pipeline" in test_cmds
    assert "make install-deploy-tools" in test_cmds
    assert "make helm-lint" in test_cmds
    assert "make helm-template" in test_cmds
    assert "--cov-fail-under=80" in test_cmds

    # build job is a parallel matrix over the four images, gated on the test job
    build = jobs["build"]
    assert "test" in build["needs"]  # builds only after tests pass
    dockerfiles = {entry["dockerfile"] for entry in build["strategy"]["matrix"]["include"]}
    assert dockerfiles == {
        "docker/api.Dockerfile",
        "docker/retrieval.Dockerfile",
        "docker/pipeline-init.Dockerfile",
        "docker/ui.Dockerfile",
    }
    build_cmds = "\n".join(s["run"] for s in build["steps"] if "run" in s)
    assert "type=gha" in build_cmds  # layer cache
    assert "docker buildx build" in build_cmds


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
