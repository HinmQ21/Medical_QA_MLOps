def test_makefile_has_full_targets():
    text = open("Makefile").read()
    for target in ["full-pipeline:", "full-pipeline-dry-run:", "smoke-full:"]:
        assert target in text


def test_full_dry_run_target_uses_run_full_dry_run():
    text = open("Makefile").read()
    assert "mlops.pipelines.run_full --profile full --dry-run" in text


def test_gitignore_excludes_generated_dirs():
    text = open(".gitignore").read()
    assert "artifacts/" in text
    assert "mlruns/" in text
