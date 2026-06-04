from pathlib import Path


def test_generator_exposes_embed_model_and_sets_env():
    src = Path("scripts/gen_retrieval_golden.py").read_text()
    assert "--embed-model" in src
    # the generator must export the encoder to the env the baseline tool reads
    assert 'os.environ["KG_ENCODER_MODEL"]' in src or "os.environ['KG_ENCODER_MODEL']" in src
    # manifest records the actual encoder, not a hardcoded large literal
    assert '"encoder_model": args.embed_model' in src
