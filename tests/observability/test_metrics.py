from medical_qa_platform.observability import metrics


def test_render_returns_text_and_content_type():
    body, content_type = metrics.render_metrics()
    assert isinstance(body, bytes)
    assert "text/plain" in content_type


def test_observe_request_appears_in_render():
    metrics.observe_request(endpoint="/predict", backend="mock", status="ok", latency_s=0.01)
    body, _ = metrics.render_metrics()
    text = body.decode()
    assert "mqa_requests_total" in text
    assert "mqa_request_latency_seconds" in text


def test_observe_retrieval_no_result():
    metrics.observe_retrieval(latency_s=0.02, no_result=True)
    text = metrics.render_metrics()[0].decode()
    assert "mqa_retrieval_no_result_total" in text


def test_observe_model_appears_in_render():
    metrics.observe_model(backend="mock", latency_s=0.05)
    text = metrics.render_metrics()[0].decode()
    assert "mqa_model_latency_seconds" in text
    assert 'backend="mock"' in text


def test_latency_histograms_have_wide_buckets():
    # Model/total latency on real backends exceeds the prometheus default 10s ceiling,
    # so the histograms must expose buckets up to at least 60s for p95 + alerting to work.
    metrics.observe_model(backend="mock", latency_s=0.05)
    metrics.observe_request(endpoint="/predict", backend="mock", status="ok", latency_s=0.05)
    text = metrics.render_metrics()[0].decode()
    assert 'mqa_model_latency_seconds_bucket{backend="mock",le="60.0"}' in text
    assert 'le="60.0"' in text  # present on mqa_request_latency_seconds too
