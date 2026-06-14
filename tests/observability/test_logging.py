import json
import logging

from medical_qa_platform.observability.logging import JsonFormatter, configure_logging, get_logger


def test_formatter_emits_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.trace_id = "abc"
    parsed = json.loads(formatter.format(record))
    assert parsed["message"] == "hello"
    assert parsed["level"] == "INFO"
    assert parsed["trace_id"] == "abc"


def test_get_logger_returns_logger():
    assert isinstance(get_logger("x"), logging.Logger)


def test_formatter_emits_all_extra_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="medical_qa_platform.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="prediction",
        args=(),
        exc_info=None,
    )
    record.trace_id = "t1"
    record.latency_ms = 12.5
    record.backend = "mock"
    record.tool_call_count = 0
    record.status = "ok"
    parsed = json.loads(formatter.format(record))
    assert parsed["message"] == "prediction"
    assert parsed["trace_id"] == "t1"
    assert parsed["latency_ms"] == 12.5
    assert parsed["backend"] == "mock"
    assert parsed["tool_call_count"] == 0
    assert parsed["status"] == "ok"


def test_formatter_omits_standard_logrecord_noise():
    # Standard LogRecord internals (pathname, lineno, args, ...) must NOT leak into JSON.
    formatter = JsonFormatter()
    record = logging.LogRecord("x", logging.INFO, __file__, 7, "hi", (), None)
    parsed = json.loads(formatter.format(record))
    for noisy in ("pathname", "lineno", "args", "msg", "levelno", "process"):
        assert noisy not in parsed


def test_configure_logging_is_idempotent():
    import logging as _logging

    from medical_qa_platform.observability.logging import JsonFormatter, configure_logging

    root = _logging.getLogger()
    configure_logging()
    configure_logging()
    configure_logging()
    json_handlers = [h for h in root.handlers if isinstance(h.formatter, JsonFormatter)]
    assert len(json_handlers) == 1
